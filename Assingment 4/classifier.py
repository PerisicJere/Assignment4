import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, AdamW
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Checking if we have cuda available, if so use cuda, else use cpu
device = torch.cuda.is_available()
if device is True:
    curr_dev = torch.cuda.current_device()
    device = "cuda"
    print(f'Using -> {torch.cuda.get_device_name(curr_dev)}')
else:
    device = "cpu"
    print(f'Using -> cpu')

# labels (difference between multi label and single is output)
label2id = {"Blues": 0, "Country": 1, "Metal": 2, "Pop": 3, "Rap": 4, "Rock": 5}
id2label = {0: "Blues", 1: "Country", 2: "Metal", 3: "Pop", 4: "Rap", 5: "Rock"}

# Import model, and set it up
model_name = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

# use cuda or cpu
model.to(device)

# input csv in df
train_df = pd.read_csv('clean_lyrics.csv')

optimizer = AdamW(model.parameters(), lr=1e-5)
epochs = 3
max_seq_length = 128
# splitting train, validation, and test data
train_df, test_df = train_test_split(train_df, test_size= 0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.11, random_state=42)

for epoch in range(epochs):
    model.train()
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch+1}/{epochs} - Training"):
        lyrics = row['Lyrics']
        genre = row['Genre'].capitalize()
        label_id = label2id.get(genre)
        if label_id is None:
            print(f"Label '{genre}' not found in label2id dictionary.")
            continue
        if not isinstance(lyrics, str):
            print(f"Lyrics is not a string: {lyrics}")
            continue
        inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=torch.tensor([label_id]).to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    correct_pred = 0
    total_pred = 0

    with torch.no_grad():
        for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epoch+1}/{epochs} - Validation"):
            lyrics = row['Lyrics']
            genre = row['Genre'].capitalize()
            label_id = label2id.get(genre)
            if label_id is None:
                print(f"Label '{genre}' not found in label2id dictionary.")
                continue
            if not isinstance(lyrics, str):
                print(f"Lyrics is not a string: {lyrics}")
                continue
            inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            _, predicted = torch.max(logits, 1)
            total_pred += 1
            if predicted == label_id:
                correct_pred += 1

    val_accuracy = correct_pred/total_pred
    print(f'Validation Accuracy: {val_accuracy}')





