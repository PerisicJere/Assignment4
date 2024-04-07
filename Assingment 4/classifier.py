import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, AdamW
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
import matplotlib.pyplot as plt

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
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-8, weight_decay=0.01)

# TODO: remove epochs and implement early stoppings keep epoch count
epochs = 1
max_seq_length = 128
# splitting train, validation, and test data
train_df, val_df = train_test_split(train_df, test_size= 0.1, random_state=42)
# parameters for early stopping
patience = 5 
best_val_accuracy = 0.0
count_without_improvement = 0

while count_without_improvement < patience:
    model.train()
    for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epochs} - Training"):
        lyrics = row['Lyrics']
        genre = row['Genre'].capitalize()
        label_id = label2id.get(genre)
        if label_id is None:
            print(f"Label '{genre}' not found in label2id dictionary.")
            continue
        if not isinstance(lyrics, str):
            print(f"Lyrics is not a string: {row['Song Title']}")
            continue
        inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs, labels=torch.tensor([label_id]).to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epochs} - Validation"):
            lyrics = row['Lyrics']
            genre = row['Genre'].capitalize()
            label_id = label2id.get(genre)
            if label_id is None:
                print(f"Label '{genre}' not found in label2id dictionary.")
                continue
            if not isinstance(lyrics, str):
                print(f"Lyrics is not a string: {row['Song Title']}")
                continue
            inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted = torch.softmax(logits, 1)
            predicted_label_id = torch.argmax(predicted, axis=1).item()
            true_labels.append(label_id)
            predicted_labels.append(predicted_label_id)
    epochs += 1
    f1_scores = f1_score(true_labels, predicted_labels, average=None)
    print(f'F1 Scores for each class: {f1_scores}')
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Validation Accuracy: {accuracy}')

    if accuracy > best_val_accuracy:
        best_val_accuracy = accuracy
        count_without_improvement = 0
        torch.save(model.state_dict(), 'High_validation.pth')
    else:
        count_without_improvement += 1
        if count_without_improvement >= patience:
            print(f'Model did improve in patience cycle.')
            break

# load test csv in data frame
test_df = pd.read_csv('test_data.csv')
# loading best model from early stopping
model.load_state_dict(torch.load('High_validation.pth'))

# test fine tuned model
model.eval()
true_labels = []
predicted_labels = []

with torch.no_grad():
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Testing"):
        lyrics = row['Lyrics']
        genre = row['Genre'].capitalize()
        label_id = label2id.get(genre)
        if label_id is None:
            print(f"Label {genre} not found")
            continue
        if not isinstance(lyrics, str):
            print(f'Lyrics is of different type {lyrics}')
            continue
        inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True,padding=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.softmax(logits, 1)
        predicted_label_id = torch.argmax(predicted, axis=1).item()
        true_labels.append(label_id)
        predicted_labels.append(predicted_label_id)

    f1_scores = f1_score(true_labels, predicted_labels, average=None)
    print("F1 Scores for each class: {f1_scores}")
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Test Accuracy: {accuracy}')


