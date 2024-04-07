import torch
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from prettytable import PrettyTable 

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

# mytable for PrettyTable
myTable = PrettyTable(["Learning rate", "Blues", "Country", "Metal", "Pop", "Rap", "Rock"])

# import model, and set it up
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = XLMRobertaForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

# load test csv in data frame
test_df = pd.read_csv('test_data.csv')

# use cuda or cpu
model.to(device)

# input csv in df
train_df = pd.read_csv('clean_lyrics.csv')
def adjust_lr(lrs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrs, weight_decay=0.01)
    return optimizer

# TODO: remove epochs and implement early stoppings keep epoch count
epochs = 3
max_seq_length = 128

# splitting train, validation, and test data
train_df, val_df = train_test_split(train_df, test_size= 0.2, random_state=42)



def train_and_test_model(optimizer):
    for epoch in range(epochs):
        model.train()
        for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Epoch {epoch} - Training"):
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
            for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Epoch {epoch} - Validation"):
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

        f1_scores = f1_score(true_labels, predicted_labels, average=None)
        print(f'F1 Scores for each class: {f1_scores}')
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Validation Accuracy: {accuracy}')
    
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
        f1_scores = list(f1_scores)
        f1_scores.insert(0, optimizer.param_groups[0]['lr'])
        myTable.add_row(f1_scores)
        print(myTable)
        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f'Test Accuracy: {accuracy}')


def main():
    learning_rates = [1e-3, 1e-5, 1e-2, 5e-5]

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        optimizer = adjust_lr(lr)
        train_and_test_model(optimizer)

if __name__ == "__main__":
    main()