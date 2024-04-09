import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
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
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)



# use cuda or cpu
model.to(device)

# input csv in df
train_df = pd.read_csv('clean_lyrics.csv')

# TODO: remove epochs and implement early stoppings keep epoch count
epochs = 4
max_seq_length = 512

# splitting train, validation, and test data
train_df, test_df = train_test_split(train_df, test_size= 0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

def train_and_test_model(learning_rates):
    for lr in learning_rates:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True)
        # this 'freezes' pretrained layers 
        for param in model.base_model.parameters():
            param.requires_grad = False
        
        for epoch in range(epochs):
            model.train()
            train_losses = []

            for index, row in tqdm(train_df.iterrows(), total=len(train_df), desc=f"Training with {lr} learning rate -> {epoch+1}/{epochs}"):
                lyrics = row['Lyrics']
                genre = row['Genre'].capitalize()
                label_id = label2id.get(genre)
                inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
                inputs = {key: val.to(device) for key, val in inputs.items()}
                outputs = model(**inputs, labels=torch.tensor([label_id]).to(device))
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(loss.item())
            # this unfreezes them
            if epoch == 3:
                for param in model.base_model.parameters():
                    param.requires_grad = True
            
            model.eval()
            true_labels = []
            predicted_labels = []

            with torch.no_grad():
                for index, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Validation for {lr} learning rate -> {epoch+1}/{epochs}"):
                    lyrics = row['Lyrics']
                    genre = row['Genre'].capitalize()
                    label_id = label2id.get(genre)
                    inputs = tokenizer(lyrics, return_tensors='pt', max_length=max_seq_length, truncation=True, padding=True)
                    inputs = {key: val.to(device) for key, val in inputs.items()}
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted = torch.softmax(logits, 1)
                    predicted_label_id = torch.argmax(predicted, axis=1).item()
                    true_labels.append(label_id)
                    predicted_labels.append(predicted_label_id)

            val_f1 = f1_score(true_labels, predicted_labels, average='macro')
            print(f'\nValidation F1 Score: {val_f1}\n')

            # adjust learning rate based on validation performance
            scheduler.step(val_f1)

        # test fine tuned model
        model.eval()
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Testing for {lr} learning rate"):
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
            accuracy = accuracy_score(true_labels, predicted_labels)
            print(f'\nTest Accuracy: {accuracy}\n')
            myTable.add_row(f1_scores)

    print(myTable)


def main():
    learning_rates = [1e-2, 1e-3, 1e-5, 5e-5]
    train_and_test_model(learning_rates)

if __name__ == "__main__":
    main()