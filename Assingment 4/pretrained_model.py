import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from prettytable import PrettyTable

label2id = {"Blues": 0, "Country": 1, "Metal": 2, "Pop": 3, "Rap": 4, "Rock": 5}

model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, label2id=label2id)

test_data = pd.read_csv("clean_lyrics.csv")
test_sentences = test_data["Lyrics"].tolist()
true_labels = test_data["Genre"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 8
num_batches = (len(test_sentences) - 1) // batch_size + 1

predicted_labels = []

for i in tqdm(range(num_batches), desc="Predicting"):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, len(test_sentences))
    batch_sentences = test_sentences[start_idx:end_idx]
    
    tokenized_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    tokenized_inputs = {key: value.to(device) for key, value in tokenized_inputs.items()}
    
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    
    logits = outputs.logits.detach().cpu()
    
    predicted_probabilities = torch.softmax(logits, dim=1)
    predicted_batch_labels = torch.argmax(predicted_probabilities, dim=1).tolist()
    predicted_labels.extend(predicted_batch_labels)

predicted_labels_string = [list(label2id.keys())[label] for label in predicted_labels]

f1_scores = {}
for genre in label2id.keys():
    true_genre_labels = [label == genre for label in true_labels]
    predicted_genre_labels = [label == genre for label in predicted_labels_string]
    f1_scores[genre.capitalize()] = f1_score(true_genre_labels, predicted_genre_labels)

print("F1 Scores for Each Genre:")
f1_table = PrettyTable(["Genre", "F1 Score"])
for genre, f1_score in f1_scores.items():
    f1_table.add_row([genre, f1_score])
print(f1_table)