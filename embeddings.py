import pandas as pd
import torch
import re
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")
model = model.eval()

def preprocess_sequence(seq):
    seq = re.sub(r"[UZOB]", "X", seq)
    return ' '.join(list(seq))

def get_embedding(sequence):
    with torch.no_grad():
        encoded_input = tokenizer(sequence, return_tensors='pt')
        output = model(**encoded_input)
        embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def process_file(filename, label, limit=500):
    df = pd.read_csv(filename)
    embeddings = []

    if label == "negative":
        limit = 1000
        label_value = 0
    else:
        label_value = 1

    for seq in tqdm(df["Name"].iloc[:limit], desc=f"Processando {label}"):
        seq = preprocess_sequence(seq)
        emb = get_embedding(seq)
        embeddings.append((label_value, emb.tolist()))
    
    return embeddings

positive_allergic_embeddings = process_file("positive_allergic.csv", "positive_allergic")
positive_infectious_embeddings = process_file("positive_infectious.csv", "positive_infectious")
negative_embeddings = process_file("negatives.csv", "negative")

with open("embeddings.txt", "w") as f:
    for label, emb in positive_allergic_embeddings + positive_infectious_embeddings + negative_embeddings:
        emb_str = ' '.join(map(str, emb))
        f.write(f"{label} {emb_str}\n")

