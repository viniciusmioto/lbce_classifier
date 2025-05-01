import pandas as pd
from tqdm import tqdm

def process_file(filename, label, limit=500):
    df = pd.read_csv(filename)
    embeddings = []

    if label == "negative":
        limit = 1000
        namefile = "neg.txt"
    else:
        namefile = "pos.txt"

    for seq in tqdm(df["Name"].iloc[:limit], desc=f"Processando {label}"):
        with open(f"{namefile}", "a") as f:
            f.write(f"{seq}\n")


positive_allergic_embeddings = process_file("positive_allergic.csv", "positive_allergic")
positive_infectious_embeddings = process_file("positive_infectious.csv", "positive_infectious")
negative_embeddings = process_file("negatives.csv", "negative")
