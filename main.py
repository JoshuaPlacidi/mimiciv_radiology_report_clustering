import pandas as pd
from data import get_dataloader
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ClinicalBERT
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle

def approximate_pca(dataloader: DataLoader, num_components: int, num_batches: int):

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalBERT().to(device)

    pb = tqdm(dataloader)

    ipca = IncrementalPCA(n_components=num_components, batch_size=dataloader.batch_size)

    for idx, batch in enumerate(pb):

        # Tokenize
        input_ids, attention_mask = model.tokenize(batch["text"])

        # Encode
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask)
        
        ipca.partial_fit(output.detach().cpu().numpy())

        with open("pca.pkl", "wb") as f:
            pickle.dump(ipca, f)

        if idx >= num_batches:
            break

    return ipca

def encode(dataloader: DataLoader, num_batches: int, save=False):

    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClinicalBERT().to(device)

    with open("pca.pkl", "rb") as f:
        ipca = pickle.load(f)

    all_note_ids = []
    all_embeddings = []

    pb = tqdm(dataloader)

    for idx, batch in enumerate(pb):
        # Tokenize
        input_ids, attention_mask = model.tokenize(batch["text"])

        # Encode
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        output = model(input_ids, attention_mask)

        output = ipca.transform(output.detach().cpu().numpy())

        # Collect results
        all_note_ids.extend(batch["note_id"])
        all_embeddings.append(output.detach().cpu().numpy())

        if save and idx % 100 == 0:

            # Stack all embeddings
            stacked_embeddings = np.vstack(all_embeddings)
            
            # Save results
            np.save("note_ids.npy", np.array(all_note_ids))
            np.save("embeddings.npy", stacked_embeddings)

            pb.set_postfix(embeddings_saved = stacked_embeddings.shape[0])

        if idx >= num_batches:
            break

    # Stack all embeddings
    stacked_embeddings = np.vstack(all_embeddings)

    if save:
        # Save results
        np.save("note_ids.npy", np.array(all_note_ids))
        np.save("embeddings.npy", stacked_embeddings)
            
if __name__ == "__main__":
    import argparse
    import pickle
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    # Fit PCA
    print("Fitting PCA")
    approximate_pca(get_dataloader(args.csv_path, batch_size=50, shuffle=True), num_components=50, num_batches=1000)

    # Encode
    print("Encoding")
    encode(get_dataloader(args.csv_path, batch_size=50, shuffle=False), save=True)