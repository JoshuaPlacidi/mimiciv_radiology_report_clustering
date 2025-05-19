import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import os

class RadiologyNoteDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        x = {
            "note_id": row["note_id"],
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "text": row["text"]
        }
        
        return x
    
def get_dataloader(csv_path: str, batch_size: int = 16, shuffle: bool = True):
    dataset = RadiologyNoteDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":

    csv_path = os.path.expanduser("~/data/mimic/mimiciv-note/2.2/radiology.csv")
    dataset = RadiologyNoteDataset(csv_path)
    print(dataset[0])