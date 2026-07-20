import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, block_size):
        self.block_size = block_size
        
        # Load the trained tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Read the raw text data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Encode the entire text
        print("Tokenizing the dataset...")
        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded.ids, dtype=torch.long)
        print(f"Dataset tokenized. Total tokens: {len(self.data)}")

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloader(file_path, tokenizer_path, block_size, batch_size, shuffle=True):
    dataset = TextDataset(file_path, tokenizer_path, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset.tokenizer
