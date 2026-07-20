import torch
import torch.optim as optim
from model import CustomLLM
from tokenizers import Tokenizer
import os

def online_finetune(text_data, max_iters=20):
    """
    Perform a few steps of online learning on the newly provided text data.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Hyperparameters must match model
    block_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 2
    dropout = 0.2
    learning_rate = 5e-4 # Smaller learning rate for fine-tuning
    
    tokenizer_file = "data/ai_subword_tokenizer.json"
    model_path = "data/model.pt"
    
    if not os.path.exists(model_path):
        print("Base model not found. Please train first.")
        return
        
    tokenizer = Tokenizer.from_file(tokenizer_file)
    vocab_size = tokenizer.get_vocab_size()
    
    # Load existing model
    model = CustomLLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model = model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Tokenize the new data
    encoded = tokenizer.encode(text_data)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    
    if len(data) <= block_size:
        print("Not enough data to train on.")
        return
        
    print(f"Starting online fine-tuning for {max_iters} steps...")
    
    # Simple training loop for a few iterations
    batch_size = 8
    
    for iter_num in range(max_iters):
        # Sample a batch
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        
        x, y = x.to(device), y.to(device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    print(f"Fine-tuning complete. Final loss: {loss.item():.4f}")
    
    # Save the updated model
    torch.save(model.state_dict(), model_path)
    print("Updated model weights saved.")
