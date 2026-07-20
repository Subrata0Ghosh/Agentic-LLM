import torch
import torch.optim as optim
from model import CustomLLM
from dataset import get_dataloader

# Hyperparameters
batch_size = 32
block_size = 64
max_iters = 500
eval_interval = 50
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configuration
n_embd = 128
n_head = 4
n_layer = 2
dropout = 0.2

print(f"Using device: {device}")

# Prepare data
text_file = "data/ai_corpus_cleaned.txt"
tokenizer_file = "data/ai_subword_tokenizer.json"

dataloader, tokenizer = get_dataloader(
    file_path=text_file,
    tokenizer_path=tokenizer_file,
    block_size=block_size,
    batch_size=batch_size
)

vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Initialize model
model = CustomLLM(
    vocab_size=vocab_size,
    n_embd=n_embd,
    block_size=block_size,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)
model = model.to(device)

# Print the number of parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {n_params/1e6:.2f}M")

# Create a PyTorch optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

print("Starting training...")
model.train()
iterator = iter(dataloader)

for iter_num in range(max_iters):
    try:
        xb, yb = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        xb, yb = next(iterator)
        
    xb, yb = xb.to(device), yb.to(device)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print loss periodically
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        print(f"Step {iter_num}: Loss {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "data/model.pt")
print("Model saved to data/model.pt")
