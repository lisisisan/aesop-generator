import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# -----------------------
# 1. Загружаем обработанный текст
# -----------------------
with open("data/aesop/data_processed.txt") as f:
    text = f.read()

tokens = text.split()
vocab = sorted(set(tokens))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for i, w in enumerate(vocab)}

seq_len = 50
X, y = [], []
for i in range(len(tokens) - seq_len):
    seq = tokens[i:i+seq_len]
    target = tokens[i+seq_len]
    X.append([word2idx[w] for w in seq])
    y.append(word2idx[target])

X = torch.tensor(X)
y = torch.tensor(y)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------
# 2. Модель
# -----------------------
class AesopLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])
        return out, hidden

model = AesopLSTM(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# -----------------------
# 3. TensorBoard
# -----------------------
os.makedirs("logs/tensorboard", exist_ok=True)
writer = SummaryWriter(log_dir="logs/tensorboard")

# -----------------------
# 4. Обучение с логированием loss
# -----------------------
losses = []

for epoch in range(20):
    total_loss = 0
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output, _ = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # логируем в TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)

writer.close()

# -----------------------
# 5. Сохраняем модель
# -----------------------
os.makedirs("data/aesop", exist_ok=True)
torch.save({
    'model_state_dict': model.state_dict(),
    'word2idx': word2idx,
    'idx2word': idx2word
}, "data/aesop/lstm_aesop.pt")

# -----------------------
# 6. Сохраняем график loss в PNG
# -----------------------
os.makedirs("logs", exist_ok=True)
plt.figure(figsize=(8,5))
plt.plot(range(1, len(losses)+1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('logs/training_loss.png')
plt.show()
