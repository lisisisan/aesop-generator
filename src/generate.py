import torch

from train import AesopLSTM  # если train.py в той же папке, иначе перенеси класс

checkpoint = torch.load("data/aesop/lstm_aesop.pt")
word2idx = checkpoint['word2idx']
idx2word = checkpoint['idx2word']

model = AesopLSTM(len(word2idx))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

seq_len = 50  # должно совпадать с train.py

def generate_text(start_words, length=100, temperature=1.0):
    words = start_words.split()
    hidden = None
    for _ in range(length):
        seq = torch.tensor([[word2idx.get(w, 0) for w in words[-seq_len:]]])
        with torch.no_grad():
            output, hidden = model(seq, hidden)
            probs = torch.softmax(output/temperature, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            words.append(idx2word[idx])
    return ' '.join(words)

print(generate_text("the fox", length=50))
