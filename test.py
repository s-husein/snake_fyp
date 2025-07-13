import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # output a single value

    def forward(self, x):
        out, _ = self.lstm(x)  # out: (B, T, hidden)
        return self.fc(out[:, -1])


class SimpleDataset(Dataset):
    def __init__(self, X, seq_len = 3):
        self.data = X
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data) - 1
    def __getitem__(self, idx):
      seq = [self.data[0]]*self.seq_len
      for i in range(self.seq_len):
            if(idx - i) <= 0:
                  continue
            seq[(self.seq_len - 1) - i] = self.data[idx - i]
      _label = self.data[idx+1]
      return torch.tensor(seq, dtype=torch.float32).unsqueeze(-1), torch.tensor(_label, dtype=torch.float32).unsqueeze(-1)
    

x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140])


ds = SimpleDataset(x, 4)


ds_loader = DataLoader(SimpleDataset(x, 2), batch_size=5, shuffle=True)

model = LSTMModel(hidden_size=512, num_layers=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1001):
    for xb, yb in ds_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


last_seq = torch.tensor([[110, 120, 130, 140]], dtype=torch.float32).unsqueeze(-1)  # shape (1, 3, 1)
with torch.no_grad():
    prediction = model(last_seq)
    print("Predicted next value:", prediction.item())


    