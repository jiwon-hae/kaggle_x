import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def data_frame_to_tensor(df, dtype=torch.float32):
    return torch.tensor(df, dtype=dtype)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(input_dim, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def forward(self, x):
        return self.linear(x)

    def fit(self, num_epochs, x_train, y_train):
        for epoch in tqdm(range(num_epochs)):
            self.train()
            y_pred = self(x_train)
            loss = self.criterion(y_pred, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    def evaluate(self, x_train, y_train, x_val, y_val):
        print('evaluate')
        self.eval()
        with torch.no_grad():
            y_pred_train = self(x_train)
            train_loss = self.criterion(y_pred_train, y_train)

            y_pred_val = self(x_val)
            val_loss = self.criterion(y_pred_val, y_val).item()

            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
