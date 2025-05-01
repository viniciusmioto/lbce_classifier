import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator

class MLPModel(BaseEstimator):
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3, lr=1e-3, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        layers = []
        last_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers).to(self.device)

    def train(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.model(X).cpu().numpy()
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.build_model()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
