import torch
from torch import nn
from torchdiffeq import odeint

class NeuralCDEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(NeuralCDEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t, z):
        return self.net(z)

class NeuralFactorCDE(nn.Module):
    def __init__(self, hidden_dim, input_dim, factor_dim, output_dim, device):
        super(NeuralFactorCDE, self).__init__()
        self.factor_dim = factor_dim
        self.device = device
        self.func = NeuralCDEFunc(hidden_dim).to(device)
        self.linear_factors = nn.Linear(hidden_dim, factor_dim).to(device)
        self.linear_output = nn.Linear(factor_dim, output_dim).to(device)
    
    def forward(self, X, time):
        batch_size, seq_len, input_dim = X.size()
        z0 = torch.zeros(batch_size, self.func.net[0].in_features).to(self.device)
        z_T = odeint(self.func, z0, time, method='rk4')[-1]  # (batch_size, hidden_dim)
        factors = self.linear_factors(z_T)  # (batch_size, factor_dim)
        out = self.linear_output(factors)
        return out, factors
