from global_import import torch
import torch.nn as nn
import torch.optim as optim

class MacroLSTM(nn.Module):
    def __init__(self, 
                 n_isos: int, n_regions: int, n_incomes: int, 
                 n_continuous: int, 
                 hidden_size: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.2,
                 lr: float = 0.001, device: str = 'cpu'):
        """
        LSTM-Bidirectional LSTM model for Macro Analysis.
        """
        super(MacroLSTM, self).__init__()
        
        self.device = device
        
        # Dimensions
        self.iso_dim = int(n_isos**0.5)
        self.region_dim = int(n_regions**0.5)
        self.income_dim = int(n_incomes**0.5)
        
        # Embeddings
        self.emb_iso = nn.Embedding(n_isos, self.iso_dim)
        self.emb_region = nn.Embedding(n_regions, self.region_dim)
        self.emb_income = nn.Embedding(n_incomes, self.income_dim)
        
        # Input Size calculation
        # Continuous features + Embeddings
        input_size = n_continuous + self.iso_dim + self.region_dim + self.income_dim
        
        # LSTM Layers
        # 1. Forward LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False)
        
        # 2. Bidirectional LSTM
        # Input to BiLSTM is expected to be hidden_size
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Output Head
        # BiLSTM outputs 2 * hidden_size per step
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_continuous) 
        )
        
        # Optimizer Integration
        self.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, iso: torch.Tensor, region: torch.Tensor, income: torch.Tensor, continuous_data: torch.Tensor):
        """
        iso: (Batch, Seq)
        region: (Batch, Seq)
        income: (Batch, Seq)
        continuous_data: (Batch, Seq, n_continuous)
        """
        # Ensure inputs are on correct device
        if iso.device != self.device: iso = iso.to(self.device)
        if region.device != self.device: region = region.to(self.device)
        if income.device != self.device: income = income.to(self.device)
        if continuous_data.device != self.device: continuous_data = continuous_data.to(self.device)

        # Embeddings
        e_iso = self.emb_iso(iso)          # (Batch, Seq, iso_dim)
        e_reg = self.emb_region(region)    # (Batch, Seq, reg_dim)
        e_inc = self.emb_income(income)    # (Batch, Seq, inc_dim)
        
        # Concatenate Features
        # x: (Batch, Seq, Input_Size)
        x = torch.cat([e_iso, e_reg, e_inc, continuous_data], dim=2)
        
        # LSTM 1
        out1, _ = self.lstm1(x) # (Batch, Seq, Hidden)
        
        # LSTM 2 (Bidirectional)
        out2, _ = self.lstm2(out1) # (Batch, Seq, Hidden*2)
        
        # Head (Reconstruction)
        # Apply to every time step
        pred = self.head(out2) # (Batch, Seq, n_continuous)
        
        return pred

    def backward_step(self, loss: torch.Tensor):
        """
        Executes the backward pass and optimizer step.
        """
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, x_iso: torch.Tensor, x_reg: torch.Tensor, x_inc: torch.Tensor, x_cont: torch.Tensor, epochs=10):
        """
        Trains the model to reconstruct the input.
        """
        self.train()
        
        # Target: Fit all indicators based on all indicators (Reconstruction)
        # In future, might be Next Step Prediction (Shifted)
        target = x_cont.to(self.device)
        
        print(f"Starting training on {self.device}...")
        
        for epoch in range(epochs):
            # Forward
            outputs = self(x_iso, x_reg, x_inc, x_cont)
            
            # Loss
            loss = self.criterion(outputs, target)
            
            # Backward & Step
            self.backward_step(loss)
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
        print("Training complete.")
        return self
