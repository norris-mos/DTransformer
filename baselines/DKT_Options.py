import torch
import torch.nn as nn
import torch.nn.functional as F


class DKT_Options(nn.Module):
    def __init__(self, n_questions, d_model=100):
        super().__init__()
        self.n_questions = n_questions
        self.d_model = d_model
        # Input encoding: question + option (1-4) 
        # Input space: q_id + option * n_questions, so total vocab size is n_questions * 4 + 1
        self.rnn = nn.LSTM(n_questions * 4 + 1, d_model, batch_first=True)
        # Output layer predicts option probabilities (4 classes) for each question
        self.fc = nn.Linear(d_model, n_questions * 4)  # 4 options per question

    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.d_model).to(self.device())
        c0 = torch.zeros(1, x.size(0), self.d_model).to(self.device())
        pad_start = torch.zeros(x.size(0), 1, x.size(2)).to(self.device())
        out, _ = self.rnn(
            torch.cat([pad_start, x], dim=1),
            (h0, c0),
        )
        # Reshape output to [batch, seq_len, n_questions, 4]
        out_reshaped = self.fc(out)[:, :-1, :].view(x.size(0), x.size(1), self.n_questions, 4)
        # Apply softmax over the 4 options for each question
        res = F.softmax(out_reshaped, dim=-1)
        return res

    def predict(self, q, s, pid=None, n=1):
        assert pid is None, "DKT_Options does not support pid input"
        q_masked = q.masked_fill(q < 0, 0)
        s_masked = s.masked_fill(s < 0, 0)
        
        # Create input encoding: q_id + (option - 1) * n_questions
        input_indices = q_masked + (s_masked - 1) * self.n_questions
        input_indices = input_indices.clamp(0, self.n_questions * 4)
        
        x = F.one_hot(input_indices, self.n_questions * 4 + 1).float().to(self.device())
        h = self(x)  # Shape: [batch, seq_len, n_questions, 4]
        
        # Get predictions for specific questions
        batch_size, seq_len = q.shape
        pred_seq_len = seq_len - n + 1
        
        if pred_seq_len <= 0:
            dummy_pred = torch.ones(batch_size, 1, 4, device=self.device()) * 0.25
            return dummy_pred, h
        
        # Extract option probabilities for the questions we want to predict
        q_pred = q[:, n-1:n-1+pred_seq_len]  # Questions to predict
        q_pred = q_pred.clamp(1, self.n_questions) - 1  # Convert to 0-based
        
        # Gather the option probabilities for the specific questions
        # h shape: [batch, seq_len, n_questions, 4]
        # We want to gather from dimension 2 (n_questions)
        batch_indices = torch.arange(batch_size).view(-1, 1, 1).expand(-1, pred_seq_len, 4)
        seq_indices = torch.arange(pred_seq_len).view(1, -1, 1).expand(batch_size, -1, 4)
        option_indices = torch.arange(4).view(1, 1, -1).expand(batch_size, pred_seq_len, -1)
        
        q_indices = q_pred.unsqueeze(-1).expand(-1, -1, 4)
        
        option_probs = h[batch_indices, seq_indices, q_indices, option_indices]
        
        return option_probs, h

    def get_loss(self, q, s, pid=None):
        assert pid is None, "DKT_Options does not support pid input"
        
        # Get predictions using the forward pass
        logits, _ = self.predict(q, s, pid)
        
        # Filter out masked positions
        valid_mask = s >= 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device(), requires_grad=True)
        
        # Reshape logits and labels for loss computation
        # logits shape: [batch, seq_len, 4]
        # s shape: [batch, seq_len]
        
        # Get valid predictions and labels
        valid_logits = logits[valid_mask]  # [num_valid, 4]
        valid_labels = s[valid_mask] - 1   # Convert from 1-4 to 0-3
        valid_labels = valid_labels.clamp(0, 3).long()
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(valid_logits, valid_labels)
        
        return loss