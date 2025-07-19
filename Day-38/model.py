from torch import nn
from torchinfo import summary

class MetaRLAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=128):
        super(MetaRLAgent, self).__init__()
        
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU()
        )
        
        self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, obs, rnn_hidden=None):
        # obs: [batch_size, seq_len, obs_dim]
        batch_size, seq_len, _ = obs.shape
        x = self.obs_encoder(obs.view(-1, obs.shape[-1]))
        x = x.view(batch_size, seq_len, -1)
        
        rnn_out, rnn_hidden = self.rnn(x, rnn_hidden)
        policy_logits = self.policy_head(rnn_out)
        value = self.value_head(rnn_out)

        return policy_logits, value, rnn_hidden

model = MetaRLAgent(obs_dim=10, action_dim=4)
summary(model, input_size=(2, 5, 10))
