import torch
import torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, log_prob_sum, seq_length):
        """
        log_prob_sum: Tensor of shape [N], the sum of log probabilities for each sequence
        seq_length: Tensor of shape [N], the length of each sequence
        We optimize a 'log temperature' here to avoid getting negative temperature
        """
        temperature = torch.exp(self.log_temperature)
        scaled_log_prob_sum = log_prob_sum / temperature
        confidence_norm = torch.exp(scaled_log_prob_sum / seq_length)
        return confidence_norm

model = TemperatureScaling()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 5
log_prob_sum = torch.tensor([-0.0175, -0.0175, -0.068656], requires_grad=True)
seq_length = torch.tensor([1, 2, 3], requires_grad=False)
labels = torch.tensor([0.0, 1.0, 0.0], requires_grad=False)

for epoch in range(num_epochs):
    optimizer.zero_grad()
    confidence_norm = model(log_prob_sum, seq_length)
    print(confidence_norm, labels)
    loss = criterion(confidence_norm, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Temperature: {torch.exp(model.log_temperature).item()}")

