import torch

class PropertyPreservingNetwork(torch.nn.Module):
    num_embeddings = 100
    hidden_size = 2
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.hidden_size)
        self.output = torch.nn.Linear(self.hidden_size, self.num_embeddings)

    def forward(self, x):
        return self.output(self.embedding(x))
