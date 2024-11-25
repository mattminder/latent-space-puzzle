import torch

class AttentionMatrix(torch.nn.Module):

    def __init__(self, n_hidden):
        super().__init__()
        self.query_layer = torch.nn.Linear(n_hidden, n_hidden)
        self.key_layer = torch.nn.Linear(n_hidden, n_hidden)

    def forward(self, embedding):
        q = self.query_layer(embedding)
        k = self.key_layer(embedding)
        return q @ k.transpose(2, 1)
    

class AttentionOutput(torch.nn.Module):

    def __init__(self, n_hidden):
        super().__init__()
        self.value_layer = torch.nn.Linear(n_hidden, n_hidden)
        self.softmax = torch.nn.Softmax(-1)

    def forward(self, embedding, attention_matrix):
        v = self.value_layer(embedding)
        softmaxxed = self.softmax(attention_matrix)
        return self.value_layer(softmaxxed @ v)
    

class NeuralNetwork(torch.nn.Module):
    """Implements a classifier with a single multihead attention layer."""

    number_heads = 3
    number_classes = 2

    def __init__(self, n_hidden=10):
        super().__init__()

        # tokens 0-9: digits, 10: CLS
        self.embedding = torch.nn.Embedding(11, n_hidden)
        
        self.attention_matrix_list = torch.nn.ModuleList(
            (
                AttentionMatrix(n_hidden) for _ in range(self.number_heads)
            )
        )
        self.attention_output_list = torch.nn.ModuleList(
            (
                AttentionOutput(n_hidden) for _ in range(self.number_heads)
            )
        )
        self.projection = torch.nn.Linear(self.number_heads * n_hidden, n_hidden)

        self.output = torch.nn.Linear(n_hidden, self.number_classes)

    def _get_logits_from_attention_matrices(self, embeddings, attention_matrices):
        attention_output = self._get_attention_output(embeddings, attention_matrices)

        # keep CLS token only for class predictions
        class_logits = self.output(attention_output[:, 0, ...])
        return class_logits

    def _get_attention_output(self, embedding, attention_matrices):
        concat = torch.concat([
            att_output_layer(embedding, att_m)
            for att_m, att_output_layer in zip(attention_matrices, self.attention_output_list)
        ], dim=-1)
        return self.projection(concat)

    def get_attention_matrices(self, embeddings):
        return [
            att_layer(embeddings)
            for att_layer in self.attention_matrix_list
        ]

    def forward(self, tokens):
        """Returns 1 if correct input was provided, 0 otherwise."""
        # first token must be CLS
        embeddings = self.embedding(tokens)
        attention_matrices = self.get_attention_matrices(embeddings)
        logits = self._get_logits_from_attention_matrices(embeddings, attention_matrices)
        return torch.softmax(logits, dim=-1)[:, 1]
