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

    number_heads = 2
    number_classes = 2
    max_sequence_length = 5
    n_hidden = 5

    def __init__(self):
        super().__init__()


        # tokens 0-9: digits, 10: CLS
        self.embedding = torch.nn.Embedding(11, self.n_hidden)
        
        self.attention_matrix_list = torch.nn.ModuleList(
            (
                AttentionMatrix(self.n_hidden) for _ in range(self.number_heads)
            )
        )
        self.attention_output_list = torch.nn.ModuleList(
            (
                AttentionOutput(self.n_hidden) for _ in range(self.number_heads)
            )
        )
        self.projection = torch.nn.Linear(self.number_heads * self.n_hidden, self.n_hidden)

        self.output = torch.nn.Linear(self.n_hidden * self.max_sequence_length, self.number_classes)

    def _get_logits_from_attention_matrices(self, embeddings, attention_matrices):
        attention_output = self._get_attention_output(embeddings, attention_matrices)
        with_skip = (embeddings + attention_output).squeeze(0)

        padded = torch.zeros((self.max_sequence_length, self.n_hidden), device=self.output.weight.device)
        padded[:with_skip.shape[0], :] = with_skip

        class_logits = self.output(padded.flatten())
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
        """Returns >0.8 if correct input was provided."""
        # first token is expected to be CLS
        embeddings = self.embedding(tokens)
        
        attention_matrices = self.get_attention_matrices(embeddings)
        logits = self._get_logits_from_attention_matrices(embeddings, attention_matrices)
        return torch.softmax(logits, dim=-1)[1]
