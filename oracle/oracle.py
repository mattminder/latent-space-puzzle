import torch
import string
import unicodedata


class Oracle(torch.nn.Module):
    """Call oracle.verify_guess to check your guess."""
    
    output_length = 256
    tokens = string.ascii_lowercase + ",.! "

    def __init__(self):
        super().__init__()
        self.vocab_size = len(self.tokens)
        self.embedding = torch.nn.Embedding(self.vocab_size, self.output_length * self.vocab_size)

    def forward(self, first_name, last_name):
        input_sequence = self.normalize(first_name) + " " + self.normalize(last_name)
        tokens = self.encode(input_sequence)
        
        output = torch.zeros(self.output_length, self.vocab_size)
        for token in tokens:
            token_tens = torch.tensor(token)
            output = output + self.embedding(token_tens).view(self.output_length, self.vocab_size)
        return output
    
    def verify_guess(self, first_name, last_name):
        embeddings = self.forward(first_name, last_name)
        argmaxxed = embeddings.argmax(-1)
        return self.decode(argmaxxed)
    
    @staticmethod
    def normalize(text):
        # Remove weird accents, according to: https://stackoverflow.com/questions/3194516
        no_accents = ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')
        return no_accents.lower().strip()

    @classmethod
    def decode(cls, token_sequence):
        return "".join([cls.tokens[i] for i in token_sequence])

    @classmethod
    def encode(cls, text):
        return [cls.tokens.find(letter) for letter in text]
