import torch
import torch.nn as nn
import torch.nn.functional as F

class TextMLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, label_map):
        super(TextMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.label_map = label_map

    def forward(self, label):
        # Get the descriptive text from the label_map
        input_text = self.label_map.get(label, "")

        # Tokenize the input text into a list of words
        words = input_text.split()

        # Map words to indices using a simple vocabulary (assuming each word has a unique index)
        vocab = {word: idx for idx, word in enumerate(words)}
        indices = [vocab[word] for word in words if word in vocab]

        if not indices:
            # If no valid words found in the vocabulary, return zeros
            return torch.zeros(1, self.fc1.out_features)

        # Convert indices to a tensor
        x_lbl = torch.tensor(indices, dtype=torch.long)

        # Apply embedding layer
        x_lbl = self.embedding(x_lbl)

        # Average pooling over the sequence
        x_lbl = torch.mean(x_lbl, dim=0, keepdim=True)

        # Apply linear layer and activation function
        x_label = F.relu(self.fc1(x_lbl))

        return x_label

label_map = {
    0: "Un avión",
    1: "Un automóvil",
    2: "Un pájaro",
    3: "Un gato",
    4: "Un ciervo",
    5: "Un perro",
    6: "Una rana",
    7: "Un caballo",
    8: "Un barco",
    9: "Un camión"
}

# The size of the vocabulary would be the number of unique words in the label_map
vocab_size = len(set(" ".join(label_map.values()).split()))

embedding_dim = 10
hidden_dim = 20

# Instantiate TextMLP with the label_map
text_mlp = TextMLP(vocab_size, embedding_dim, hidden_dim, label_map)

# Input label
input_label = 1 

# Forward pass
output = text_mlp(input_label)
print(output)
