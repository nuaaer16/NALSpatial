import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the corpus
df = pd.read_excel('spatial_query_corpus.xlsx')

# Convert category labels to numbers
labels = ['Range Query', 'Nearest Neighbor Query', 'Spatial Join Query', 'Distance Join Query', \
          'Aggregation-count Query', 'Aggregation-sum Query', 'Aggregation-max Query', \
            'Basic-distance Query', 'Basic-direction Query', 'Basic-length Query', 'Basic-area Query']
df['cat'] = df['cat'].apply(lambda x: labels.index(x))

# Split the data set
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Build vocabulary and text embedding vectors
words = df['review'].str.split(expand=True).unstack().value_counts()
word_to_idx = {word: idx + 2 for idx, word in enumerate(words.index)}
word_to_idx['<PAD>'] = 0
word_to_idx['<UNK>'] = 1
max_length = df['review'].str.split().apply(len).max()
train_vectors = np.stack(train_df['review'].apply(lambda x: np.array([word_to_idx.get(word, 1) for word in x.split()] + [0]*(max_length-len(x.split())))))
test_vectors = np.stack(test_df['review'].apply(lambda x: np.array([word_to_idx.get(word, 1) for word in x.split()] + [0]*(max_length-len(x.split())))))

# Convert data to Tensor
train_labels = torch.tensor(train_df['cat'].values)
test_labels = torch.tensor(test_df['cat'].values)
train_vectors = torch.LongTensor(train_vectors)
test_vectors = torch.LongTensor(test_vectors)

# Define the LSTMCNN model
class LSTMCNN(nn.Module):
    def __init__(self, num_embeddings, embedding_size, hidden_size, num_classes, num_filters=100, kernel_sizes=[3,4,5]):
        super(LSTMCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.convolution_layers = nn.ModuleList([
            nn.Conv1d(in_channels=2*hidden_size, out_channels=num_filters, kernel_size=kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_output, (h_n, c_n) = self.lstm(x)
        x = lstm_output.permute(0, 2, 1)
        convolution_outputs = []
        for convolution in self.convolution_layers:
            convolution_output = convolution(x)
            convolution_output = nn.functional.relu(convolution_output)
            max_pool_output = nn.functional.max_pool1d(convolution_output, kernel_size=convolution_output.size(2))
            convolution_outputs.append(max_pool_output)
        concatenated_tensor = torch.cat(convolution_outputs, dim=1)
        flatten_tensor = concatenated_tensor.view(concatenated_tensor.size(0), -1)
        dropout_output = self.dropout(flatten_tensor)
        logits = self.linear(dropout_output)
        return logits

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(labels)
model = LSTMCNN(len(word_to_idx), embedding_size=100, hidden_size=64, num_classes=num_classes, num_filters=100, kernel_sizes=[3,4,5]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    logits = model(train_vectors.to(device))
    loss = criterion(logits, train_labels.to(device))
    loss.backward()
    optimizer.step()
    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    logits = model(test_vectors.to(device))
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = (predicted_classes == test_labels.to(device)).float().mean().item()
    print(f'Accuracy: {accuracy:.4f}')

# Store the model and related information
torch.save(model.state_dict(), 'model.pth')
torch.save(word_to_idx, 'word_to_idx.pth')
torch.save(max_length, 'max_length.pth')
