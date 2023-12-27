import torch
import torch.nn as nn
import pickle
import numpy as np


labels = ['Range Query', 'Nearest Neighbor Query', 'Spatial Join Query', 'Distance Join Query', \
          'Aggregation-count Query', 'Aggregation-sum Query', 'Aggregation-max Query', \
            'Basic-distance Query', 'Basic-direction Query', 'Basic-length Query', 'Basic-area Query']

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


# Load the model and related information
word_to_idx = torch.load('word_to_idx.pth')
max_length = torch.load('max_length.pth')
num_classes = len(labels)
model = LSTMCNN(len(word_to_idx), embedding_size=100, hidden_size=64, num_classes=num_classes, num_filters=100, kernel_sizes=[3,4,5])
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Predict the type of text
def predict_type(text):
    vector = np.array([word_to_idx.get(word, 1) for word in text.split()] + [0]*(max_length-len(text.split())))
    vector_tensor = torch.LongTensor(vector).unsqueeze(0)
    with torch.no_grad():
        logits = model(vector_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
    return labels[predicted_class]


print(predict_type('What are the kinos in thecenter?'))
print(predict_type('Find the 13 nearest kneipen to the flaechen named Viktoriapark?'))
print(predict_type('What kinos are within 1.5 kilometers of each Flaechen?'))
print(predict_type('How many kinos are in each Flaechen?'))
print(predict_type('Returns the distance between mehringdamm and alexanderplatz'))
