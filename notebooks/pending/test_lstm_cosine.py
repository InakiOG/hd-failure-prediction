import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, n_neurons, input_shape, output_shape=2):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=input_shape, hidden_size=n_neurons, batch_first=True)
        self.fc = nn.Linear(n_neurons, output_shape)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last output of the LSTM
        return out

X_train = np.arange(0, 100, 0.5)
y_train_sin = np.sin(X_train)
y_train_cos = np.cos(X_train)

X_test = np.arange(100, 200, 0.5)
y_test_sin = np.sin(X_test)
y_test_cos = np.cos(X_test)


train_series = np.stack((y_train_sin, y_train_cos), axis=1)
test_series = np.stack((y_test_sin, y_test_cos), axis=1)

fig, ax = plt.subplots(1, 1, figsize=(15, 4))
ax.plot(X_train, y_train_sin, lw=3, label='train sine')
ax.plot(X_train, y_train_cos, lw=3, label='train cosine')
ax.plot(X_test, y_test_sin, lw=3, label='test sine')
ax.plot(X_test, y_test_cos, lw=3, label='test cosine')
ax.legend(loc="lower left")
plt.savefig('desired.png')

# then torch
train_series = torch.from_numpy(train_series).double()  # Convert to float64
test_series = torch.from_numpy(test_series).double()    # Convert to float64

# LSTM expects input of (batch, sequence, features)
# So shape should be (1, 179, 20) and labels (1, 1, 179)
look_back = 20

train_dataset = []
train_labels = []
for i in range(len(train_series) - look_back):
    train_dataset.append(train_series[i:i + look_back])
    train_labels.append(train_series[i + look_back])

train_dataset = torch.stack(train_dataset)
train_labels = torch.stack(train_labels)

test_dataset = []
for i in range(len(test_series) - look_back):
    test_dataset.append(test_series[i:i + look_back])

test_dataset = torch.stack(test_dataset)

n_neurons = 8

model = Net(n_neurons, input_shape=2).double()  # Ensure the model uses float64
loss_function = nn.L1Loss()
optimizer = optim.ASGD(model.parameters(), lr=0.01)

loss_curve = []
for epoch in range (10000):
    print(f"epoch: {epoch}/10000")
    loss_total = 0
    
    model.zero_grad()
    
    predictions = model(train_dataset)
    
    loss = loss_function(predictions, train_labels)
    loss_total += loss.item()
    loss.backward()
    optimizer.step()
    loss_curve.append(loss_total)
    
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(loss_curve, lw=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (L1)")
plt.savefig('loss.png')

with torch.no_grad():
    test_predictions = model(test_dataset).squeeze()

test_predictions = test_predictions.detach().numpy()

x = np.arange(100 + look_back, 100 + look_back + len(test_predictions) * 0.5, 0.5)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
ax.plot(X_train, y_train_sin, lw=2, label='train sine')
ax.plot(X_train, y_train_cos, lw=2, label='train cosine')
ax.plot(X_test, y_test_sin, lw=3, c='y', label='test sine')
ax.plot(X_test, y_test_cos, lw=3, c='g', label='test cosine')
ax.plot(x, test_predictions[:, 0], lw=3, c='r', linestyle=':', label='predicted sine')
ax.plot(x, test_predictions[:, 1], lw=3, c='b', linestyle=':', label='predicted cosine')
ax.legend(loc="lower left")
plt.savefig('predictions.png')
print("done!")