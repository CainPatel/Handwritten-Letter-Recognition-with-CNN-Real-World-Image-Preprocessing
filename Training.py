# Handwritten text identifier
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# transform image to tensor and standardize pixel values
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# get the letters from teh EMNIST dataset, and tranform them
train_ds = datasets.EMNIST(root="./data", split="letters", train=True,download = True, transform=transform)
test_ds = datasets.EMNIST(root="./data", split="letters", train = False, download = True, transform=transform)

# segments the dataset into mini batches
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16, kernel_size=3, padding=1) # first convolution layer 1 to 16 channels (feature scans)
        self.conv2 = nn.Conv2d(16,32, kernel_size=3, padding=1) # second convolution layer 16 to 32 channels (feature scans)
        self.pool = nn.MaxPool2d(2,2) #reduces the spatial size of the feature scans by halving them 28x28 is reduced to 7x7
        self.fc1 = nn.Linear(32*7*7,64) # total nodes goes from 32 channels * 7 pixel * 7 pixel to 64 flat
        self.fc2 = nn.Linear(64,26) # total nodes goes from 64 to 26 flat
        self.relu = nn.ReLU() #relu function used for optimization

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # creates first convolution layer to extract features, relu zeros neg, pools
        x = self.pool(self.relu(self.conv2(x))) # 2nd conv layer extracts features, relu zeros neg, pools again
        x = x.view(x.size(0), -1) # flattens the vector so dense layer can take it
        x = self.relu(self.fc1(x)) # dense layer + ReLU
        x = self.fc2(x) # final layer with 26 raw logits returned
        return x

model = CNN()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for X_batch, y_batch in train_loader:
        y_batch -= 1
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
    model.eval()
    correct, total = 0,0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch -= 1
            logits = model(X_batch)
            pred = logits.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(pred.tolist())
            all_labels.extend(y_batch.tolist())
    acc = correct/total

    cm = confusion_matrix(all_labels,all_preds)

    letters = "abcdefghijklmnopqrstuvwxyz"
    
    print(f"Epoch: {epoch} Accuracy: {acc}")
    print("Most confused pairs:")
    for i in range(26):
        for j in range(26):
            if i != j and cm[i,j] > 30: #threshold for notable confusion
                print(f" '{letters[i]}' mistaken for '{letters[j]}': {cm[i,j]} times")

torch.save(model.state_dict(), "emnist_cnn.pth")
# Epoch: 0 Accuracy: 0.9050961538461538
# Most confused pairs:
#  'a' mistaken for 'q': 39 times
#  'c' mistaken for 'e': 64 times
#  'g' mistaken for 'b': 38 times
#  'g' mistaken for 'q': 171 times
#  'i' mistaken for 'l': 145 times
#  'l' mistaken for 'i': 195 times
#  'q' mistaken for 'g': 46 times
# Epoch: 1 Accuracy: 0.9178846153846154
# Most confused pairs:
#  'g' mistaken for 'q': 117 times
#  'i' mistaken for 'l': 162 times
#  'j' mistaken for 'i': 33 times
#  'l' mistaken for 'i': 177 times
#  'q' mistaken for 'a': 31 times
#  'q' mistaken for 'g': 70 times
#  'v' mistaken for 'u': 39 times
# Epoch: 2 Accuracy: 0.9223076923076923
# Most confused pairs:
#  'd' mistaken for 'o': 37 times
#  'g' mistaken for 'q': 65 times
#  'i' mistaken for 'l': 305 times
#  'l' mistaken for 'i': 92 times
#  'q' mistaken for 'a': 42 times
#  'q' mistaken for 'g': 111 times
#  'v' mistaken for 'u': 36 times
# Epoch: 3 Accuracy: 0.9250961538461538
# Most confused pairs:
#  'g' mistaken for 'q': 78 times
#  'i' mistaken for 'l': 188 times
#  'l' mistaken for 'i': 163 times
#  'n' mistaken for 'm': 45 times
#  'q' mistaken for 'g': 100 times
#  'v' mistaken for 'u': 36 times
# Epoch: 4 Accuracy: 0.9310096153846154
# Most confused pairs:
#  'd' mistaken for 'o': 42 times
#  'g' mistaken for 'q': 70 times
#  'i' mistaken for 'l': 144 times
#  'l' mistaken for 'i': 206 times
#  'q' mistaken for 'g': 99 times
#  'u' mistaken for 'v': 38 times