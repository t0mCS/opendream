import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KNeighborsClassifier
import joblib  # For saving/loading sklearn models

class ImageRepresentationDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prepare_data_for_training(all_representations):
    features = []
    labels = []
    class_names = list(all_representations.keys())
    lb = LabelBinarizer()
    
    for class_name, reps in all_representations.items():
        for rep in reps:
            features.append(rep)
            labels.append(class_name)

    features = torch.stack(features)
    labels = lb.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = ImageRepresentationDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
    val_dataset = ImageRepresentationDataset(X_val, torch.tensor(y_val, dtype=torch.float32))

    return train_dataset, val_dataset, len(class_names), lb.classes_

def train_classifier(train_dataset, val_dataset, input_dim, output_dim, epochs=10, batch_size=32, lr=0.001, patience=5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(input_dim, output_dim).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(epochs):
        if early_stop:
            break

        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(targets, 1)[1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(targets, 1)[1])
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), 'artifacts/best_classifier.pth')
                print(f"Model saved at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            early_stop = True

    return model

def train_knn_classifier(train_dataset, val_dataset, k=5):
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Convert dataset to numpy arrays
    for features, labels in train_loader:
        X_train = features.cpu().numpy()
        y_train = torch.argmax(labels, dim=1).numpy()

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Save KNN classifier and class labels
    return knn, val_loader.dataset.labels
