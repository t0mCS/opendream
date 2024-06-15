import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from typing import List

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

def load_model_and_tokenizer(model_id, revision):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision, device_map={"": 'cuda'})
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

def extract_image_representation(image_path, model):
    image = Image.open(image_path)
    with torch.no_grad():
        representation = model.encode_image(image).flatten()
    return representation

def process_image_directory(directory_path, model):
    representations = []
    for image_file in tqdm(os.listdir(directory_path), desc=f'Processing {os.path.basename(directory_path)}'):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(directory_path, image_file)
            representation = extract_image_representation(image_path, model)
            representations.append(representation)
    return representations

def extract_features_from_directory(base_dir):
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    model, tokenizer = load_model_and_tokenizer(model_id, revision)

    all_representations = {}

    for class_dir in os.listdir(base_dir):
        if class_dir.startswith('class_'):
            class_name = class_dir.split('class_')[1]
            full_class_dir = os.path.join(base_dir, class_dir)
            if os.path.isdir(full_class_dir):
                representations = process_image_directory(full_class_dir, model)
                all_representations[class_name] = representations

    return all_representations

def prepare_data_for_training(all_representations):
    features = []
    labels = []
    class_names = list(all_representations.keys())
    mlb = MultiLabelBinarizer(classes=class_names)
    
    for class_name, reps in all_representations.items():
        for rep in reps:
            features.append(rep)
            labels.append([class_name])

    features = torch.stack(features)
    labels = mlb.fit_transform(labels)

    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = ImageRepresentationDataset(X_train, torch.tensor(y_train, dtype=torch.float32))
    val_dataset = ImageRepresentationDataset(X_val, torch.tensor(y_val, dtype=torch.float32))

    return train_dataset, val_dataset, len(class_names)

def train_classifier(train_dataset, val_dataset, input_dim, output_dim, epochs=10, batch_size=32, lr=0.001, patience=5):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleMLP(input_dim, output_dim).cuda()

    criterion = nn.BCEWithLogitsLoss()
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
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            try:
                torch.save(model.state_dict(), 'best_classifier.pth')
                print(f"Model saved at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            early_stop = True

    return model

def finetune(data_dir, class_lbls: List[str]=None):
    all_representations = extract_features_from_directory(data_dir)
    train_dataset, val_dataset, num_classes = prepare_data_for_training(all_representations)

    input_dim = train_dataset[0][0].shape[0]
    classifier = train_classifier(train_dataset, val_dataset, input_dim, num_classes)

    if class_lbls:
        torch.save((classifier.state_dict(), class_lbls), 'best_classifier.pth')
    else:
        torch.save(classifier.state_dict(), 'best_classifier.pth')

def inference(image_path):
    # Load the vision model and tokenizer
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    vision_model, tokenizer = load_model_and_tokenizer(model_id, revision)

    # Extract image representation
    image_representation = extract_image_representation(image_path, vision_model)

    # Load the classifier model and class labels
    try:
        classifier_state, class_lbls = torch.load('best_classifier.pth')
    except ValueError:
        classifier_state = torch.load('best_classifier.pth')
        class_lbls = None

    input_dim = image_representation.shape[0]
    output_dim = len(class_lbls) if class_lbls else classifier_state['fc2.weight'].size(0)
    classifier = SimpleMLP(input_dim, output_dim).cuda()
    classifier.load_state_dict(classifier_state)
    classifier.eval()

    # Run inference
    image_representation = image_representation.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    with torch.no_grad():
        output = classifier(image_representation)
    predicted_labels = (torch.sigmoid(output) > 0.5).cpu().numpy()

    if class_lbls:
        predicted_class_labels = [class_lbls[i] for i in range(len(predicted_labels[0])) if predicted_labels[0][i]]
        return predicted_class_labels
    else:
        return predicted_labels

if __name__ == "__main__":
    # Example usage:
    data_dir = '/home/azureuser/opendream/backend/data/finetune_ckpt_1'
    finetune(data_dir)

    image_path = 'test_img.jpg'
    predictions = inference(image_path)
    print("Predictions:", predictions)
