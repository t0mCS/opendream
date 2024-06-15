import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from classifier import SimpleMLP
from sklearn.neighbors import KNeighborsClassifier
import joblib

def load_model_and_tokenizer(model_id, revision):
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision, device_map={"": 'cuda'})
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

def extract_image_representation(image_path, model):
    image = Image.open(image_path)
    with torch.no_grad():
        representation = model.encode_image(image).flatten()
    return representation

def load_classifier(input_dim, output_dim, model_path='artifacts/best_classifier.pth'):
    classifier_state, class_labels = torch.load(model_path)
    model = SimpleMLP(input_dim, output_dim).cuda()
    model.load_state_dict(classifier_state)
    model.eval()
    return model, class_labels

def load_knn_classifier(model_path='artifacts/best_knn_classifier.joblib'):
    knn, class_labels = joblib.load(model_path)
    return knn, class_labels

def inference(image_path, model_path='artifacts/best_classifier.pth', classifier_type='MLP'):
    # Load the vision model and tokenizer
    model_id = "vikhyatk/moondream2"
    revision = "2024-05-20"
    vision_model, tokenizer = load_model_and_tokenizer(model_id, revision)

    # Extract image representation
    image_representation = extract_image_representation(image_path, vision_model)

    if classifier_type == 'MLP':
        # Load the classifier model and class labels
        classifier_state, class_labels = torch.load(model_path)
        input_dim = image_representation.shape[0]
        output_dim = len(class_labels)
        classifier = SimpleMLP(input_dim, output_dim).cuda()
        classifier.load_state_dict(classifier_state)
        classifier.eval()

        # Run inference
        image_representation = image_representation.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
        with torch.no_grad():
            output = classifier(image_representation)
        predicted_label_idx = torch.argmax(output, dim=1).item()
        predicted_label = class_labels[predicted_label_idx]
    elif classifier_type == 'KNN':
        knn, class_labels = load_knn_classifier(model_path='artifacts/best_knn_classifier.joblib')
        image_representation = image_representation.unsqueeze(0).cpu().numpy()  # Add batch dimension and move to CPU
        predicted_label_idx = knn.predict(image_representation)[0]
        predicted_label = class_labels[predicted_label_idx]
        predicted_label = torch.nonzero(predicted_label).item()
    print(predicted_label)
    class_name = class_labels[predicted_label]
    return class_name

if __name__ == "__main__":
    image_path = 'test_img.jpg'
    classifier_type = 'KNN'  # Change this to 'MLP' or 'KNN' as needed
    predicted_label = inference(image_path, classifier_type=classifier_type)
    print("Predicted label:", predicted_label)
