from typing import List
import torch
import os
import joblib
from feature_extractor import extract_features_from_directory
from classifier import prepare_data_for_training, train_classifier, train_knn_classifier
from inference import inference as inference_function

def finetune(data_dir, class_lbls: List[str] = None, classifier_type='MLP'):
    os.makedirs('artifacts', exist_ok=True)

    all_representations = extract_features_from_directory(data_dir)
    torch.save(all_representations, 'artifacts/image_representations.pt')
    
    train_dataset, val_dataset, num_classes, class_labels = prepare_data_for_training(all_representations)

    input_dim = train_dataset[0][0].shape[0]
    if classifier_type == 'MLP':
        classifier = train_classifier(train_dataset, val_dataset, input_dim, num_classes)
        torch.save((classifier.state_dict(), class_labels), 'artifacts/best_classifier.pth')
    elif classifier_type == 'KNN':
        classifier, class_labels = train_knn_classifier(train_dataset, val_dataset)
        knn_model_path = 'artifacts/best_knn_classifier.joblib'
        joblib.dump((classifier, class_labels), knn_model_path)

    print('Finetuning completed and model saved to artifacts')

def inference(image_path, classifier_type='MLP'):
    predicted_label = inference_function(image_path, classifier_type=classifier_type)
    return predicted_label

if __name__ == "__main__":
    data_dir = '/home/azureuser/opendream/backend/data/finetune_ckpt_1'
    image_path = 'test_img.jpg'
    classifier_type = 'KNN'  # 'MLP' or 'KNN'

    # finetune(data_dir, classifier_type=classifier_type)

    prediction = inference(image_path, classifier_type=classifier_type)

    print("Predicted label:", prediction)
