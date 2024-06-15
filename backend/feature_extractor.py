import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

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

if __name__ == "__main__":
    base_dir = '/home/azureuser/opendream/backend/data/finetune_ckpt_1'
    all_representations = extract_features_from_directory(base_dir)
    os.makedirs('artifacts', exist_ok=True)
    torch.save(all_representations, 'artifacts/image_representations.pt')
    print('Features extracted and saved to artifacts/image_representations.pt')
