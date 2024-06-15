from torch.utils.data import Dataset
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from bitsandbytes.optim import Adam8bit
import math
from einops import rearrange
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import random

class CaptchaDataset(Dataset):
    def __init__(self, split='train'):
        self.data = load_dataset(
            "project-sloth/captcha-images",
            revision="refs/convert/parquet",
        )[split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "image": sample["image"], # Should be a PIL image
            "qa": [
                {
                    "question": "What does the text say?",
                    "answer": sample["solution"],
                }
            ]
        }

# Initialize moondream. Change DEVICE to 'mps' if you're on an M1 Mac, or 'cpu' if you don't have a
# GPU. Note that fine-tuning on CPU will be very slow.s
DEVICE = "cuda"
DTYPE = torch.float32 if DEVICE == "cpu" else torch.float16 # CPU doesn't support float16
MD_REVISION = "2024-03-13"

tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision=MD_REVISION)

