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
import glob
import shutil

def process_images_from_local_upload(path, tag_training_data, num_examples):
    # This flow will copy locally uploaded images for a class to the training 
    # This will match common image file extensions; adjust as necessary
    image_files = glob(os.path.join(path, '*.jpg')) + \
                glob(os.path.join(path, '*.jpeg')) + \
                glob(os.path.join(path, '*.png')) 

    # Sort the files to ensure consistency in order
    image_files.sort()
    
    # Get the first `num_examples` images
    images_to_copy = image_files[:num_examples]
    
    # Copy each image to the destination directory
    for image_file in images_to_copy:
        shutil.copy(image_file, tag_training_data)

def fetch_images_for_label(label, out_dir, num_images = 20):

    # Send a GET request to the Google image search URL
    url = f"https://www.google.com/search?q={label}&tbm=isch"
    num_pages = 5

    for page in range(num_pages):
        # Modify the URL to include the page number
        url_with_page = url + f"&start={page * 20}"
        response = requests.get(url_with_page)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the image elements in the search results
        image_elements = soup.find_all('img')

        # Download the images from the current page
        for i, image_element in enumerate(image_elements[:num_images], start=1):
            # Get the image source URL
            image_url = image_element['src']

            # Check if the image URL is a relative path
            if not image_url.startswith('http'):
                # Construct the complete URL by joining the base URL and the relative path
                image_url = urljoin(url, image_url)

            # Send a GET request to the image URL
            image_response = requests.get(image_url)

            # Save the image to the downloaded_images directory
            with open(f"{out_dir}/image_{page * num_images + i}.jpg", 'wb') as file:
                file.write(image_response.content)
