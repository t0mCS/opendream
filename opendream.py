from import_reqs import *
from finetune import fine_tune
from validate import validate
# Create a directory to store the downloaded images
os.makedirs('downloaded_images', exist_ok=True)

# Prompt the user for the search query
query = input("Enter your search query: ")
num_images = 20
num_pages = 3  # Number of pages to scroll through

# Send a GET request to the Google image search URL
url = f"https://www.google.com/search?q={query}&tbm=isch"

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
        with open(f"downloaded_images/image_{page * num_images + i}.jpg", 'wb') as file:
            file.write(image_response.content)

print("Images downloaded successfully.")

# write a generalDataset class that given [image, solution] creates a dataset
class GeneralDataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (list of tuples): A list of (image, solution) tuples where
                - image is a path to the image or a PIL Image
                - solution is the text solution for the captcha
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, solution = self.data[idx]

        # If the image is a path, open it as a PIL image
        if isinstance(image, str):
            image = Image.open(image)

        return {
            "image": image,
            "qa": [
                {
                    "question": "What does the text say?",
                    "answer": solution,
                }
            ]
        }

# Example usage:
# Assuming 'data' is a list of tuples where each tuple is (image_path_or_PIL_image, solution)
data = []
for i in range(num_pages * num_images):
    data.append((f"downloaded_images/image_{i + 1}.jpg", query))

training_data = data[:int(0.8 * len(data))]
validation_data = data[int(0.8 * len(data)):int(0.9 * len(data))]
test_data = data[int(0.9 * len(data)):]

datasets = {
    "train": GeneralDataset(training_data),
    "val": GeneralDataset(validation_data),
    "test": GeneralDataset(test_data),
}

print(datasets)

fine_tune(datasets)

validate(datasets)