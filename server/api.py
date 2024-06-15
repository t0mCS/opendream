from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import glob 
import shutil
import api_helper as api_helper

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
import main as model

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    """
    home(): returns a welcome message
    """
    return 'Welcome to opendream API!'

@app.route('/finetune', methods=['POST'])
def fintune():
    """
    fintune(): runs fine tuning for a user request
    """
    # Get the JSON data from the request
    data = request.get_json()
    listed_labels = []

    # Check if 'labels' key exists in the JSON data
    if 'labels' in data and 'job_name' in data:
        labels = data['labels']
        job_name = data['job_name']


        #create path to make a directory for fine tuning
        current_path = os.getcwd()
        directory_path = os.path.join(current_path, f'finetunes/{job_name}')

        # Create the directory if it doesn't exist
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while creating the directory: {e}")

        #parse labels and check what where we need to fetch the data from
        for label in labels:
            if 'tag' not in label:

                #Validate that it is a valid label provided
                return jsonify({
                    'status': 'error',
                    'message': 'All labels provided in the body of this request must have a [tag] attribute'
                }), 400
            
            
            tag = label['tag']
            path =  label['path'] if 'path' in label else None
            num_examples = label['num_examples'] if 'num_exmaples' in label else 20
            tagged_training_data_path = f'{directory_path}/{tag}'
            listed_labels.append(tag)

            # Create the directory if it doesn't exist
            try:
                os.makedirs(tagged_training_data_path, exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating the directory: {e}")

            #Determine where to fetch the image from 
            if path:
                api_helper.process_images_from_local_upload(path, tagged_training_data_path, num_examples)
            else:
                api_helper.fetch_images_for_label(tag, tagged_training_data_path, num_examples)

        #call fine tuning
        model.finetune(directory_path, listed_labels)

        return jsonify({
            'status': 'success'
        })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No labels found in the request body or job_name parameter is missing'
        }), 400

    return jsonify({"query": query})

@app.route('/inference', methods=['POST'])
def inference():
    """
    inference(): returns the image path from the request
    """
    # Get the image path from the request
    image_path = request.json.get('image_path')

    return jsonify({"image_path": image_path})

if __name__ == '__main__':
    app.run()