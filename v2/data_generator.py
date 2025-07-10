import os
import json
import random
from PIL import Image
import numpy as np

def load_image(image_path):
    """Load and return image as numpy array"""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_json_data(json_path):
    """Load JSON data and extract required fields"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return {
            'loc': data.get('grasp_wrt_crop'),
            'done': data.get('grasp_success')
        }
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return None

def generate_context_data(subdirectory, num_context=4):
    """Generate context data by randomly selecting images from subdirectory"""
    context_data = []
    
    # Get all attempt files in subdirectory
    attempt_files = []
    for file in os.listdir(subdirectory):
        if file.startswith('attempt_') and file.endswith('_rgb.png'):
            attempt_num = file.split('_')[1]
            attempt_files.append(attempt_num)
    
    # Randomly select context attempts
    selected_attempts = random.sample(attempt_files, min(num_context, len(attempt_files)))
    
    for attempt_num in selected_attempts:
        img_path = os.path.join(subdirectory, f'attempt_{attempt_num}_rgb.png')
        json_path = os.path.join(subdirectory, f'attempt_{attempt_num}.json')
        
        img = load_image(img_path)
        json_data = load_json_data(json_path)
        
        if img is not None and json_data is not None:
            context_data.append({
                'img': img,
                'loc': json_data['loc'],
                'done': json_data['done']
            })
    
    return context_data

def generate_data_for_subdirectory(subdirectory):
    """Generate complete data structure for a subdirectory"""
    all_data = []
    
    # Get all attempt files
    attempt_files = []
    for file in os.listdir(subdirectory):
        if file.startswith('attempt_') and file.endswith('_rgb.png'):
            attempt_num = file.split('_')[1]
            attempt_files.append(attempt_num)
    
    for attempt_num in attempt_files:
        # Generate context data (4 random images from same subdirectory)
        context = generate_context_data(subdirectory, num_context=4)
        
        # Load input data
        input_img_path = os.path.join(subdirectory, f'attempt_{attempt_num}_rgb.png')
        input_json_path = os.path.join(subdirectory, f'attempt_{attempt_num}.json')
        
        input_img = load_image(input_img_path)
        input_json_data = load_json_data(input_json_path)
        
        if input_img is not None and input_json_data is not None:
            data_entry = {
                'context': context,
                'input': {
                    'img': input_img,
                    'loc': input_json_data['loc']
                },
                'done': input_json_data['done']
            }
            all_data.append(data_entry)
    
    return all_data

def generate_dataset():
    """Generate dataset from all subdirectories in dataset folder"""
    dataset_dir = './v2/dataset'
    all_dataset = []
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found!")
        return []
    
    # Process each subdirectory
    for subdir in os.listdir(dataset_dir):
        subdir_path = os.path.join(dataset_dir, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir}")
            subdir_data = generate_data_for_subdirectory(subdir_path)
            all_dataset.extend(subdir_data)
    
    return all_dataset

if __name__ == "__main__":
    # Generate the complete dataset
    dataset = generate_dataset()
    print(f"Generated dataset with {len(dataset)} entries")
    
    # Example: Print structure of first entry
    if dataset:
        print("\nExample data structure:")
        print(f"Context length: {len(dataset[0]['context'])}")
        print(f"Input image shape: {dataset[0]['input']['img'].shape}")
        print(f"Input location: {dataset[0]['input']['loc']}")
        print(f"Done status: {dataset[0]['done']}")