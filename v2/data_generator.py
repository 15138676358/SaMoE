import os
import json
import random
from PIL import Image
import numpy as np


def load_image(image_path):
    """Load and return image as numpy array"""
    try:
        img = Image.open(image_path)
        img = img.resize((88, 88), Image.Resampling.LANCZOS)  # Resize to 88x88
        img = np.transpose(np.array(img), (2, 0, 1))  # from (H, W, C) to (C, H, W)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def generate_data_for_subdirectory(subdirectory, num_context=4):
    """Generate complete data structure for a subdirectory"""
    json_path = os.path.join(subdirectory, f'attempts.json')
    json_data = json.load(open(json_path)) if os.path.exists(json_path) else None
    if json_data is None:
        print(f"JSON data not found in {subdirectory}")
        return []
    
    input = {'object': [], 'img': [], 'loc': []}
    context = {'imgs': [], 'locs': [], 'dones': []}
    output = {'done': []}
    
    # Get all attempt files
    attempt_idx = []
    for file in sorted(os.listdir(subdirectory)):
        if file.startswith('attempt_') and file.endswith('_rgb.png'):
            attempt_id = file.split('_')[1]
            attempt_idx.append(attempt_id)
    
    for attempt_id in attempt_idx:
        # Generate context data (4 random images from same subdirectory)
        input_id = attempt_id
        context_idx = random.sample(attempt_idx, num_context)
        
        input['object'].append(subdirectory.split('/')[-1])
        input['img'].append(load_image(os.path.join(subdirectory, f'attempt_{input_id}_rgb.png')))
        input['loc'].append([loc / 352.0 for loc in json_data['attempt_' + input_id]['grasp_wrt_crop']])  # Normalize to [0, 1]
        output['done'].append([json_data['attempt_' + input_id]['grasp_success']])

        context['imgs'].append([load_image(os.path.join(subdirectory, f'attempt_{id}_rgb.png')) for id in context_idx])
        context['locs'].append([[loc / 352.0 for loc in json_data['attempt_' + id]['grasp_wrt_crop']] for id in context_idx])  # Normalize to [0, 1]
        context['dones'].append([[json_data['attempt_' + id]['grasp_success']] for id in context_idx])
    
    return input, context, output

def generate_dataset(object_idx):
    """Generate dataset from all subdirectories in dataset folder"""
    all_input, all_context, all_output = {'object': [], 'img': [], 'loc': []}, {'imgs': [], 'locs': [], 'dones': []}, {'done': []}
    
    # Process each subdirectory
    for subdir in object_idx:
        subdir_path = f"./v2/dataset/{subdir}"
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir}")
            input, context, output = generate_data_for_subdirectory(subdir_path)
            all_input['object'].extend(input['object'])
            all_input['img'].extend(input['img'])
            all_input['loc'].extend(input['loc'])
            all_context['imgs'].extend(context['imgs'])
            all_context['locs'].extend(context['locs'])
            all_context['dones'].extend(context['dones'])
            all_output['done'].extend(output['done'])
    
    return all_input, all_context, all_output

if __name__ == "__main__":
    # Generate the complete dataset
    dataset_dir = './v2/dataset'
    object_idx = sorted([subdir for subdir in os.listdir(dataset_dir)])
    # split the object_idx into 80% train and 20% test
    random.shuffle(object_idx)
    train_size = int(0.8 * len(object_idx))
    train_object_idx = np.random.choice(object_idx, train_size, replace=False)
    test_object_idx = [obj for obj in object_idx if obj not in train_object_idx]
    # train_object_idx = object_idx[:-15]
    # test_object_idx = object_idx[-15:]

    input, context, output = generate_dataset(train_object_idx)
    # 将数据转换为nparray,保存为npz
    np.savez_compressed(
        'v2/train_dataset.npz', 
        object=np.array(input['object']),
        input_img=np.array(input['img']),
        input_loc=np.array(input['loc']),
        context_imgs=np.array(context['imgs']),
        context_locs=np.array(context['locs']),
        context_dones=np.array(context['dones']),
        output_done=np.array(output['done'])
    )

    print(f"Generated training dataset with {len(output['done'])} entries")
    
    input, context, output = generate_dataset(test_object_idx)
    np.savez_compressed(
        'v2/test_dataset.npz', 
        object=np.array(input['object']),
        input_img=np.array(input['img']),
        input_loc=np.array(input['loc']),
        context_imgs=np.array(context['imgs']),
        context_locs=np.array(context['locs']),
        context_dones=np.array(context['dones']),
        output_done=np.array(output['done'])
    )
    print(f"Generated testing dataset with {len(output['done'])} entries")
    