"""
The expert weights can be analyzed by inputting data from the same object name.
The expert outputs can be analyzed by travering through the (x, y) pixel space.
The model predictions can be analyzed by inputting test dataset.
The expert outputs and weights can be visualized as heatmaps.
The model predictions can be visualized as a scatter plot.
"""
import data_generator
import matplotlib.pyplot as plt
from models import End2EndModel, MoEModel_Exp, MoEModel_Imp, SaMoEModel
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def analyze_expert_weights(model, object_idx, data):
    """
    分析专家激活情况：object作为x轴，专家作为y轴，激活值用颜色表示
    """
    print("Analyzing expert activation heatmap...")
    
    mean_weights_heatmap, std_weights_heatmap = [], []
    consistency_heatmap = []

    for object_id in tqdm(object_idx, desc="Analyzing objects"):
        subset_idx = [i for i, obj in enumerate(data['object']) if obj == object_id]
        context = {'imgs': torch.tensor(data['context_imgs'][subset_idx]).float().to(device) / 255.0, 
                   'locs': torch.tensor(data['context_locs'][subset_idx]).float().to(device),
                   'dones': torch.tensor(data['context_dones'][subset_idx]).float().to(device)}
        input = {'img': torch.tensor(data['input_img'][subset_idx]).float().to(device) / 255.0,
                'loc': torch.tensor(data['input_loc'][subset_idx]).float().to(device)}
        output_done = torch.tensor(data['output_done'][subset_idx]).float().to(device)

        with torch.no_grad():
            expert_weights, expert_outputs = model(context, input)
            expert_accuracy = F.softmax(-torch.pow(expert_outputs - output_done.unsqueeze(1).repeat(1, len(model.experts), 1), 2), dim=1)  # (batch_size, num_experts)
            accuracy_e = torch.transpose_copy(expert_accuracy.squeeze(-1), 0, 1)
            feats_e = accuracy_e - accuracy_e.mean(dim=1, keepdim=True)
            feats_e = feats_e / (feats_e.std(dim=1, keepdim=True) + 1e-8)
            weights_t = torch.transpose_copy(expert_weights, 0, 1)
            feats_w = weights_t - weights_t.mean(dim=1, keepdim=True)
            feats_w = feats_w / (feats_w.std(dim=1, keepdim=True) + 1e-8)
            consistency_matrix = torch.matmul(feats_e, feats_w.T) / feats_w.shape[1]
            mask_e = torch.eye(consistency_matrix.size(0), dtype=torch.bool, device=consistency_matrix.device)
        
        expert_weights = expert_weights.cpu().numpy()
        mean_weights_heatmap.append(np.mean(expert_weights, axis=0))  
        std_weights_heatmap.append(np.std(expert_weights, axis=0))
        expert_consistency = consistency_matrix[mask_e].flatten().cpu().numpy()
        consistency_heatmap.append(expert_consistency)

        
    mean_weights_heatmap, std_weights_heatmap, consistency_heatmap = np.array(mean_weights_heatmap), np.array(std_weights_heatmap), np.array(consistency_heatmap)

    # draw heatmap
    for heatmap, title, filename in zip(
        [mean_weights_heatmap, std_weights_heatmap, consistency_heatmap],
        ['Mean Expert Activation', 'Std Expert Activation', 'Expert Consistency'],
        ['expert_activation_mean_heatmap.png', 'expert_activation_std_heatmap.png', 'expert_consistency_heatmap.png']
    ):
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='coolwarm', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Activation Value')
        plt.xticks(ticks=np.arange(len(model.experts)), labels=[f'Expert {i+1}' for i in range(len(model.experts))])
        plt.yticks(ticks=np.arange(len(object_idx)), labels=object_idx)
        plt.xlabel('Experts')
        plt.ylabel('Objects')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f'v2/{filename}')
        plt.close()

def analyze_expert_variability(model, data):
    print("Analyzing expert predictions scatter plot...")
    context = {'imgs': torch.tensor(data['context_imgs']).float().to(device) / 255.0, 
               'locs': torch.tensor(data['context_locs']).float().to(device),
               'dones': torch.tensor(data['context_dones']).float().to(device)}
    input = {'img': torch.tensor(data['input_img']).float().to(device) / 255.0,
             'loc': torch.tensor(data['input_loc']).float().to(device)}
    output_done = torch.tensor(data['output_done']).float().to(device)

    with torch.no_grad():
        if isinstance(model, End2EndModel):
            output_pred = model(context, input)
        else:
            expert_weights, expert_outputs = model(context, input)
            output_pred = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)
            weights_t = torch.transpose_copy(expert_weights, 0, 1)
            feats_w = weights_t - weights_t.mean(dim=1, keepdim=True)
            feats_w = feats_w / (feats_w.std(dim=1, keepdim=True) + 1e-8)
            corr_matrix = torch.matmul(feats_w, feats_w.T) / feats_w.shape[1]  # shape: (num_experts, num_experts)

    corr_matrix = corr_matrix.cpu().numpy()
    # draw heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(ticks=np.arange(len(model.experts)), labels=[f'Expert {i+1}' for i in range(len(model.experts))])
    plt.yticks(ticks=np.arange(len(model.experts)), labels=[f'Expert {i+1}' for i in range(len(model.experts))])
    plt.xlabel('Experts')
    plt.ylabel('Experts')
    plt.title('Expert Correlation Matrix')
    plt.tight_layout()
    plt.savefig('v2/expert_correlation_heatmap.png')
    plt.close()

def analyze_expert_outputs(model, input_img=np.zeros((3, 88, 88))):
    """
    分析专家输出，在0-1范围采样88*88图像作为输入，生成专家输出热力图
    """
    print("Analyzing expert outputs heatmap...")
    input_img = torch.tensor(input_img).float()  # (C, H, W)
    img_gray = input_img.mean(axis=0)  # (H, W)
    mask = img_gray > 64  # (H, W)
    input_locs = torch.nonzero(mask, as_tuple=False).to(device)  # (N, 2), 每行是(y, x)
    input_imgs = input_img.unsqueeze(0).expand(len(input_locs), -1, -1, -1).to(device)  # (N, C, H, W)
    input = {'img': input_imgs.float() / 255.0, 'loc': input_locs.float() / 88.0}

    with torch.no_grad():
        input_feature = model.input_module(input)
        output_pred = torch.stack([expert(input_feature) for expert in model.experts], dim=0).squeeze(-1)  # (num_experts, N)
    
    # output_pred = input_feature.transpose(0, 1)  # Debug: draw the feature map of input module
    # draw heatmap
    n_rows, n_cols = 4, int(np.ceil(len(model.experts) / 4))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()
    mask = mask.cpu().numpy()
    for i in range(len(model.experts)):
        ax = axes[i]
        empty_img = np.zeros((88, 88))
        empty_img[mask] = output_pred[i].cpu().numpy()
        ax.imshow(empty_img, cmap='coolwarm', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'Expert {i+1}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_expert_predictions(model, data):
    print("Analyzing expert predictions scatter plot...")
    context = {'imgs': torch.tensor(data['context_imgs']).float().to(device) / 255.0, 
               'locs': torch.tensor(data['context_locs']).float().to(device),
               'dones': torch.tensor(data['context_dones']).float().to(device)}
    input = {'img': torch.tensor(data['input_img']).float().to(device) / 255.0,
             'loc': torch.tensor(data['input_loc']).float().to(device)}
    output_done = torch.tensor(data['output_done']).float().to(device)

    with torch.no_grad():
        if isinstance(model, End2EndModel):
            output_pred = model(context, input)
        else:
            expert_weights, expert_outputs = model(context, input)
            output_pred = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)

    output_pred = output_pred.cpu().numpy()
    output_done = output_done.cpu().numpy()
    auroc_score = roc_auc_score(output_done, output_pred)

    # draw histogram of pred, done=true is blue, done=false is red
    plt.figure(figsize=(8, 6))
    plt.hist(output_pred[output_done == 1], bins=50, alpha=0.5, color='blue', label='Done = True')
    plt.hist(output_pred[output_done == 0], bins=50, alpha=0.5, color='red', label='Done = False')
    plt.xlabel('Predicted Done Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Done Values, AUROC: {:.4f}'.format(auroc_score))
    plt.legend()
    plt.tight_layout()
    plt.savefig('v2/expert_predictions_histogram.png')
    plt.close()

def main():
    # Example usage
    model_path = "v2/model_sam.pth"
    model_args = (15,32)  # num_experts, hidden_size
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train a model first.")
    
    model = SaMoEModel(*model_args).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    image_path = 'v2/dataset/t_shape-08231207/attempt_0_rgb.png'
    # image_path = 'v2/dataset/long_l_shape-08272053/attempt_0_rgb.png'
    img = data_generator.load_image(image_path)
    analyze_expert_outputs(model, img)

    # Load dataset from .npz file
    data = np.load('v2/dataset.npz')
    object_idx = sorted([subdir for subdir in os.listdir('./v2/dataset')])
    analyze_expert_weights(model, object_idx, data)
    analyze_expert_variability(model, data)
    analyze_expert_predictions(model, data)



if __name__ == "__main__":
    main()