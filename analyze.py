"""
analyze.py
This module provides functions to analyze MoE model characteristics.
It includes:
1. Saving and loading models with training information.
2. Extracting expert weights for given contexts.
3. Analyzing expert activation heatmaps.
4. Analyzing noise resistance of the model.
5. Visualizing results with heatmaps and line plots.
6. Running a full analysis pipeline that combines all analyses and visualizations.
"""

import matplotlib.pyplot as plt
from models import End2EndModel, MoEModel_Exp, MoEModel_Imp
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch


def get_expert_weights(model, context_tensor):
    """
    Extract expert weights from MoE model for given context.
    
    Args:
        model: Trained MoE model
        context_tensor: Context tensor of shape (B, 8)
        
    Returns:
        torch.Tensor: Expert weights of shape (B, num_experts)
    """
    model.eval()
    with torch.no_grad():
        if isinstance(model, MoEModel_Exp):
            # Explicit Bayesian gate mechanism
            num_experts = len(model.experts)
            context_x, context_y = context_tensor[:, 0::2], context_tensor[:, 1::2]
            flattened_context_x = context_x.view(-1, 1)
            flattened_context_y = context_y.view(-1, 1)
            
            expert_predictions = torch.stack([expert(flattened_context_x) for expert in model.experts], dim=0)
            flattened_context_y_expanded = flattened_context_y.unsqueeze(0).expand(num_experts, -1, 1)
            error_context_y = (expert_predictions - flattened_context_y_expanded).view(num_experts, -1, 4).transpose(0, 1)
            error_context = torch.sum(torch.pow(error_context_y, 2), dim=2)
            weights_context = torch.softmax(-error_context, dim=1)

        elif isinstance(model, MoEModel_Imp):
            # Implicit Bayesian gate mechanism
            gate_output = torch.softmax(model.gate(context_tensor), dim=1)
            weights_context = gate_output
        
        elif isinstance(model, End2EndModel):
            # End-to-end model does not have experts, return uniform weights
            num_experts = 1
            weights_context = torch.ones(context_tensor.size(0), num_experts) / num_experts
        
    return weights_context

def analyze_expert_weights(model, mu_range=(0, 1), num_mu_points=100, num_samples_per_mu=10):
    """
    分析专家激活情况：mu作为x轴，专家作为y轴，激活值用颜色表示
    
    Args:
        model: Trained MoE model
        mu_range: Range of mu values (min, max)
        num_mu_points: Number of mu points to sample
        sigma_fixed: Fixed sigma value for analysis
        num_samples_per_mu: Number of samples to generate for each mu value
        
    Returns:
        dict: Analysis results containing heatmap data
    """
    print("Analyzing expert activation heatmap...")
    
    mu_values = np.linspace(mu_range[0], mu_range[1], num_mu_points)
    mean_weights_heatmap, std_weights_heatmap = [], []
    
    for mu in mu_values:
        sigma = np.random.uniform(0.2, 0.8)
        mu_weights = []
        
        for _ in range(num_samples_per_mu):
            # Generate synthetic data with fixed mu and sigma
            X = np.random.uniform(0, 1, 5)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
            context = np.column_stack((X[:4], y[:4])).reshape(-1)
            context_tensor = torch.FloatTensor(context).unsqueeze(0)

            weights = get_expert_weights(model, context_tensor).squeeze(0)
            mu_weights.append(weights.numpy())
        
        # Average activations across samples for this mu
        mu_weights = np.array(mu_weights)  # Shape: (num_samples_per_mu, num_experts)
        mean_weights_heatmap.append(np.mean(mu_weights, axis=0))
        std_weights_heatmap.append(np.std(mu_weights, axis=0))
        
    mean_weights_heatmap, std_weights_heatmap = np.array(mean_weights_heatmap), np.array(std_weights_heatmap)
    
    return mean_weights_heatmap, std_weights_heatmap

def analyze_expert_outputs(model, input_range=(0, 1), num_input_points=100):
    """
    分析专家输出：输入作为y轴，专家作为x轴，输出用颜色表示
    Args:
        model: Trained MoE model
        input_range: Range of input values (min, max)
        num_input_points: Number of input points to sample
    Returns:
        dict: Analysis results containing heatmap data
    """
    print("Analyzing expert outputs heatmap...")
    
    model.eval()
    input_values = np.linspace(input_range[0], input_range[1], num_input_points)
    input_tensor = torch.FloatTensor(input_values).unsqueeze(1)  # Shape: (num_input_points, 1)
    with torch.no_grad():
        expert_outputs_heatmap = [expert(input_tensor).squeeze(-1) for expert in model.experts]  # List of tensors from each expert
    
    return np.array(expert_outputs_heatmap).transpose(1, 0)

def analyze_predictions(model, mu_range=(0, 1), num_mu_points=100, num_samples_per_mu=10):
    """
    分析模型预测结果：真值为x轴，预测值为y轴
    Args:
        model: Trained MoE model
        mu_range: Range of mu values (min, max)
        num_mu_points: Number of mu points to sample
        num_samples_per_mu: Number of samples to generate for each mu value
    Returns:
        prediction, ground_truth
    """
    print("Analyzing model predictions...")
    
    model.eval()
    mu_values = np.linspace(mu_range[0], mu_range[1], num_mu_points)
    predictions, ground_truth = [], []
    
    for mu in mu_values:
        sigma = np.random.uniform(0.2, 0.8)
        for _ in range(num_samples_per_mu):
            # Generate synthetic data with fixed mu and sigma
            X = np.random.uniform(0, 1, 5)
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
            context = np.column_stack((X[:4], y[:4])).reshape(-1)
            input_data = torch.FloatTensor(X[4:]).unsqueeze(0)
            context_tensor = torch.FloatTensor(context).unsqueeze(0)

            with torch.no_grad():
                output = model(context_tensor, input_data).squeeze().item()
            predictions.append(output)
            ground_truth.append(y[4])
    
    return np.array(predictions), np.array(ground_truth)

def visualize_heatmap(heatmap, save_path=None):
    """
    可视化专家激活热力图
    
    Args:
        heatmap: Results from analyze_expert_weights
        save_path: Path to save the figure
    """
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Expert Activation')
    plt.xlabel('Experts')
    plt.ylabel('Mu Values')
    plt.title('Expert Activation Heatmap')
    plt.xticks(ticks=np.arange(heatmap.shape[1]), labels=[f'Expert {i+1}' for i in range(heatmap.shape[1])])
    plt.yticks(ticks=np.arange(0, heatmap.shape[0], 10), labels=[f'Mu {i+1}' for i in range(0, heatmap.shape[0], 10)])
    if save_path:
        plt.savefig(save_path)

def visualize_predictions(predictions, ground_truth, save_path=None):
    """
    可视化模型预测结果
    
    Args:
        predictions: Model predictions
        ground_truth: Ground truth values
        save_path: Path to save the figure
    """
    
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], color='red', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')
    plt.title('Model Predictions vs Ground Truth')
    if save_path:
        plt.savefig(save_path)

def main():
    # Example usage
    model_path = "trained_model_exp.pth"
    model_args = (10, 8, 1, 32, 1)  # num_experts, context_size, input_size, hidden_size, output_size
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please train a model first.")
    
    model = MoEModel_Exp(*model_args)
    model.load_state_dict(torch.load(model_path))
    
    mu_range = (0, 1)
    num_mu_points = 100
    num_samples_per_mu = 10
    
    mean_weights_heatmap, std_weights_heatmap = analyze_expert_weights(model, mu_range, num_mu_points, num_samples_per_mu)
    expert_outputs_heatmap = analyze_expert_outputs(model, input_range=(0, 1), num_input_points=100)
    predictions, ground_truth = analyze_predictions(model, mu_range, num_mu_points, num_samples_per_mu)
    
    visualize_heatmap(mean_weights_heatmap, save_path="mean_weights_heatmap.png")
    visualize_heatmap(std_weights_heatmap, save_path="std_weights_heatmap.png")
    visualize_heatmap(expert_outputs_heatmap, save_path="expert_outputs_heatmap.png")
    visualize_predictions(predictions, ground_truth, save_path="predictions_vs_ground_truth.png")



if __name__ == "__main__":
    main()