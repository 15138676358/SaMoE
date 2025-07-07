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

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models import MoEModel
from data_generator import generate_synthetic_data
import os
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

def save_model(model, model_path, optimizer=None, epoch=None, loss=None):
    """Save the trained model with optional training information."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        save_dict['epoch'] = epoch
    if loss is not None:
        save_dict['loss'] = loss
        
    torch.save(save_dict, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_path, model_class, *args, **kwargs):
    """Load a saved model."""
    checkpoint = torch.load(model_path)
    model = model_class(*args, **kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")
    return model, checkpoint

def get_expert_weights(model, context_tensor):
    """
    Extract expert weights from MoE model for given context.
    
    Args:
        model: Trained MoE model
        context_tensor: Context tensor of shape (B, 8)
        
    Returns:
        torch.Tensor: Expert weights of shape (B, num_experts)
    """
    if not isinstance(model, MoEModel):
        raise ValueError("This function is only for MoE models")
    
    model.eval()
    with torch.no_grad():
        num_experts = len(model.experts)
        context_x, context_y = context_tensor[:, 0::2], context_tensor[:, 1::2]
        flattened_context_x = context_x.view(-1, 1)
        flattened_context_y = context_y.view(-1, 1)
        
        expert_predictions = torch.stack([expert(flattened_context_x) for expert in model.experts], dim=0)
        flattened_context_y_expanded = flattened_context_y.unsqueeze(0).expand(num_experts, -1, 1)
        error_context_y = (expert_predictions - flattened_context_y_expanded).view(num_experts, -1, 4).transpose(0, 1)
        error_context = torch.sum(torch.pow(error_context_y, 2), dim=2)
        weights_context = torch.softmax(-error_context, dim=1)
        
    return weights_context

def analyze_expert_activation_heatmap(model, mu_range=(0, 1), num_mu_points=100, sigma_fixed=0.5, num_samples_per_mu=10):
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
    num_experts = len(model.experts)
    
    # Initialize heatmap matrix: (num_experts, num_mu_points)
    activation_heatmap = np.zeros((num_experts, num_mu_points))
    
    for i, mu in enumerate(mu_values):
        if i % 20 == 0:
            print(f"Processing mu point {i+1}/{num_mu_points}")
        
        mu_activations = []
        
        for _ in range(num_samples_per_mu):
            # Generate synthetic data with fixed mu and sigma
            X = np.random.uniform(0, 1, 5)
            y = (1 / (sigma_fixed * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma_fixed) ** 2)
            context = np.column_stack((X[:4], y[:4])).reshape(-1)
            
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            weights = get_expert_weights(model, context_tensor)
            mu_activations.append(weights.squeeze().numpy())
        
        # Average activations across samples for this mu
        avg_activations = np.mean(mu_activations, axis=0)
        activation_heatmap[:, i] = avg_activations
    
    results = {
        'activation_heatmap': activation_heatmap,
        'mu_values': mu_values,
        'num_experts': num_experts,
        'sigma_fixed': sigma_fixed
    }
    
    return results

def visualize_activation_heatmap(results, save_path=None):
    """
    可视化专家激活热力图
    
    Args:
        results: Results from analyze_expert_activation_heatmap
        save_path: Path to save the figure
    """
    activation_heatmap = results['activation_heatmap']
    mu_values = results['mu_values']
    num_experts = results['num_experts']
    sigma_fixed = results['sigma_fixed']
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    im = plt.imshow(activation_heatmap, 
                    cmap='viridis', 
                    aspect='auto',
                    extent=[mu_values[0], mu_values[-1], 0, num_experts])
    
    plt.colorbar(im, label='Expert Activation Weight')
    plt.xlabel('μ values')
    plt.ylabel('Expert Index')
    plt.title(f'Expert Activation Heatmap (σ = {sigma_fixed:.2f})')
    
    # Add expert labels on y-axis
    plt.yticks(range(num_experts), [f'Expert {i}' for i in range(num_experts)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Activation heatmap saved to {save_path}")
    
    plt.show()

def analyze_noise_resistance(model, mu_values, sigma_fixed=0.5, num_samples_per_mu=50):
    """
    分析专家噪声响应：评估网络对同一mu值不同样本的噪声抵抗程度
    
    Args:
        model: Trained MoE model
        mu_values: List of mu values to test
        sigma_fixed: Fixed sigma value for analysis
        num_samples_per_mu: Number of samples to generate for each mu
        
    Returns:
        dict: Analysis results containing noise resistance metrics
    """
    print("Analyzing noise resistance...")
    
    num_experts = len(model.experts)
    noise_metrics = []
    
    for mu in mu_values:
        print(f"Processing μ = {mu:.3f}")
        
        # Collect activations for multiple samples with same mu
        activations_for_mu = []
        
        for _ in range(num_samples_per_mu):
            # Generate synthetic data with fixed mu and sigma but different random X
            X = np.random.uniform(0, 1, 5)
            y = (1 / (sigma_fixed * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma_fixed) ** 2)
            context = np.column_stack((X[:4], y[:4])).reshape(-1)
            
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            weights = get_expert_weights(model, context_tensor)
            activations_for_mu.append(weights.squeeze().numpy())
        
        activations_for_mu = np.array(activations_for_mu)  # Shape: (num_samples, num_experts)
        
        # Calculate various noise resistance metrics
        metrics = calculate_noise_metrics(activations_for_mu, mu)
        noise_metrics.append(metrics)
    
    results = {
        'mu_values': mu_values,
        'noise_metrics': noise_metrics,
        'sigma_fixed': sigma_fixed,
        'num_samples_per_mu': num_samples_per_mu
    }
    
    return results

def calculate_noise_metrics(activations, mu):
    """
    计算噪声抵抗指标
    
    Args:
        activations: Array of shape (num_samples, num_experts)
        mu: Current mu value
        
    Returns:
        dict: Various noise resistance metrics
    """
    # 1. 标准差 (Standard Deviation) - 越小表示越稳定
    std_per_expert = np.std(activations, axis=0)
    mean_std = np.mean(std_per_expert)
    
    # 2. 变异系数 (Coefficient of Variation) - 标准差/均值
    mean_per_expert = np.mean(activations, axis=0)
    cv_per_expert = std_per_expert / (mean_per_expert + 1e-8)  # 避免除零
    mean_cv = np.mean(cv_per_expert)
    
    # 3. 熵变化 (Entropy Variation) - 分布的不确定性
    entropies = []
    for sample_activations in activations:
        # 计算每个样本的激活分布熵
        entropy_val = entropy(sample_activations + 1e-8)  # 避免log(0)
        entropies.append(entropy_val)
    entropy_std = np.std(entropies)
    
    # 4. 主导专家一致性 (Dominant Expert Consistency)
    dominant_experts = np.argmax(activations, axis=1)
    dominant_expert_mode = np.bincount(dominant_experts).argmax()
    consistency = np.mean(dominant_experts == dominant_expert_mode)
    
    # 5. 分布散度 (Distribution Divergence) - 使用KL散度
    mean_distribution = np.mean(activations, axis=0)
    kl_divergences = []
    for sample_activations in activations:
        kl_div = entropy(sample_activations + 1e-8, mean_distribution + 1e-8)
        kl_divergences.append(kl_div)
    mean_kl_divergence = np.mean(kl_divergences)
    
    return {
        'mu': mu,
        'mean_std': mean_std,
        'mean_cv': mean_cv,
        'entropy_std': entropy_std,
        'dominant_consistency': consistency,
        'mean_kl_divergence': mean_kl_divergence,
        'activations': activations,
        'std_per_expert': std_per_expert,
        'cv_per_expert': cv_per_expert
    }

def visualize_noise_resistance(results, save_path=None):
    """
    可视化噪声抵抗分析结果
    
    Args:
        results: Results from analyze_noise_resistance
        save_path: Path to save the figure
    """
    mu_values = results['mu_values']
    noise_metrics = results['noise_metrics']
    
    # Extract metrics for plotting
    mean_stds = [m['mean_std'] for m in noise_metrics]
    mean_cvs = [m['mean_cv'] for m in noise_metrics]
    entropy_stds = [m['entropy_std'] for m in noise_metrics]
    consistencies = [m['dominant_consistency'] for m in noise_metrics]
    kl_divergences = [m['mean_kl_divergence'] for m in noise_metrics]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Standard Deviation
    axes[0, 0].plot(mu_values, mean_stds, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('μ values')
    axes[0, 0].set_ylabel('Mean Standard Deviation')
    axes[0, 0].set_title('Activation Stability (Lower = More Stable)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Coefficient of Variation
    axes[0, 1].plot(mu_values, mean_cvs, 'r-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('μ values')
    axes[0, 1].set_ylabel('Mean Coefficient of Variation')
    axes[0, 1].set_title('Relative Variability (Lower = More Stable)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Entropy Standard Deviation
    axes[0, 2].plot(mu_values, entropy_stds, 'g-o', linewidth=2, markersize=4)
    axes[0, 2].set_xlabel('μ values')
    axes[0, 2].set_ylabel('Entropy Standard Deviation')
    axes[0, 2].set_title('Distribution Uncertainty Variation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Dominant Expert Consistency
    axes[1, 0].plot(mu_values, consistencies, 'm-o', linewidth=2, markersize=4)
    axes[1, 0].set_xlabel('μ values')
    axes[1, 0].set_ylabel('Dominant Expert Consistency')
    axes[1, 0].set_title('Expert Selection Consistency (Higher = More Consistent)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: KL Divergence
    axes[1, 1].plot(mu_values, kl_divergences, 'c-o', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('μ values')
    axes[1, 1].set_ylabel('Mean KL Divergence')
    axes[1, 1].set_title('Distribution Divergence from Mean')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Combined Noise Resistance Score
    # Normalize metrics to [0, 1] and combine (lower is better for most metrics)
    normalized_stds = np.array(mean_stds) / np.max(mean_stds)
    normalized_cvs = np.array(mean_cvs) / np.max(mean_cvs)
    normalized_entropy_stds = np.array(entropy_stds) / np.max(entropy_stds)
    normalized_kl = np.array(kl_divergences) / np.max(kl_divergences)
    # For consistency, higher is better, so we use (1 - consistency)
    normalized_inconsistency = 1 - np.array(consistencies)
    
    combined_score = (normalized_stds + normalized_cvs + normalized_entropy_stds + 
                     normalized_kl + normalized_inconsistency) / 5
    
    axes[1, 2].plot(mu_values, combined_score, 'k-o', linewidth=2, markersize=4)
    axes[1, 2].set_xlabel('μ values')
    axes[1, 2].set_ylabel('Combined Noise Sensitivity Score')
    axes[1, 2].set_title('Overall Noise Sensitivity (Lower = More Robust)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Noise resistance analysis saved to {save_path}")
    
    plt.show()

def print_noise_resistance_summary(results):
    """
    打印噪声抵抗分析摘要
    
    Args:
        results: Results from analyze_noise_resistance
    """
    mu_values = results['mu_values']
    noise_metrics = results['noise_metrics']
    
    print("=" * 60)
    print("NOISE RESISTANCE ANALYSIS SUMMARY")
    print("=" * 60)
    
    mean_stds = [m['mean_std'] for m in noise_metrics]
    consistencies = [m['dominant_consistency'] for m in noise_metrics]
    
    # Find most and least stable mu values
    most_stable_idx = np.argmin(mean_stds)
    least_stable_idx = np.argmax(mean_stds)
    
    print(f"Most stable μ: {mu_values[most_stable_idx]:.3f} (std: {mean_stds[most_stable_idx]:.4f})")
    print(f"Least stable μ: {mu_values[least_stable_idx]:.3f} (std: {mean_stds[least_stable_idx]:.4f})")
    
    # Find most and least consistent mu values
    most_consistent_idx = np.argmax(consistencies)
    least_consistent_idx = np.argmin(consistencies)
    
    print(f"Most consistent μ: {mu_values[most_consistent_idx]:.3f} (consistency: {consistencies[most_consistent_idx]:.4f})")
    print(f"Least consistent μ: {mu_values[least_consistent_idx]:.3f} (consistency: {consistencies[least_consistent_idx]:.4f})")
    
    print(f"\nOverall stability range: {np.min(mean_stds):.4f} - {np.max(mean_stds):.4f}")
    print(f"Overall consistency range: {np.min(consistencies):.4f} - {np.max(consistencies):.4f}")

def full_analysis_pipeline(model_path, model_class, model_args, save_dir="analysis_results"):
    """
    运行完整的分析流程
    
    Args:
        model_path: Path to the saved model
        model_class: The model class
        model_args: Arguments to initialize the model
        save_dir: Directory to save analysis results
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    model, checkpoint = load_model(model_path, model_class, *model_args)
    
    print("Starting comprehensive MoE analysis...\n")
    
    # 1. Expert activation heatmap analysis
    print("1. Expert Activation Heatmap Analysis")
    activation_results = analyze_expert_activation_heatmap(
        model, 
        mu_range=(0, 1), 
        num_mu_points=50, 
        sigma_fixed=0.5,
        num_samples_per_mu=5
    )
    
    heatmap_path = os.path.join(save_dir, "expert_activation_heatmap.png")
    visualize_activation_heatmap(activation_results, heatmap_path)
    
    # 2. Noise resistance analysis
    print("\n2. Noise Resistance Analysis")
    mu_test_values = np.linspace(0.1, 0.9, 10)  # Test 10 mu values
    noise_results = analyze_noise_resistance(
        model, 
        mu_test_values, 
        sigma_fixed=0.5, 
        num_samples_per_mu=30
    )
    
    noise_path = os.path.join(save_dir, "noise_resistance_analysis.png")
    visualize_noise_resistance(noise_results, noise_path)
    print_noise_resistance_summary(noise_results)
    
    print(f"\nAnalysis complete! Results saved in {save_dir}")
    
    return model, activation_results, noise_results

if __name__ == "__main__":
    # Example usage
    model_path = "trained_moe_model.pth"
    model_args = (4, 8, 1, 32, 1)  # num_experts, context_size, input_size, hidden_size, output_size
    
    try:
        model, activation_results, noise_results = full_analysis_pipeline(
            model_path, MoEModel, model_args
        )
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Please train a model first.")