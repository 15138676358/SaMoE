"""
models.py
This module defines the structure of two models: The MoEModel and the End2EndModel.
"""
from abc import ABC, abstractmethod
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

class End2EndModel(nn.Module):
    """
    End-to-end model that processes context, input, and output_gt.
    """
    def __init__(self, context_size=8, hidden_size=32, output_size=1):
        super(End2EndModel, self).__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.output_decoder = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, context, input):
        context_encoded = self.context_encoder(context)
        combined = torch.cat((context_encoded, input), dim=1)
        output = self.output_decoder(combined)
        
        return output

class MoEModel(nn.Module):
    """
    Mixture of Experts (MoE) model that processes context, input, and output_gt.
    """
    def __init__(self, num_experts=4, context_size=8, input_size=1, hidden_size=32, output_size=1):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_experts)
        ])

    @abstractmethod
    def _get_expert_weights(self, context):
        """
        Get the weights for each expert based on the context.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def forward(self, context, input):
        expert_weights = self._get_expert_weights(context)
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)
        combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)

        return combined_output

class MoEModel_Imp(MoEModel):
    """
    Mixture of Experts (MoE) model that processes context, input, and output_gt.
    The gate mechanism is implicit gate neural network.
    """
    @override
    def __init__(self, num_experts=4, context_size=8, input_size=1, hidden_size=32, output_size=1):
        super(MoEModel_Imp, self).__init__(num_experts=num_experts, context_size=context_size, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        self.expert_weights_gate = nn.Sequential(
            nn.Linear(context_size, num_experts),
            nn.ReLU(),
            nn.Linear(num_experts, num_experts),
            nn.Softmax(dim=1)  # Ensure the weights sum to 1
        )   
    
    @override
    def _get_expert_weights(self, context):
        """
        Get the weights for each expert based on the context.
        This method uses a softmax function to compute the weights.
        """
        expert_weights = self.expert_weights_gate(context)
        
        return expert_weights

class MoEModel_Exp(MoEModel):
    """
    Mixture of Experts (MoE) model that processes context, input, and output_gt.
    The gate mechanism is explicit bayesian.
    """
    @override
    def _get_expert_weights(self, context):
        """
        Get the weights for each expert based on the context.
        This method uses a softmax function to compute the weights.
        """
        num_experts = len(self.experts)
        # context dimension is (B, 4, 2), separate into (B, 4) and (B, 4)
        context_x, context_y = context[:, 0::2], context[:, 1::2]  # (B, 4), (B, 4)
        flattened_context_x, flattened_context_y = context_x.view(-1, 1), context_y.view(-1, 1)  # (B*4, 1), (B*4, 1)
        
        # Use torch.stack instead of torch.tensor to preserve gradients
        expert_predictions = torch.stack([expert(flattened_context_x) for expert in self.experts], dim=0)  # (num_experts, B*4, 1)
        flattened_context_y_expanded = flattened_context_y.unsqueeze(0).expand(num_experts, -1, 1)  # (num_experts, B*4, 1)
        expert_errors_y = (expert_predictions - flattened_context_y_expanded).view(num_experts, -1, 4).transpose(0, 1)  # (B, num_experts, 4)
        
        expert_errors = torch.sum(torch.pow(expert_errors_y, 2), dim=2)  # (B, num_experts)
        expert_weights = F.softmax(-expert_errors, dim=1)  # (B, num_experts)

        return expert_weights

    
class SaMoEModel(MoEModel_Exp):
    """
    SaMoE model that extends MoEModel_Exp with expert add/prune method.
    """
    @override
    def __init__(self, num_experts=4, context_size=8, input_size=1, hidden_size=32, output_size=1):
        super(SaMoEModel, self).__init__(num_experts=num_experts, context_size=context_size, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

        self.expert_trace = torch.ones(num_experts)  # Initialize with 1 to avoid division by zero

    @override
    def forward(self, context, input):
        expert_weights = self._get_expert_weights(context)
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)
        combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)
        
        # Update expert trace based on the frequency of activation
        with torch.no_grad():
            self.expert_trace = self.expert_trace + torch.sum(expert_weights, dim=0)
        
        return combined_output

    def evolve_experts(self, threshold=0.1):
        """
        Evolve experts based on their frequency of activation.
        This method can be called periodically to update the experts.
        """
        # Step 1: Prune
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        remove_mask = expert_priority < threshold
        remove_indices = torch.where(remove_mask)[0].tolist()
        print(f"Current expert frequencies: {expert_priority.detach()}")
        print(f"Experts to remove (freq < {threshold:.4f}): {remove_indices}")
        
        if len(remove_indices) > 0:
            # Remove experts with low frequency
            for idx in sorted(remove_indices, reverse=True):
                del self.experts[idx]
            self.expert_trace = self.expert_trace[~remove_mask]
        
        # Step 2: Add
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        add_mask = expert_priority > 0.3 / max(threshold, 0.001)  # The 1 / threshold
        add_indices = torch.where(add_mask)[0].tolist()
        print(f"Experts to add (freq > {(0.3 / max(threshold, 0.001)):.4f}): {add_indices}")
        
        if len(add_indices) > 0:
            for idx in sorted(add_indices, reverse=True):
                new_expert = copy.deepcopy(self.experts[idx])
                with torch.no_grad():  # add small noise to the new expert
                    for param in new_expert.parameters():
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)
                self.experts.append(new_expert)
                self.expert_trace[idx] /= 2
                self.expert_trace = torch.cat([self.expert_trace, torch.ones(1) * self.expert_trace[idx]])
        
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        print(f"Updated expert frequencies: {expert_priority.detach()}")

class SaMoEModel_Ab1(SaMoEModel):
    """
    SaMoE model with a different expert weights calculation method.
    The weight is weight / trace to make the trace more equalized.
    """
    @override
    def forward(self, context, input):
        trace_weights = self.expert_trace.unsqueeze(0)
        expert_weights = self._get_expert_weights(context)
        expert_weights = expert_weights / torch.sqrt(trace_weights)
        # Ensure expert_weights is not zero to avoid division by zero
        expert_weights = F.softmax(expert_weights, dim=1)
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)
        combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)
        
        # Update expert trace based on the frequency of activation
        with torch.no_grad():
            self.expert_trace = self.expert_trace + torch.sum(expert_weights, dim=0)
        
        return combined_output

class SaMoEModel_Ab2(SaMoEModel):
    """
    SaMoE model with a different expert weights calculation method.
    The weight is prior * bayes.
    """
    @override
    def __init__(self, num_experts=4, context_size=8, input_size=1, hidden_size=32, output_size=1):
        super(SaMoEModel_Ab2, self).__init__(num_experts=num_experts, context_size=context_size, input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        
        self.prior_weights_gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=1)  # Ensure the prior weights sum to 1
        )

    @override
    def _get_expert_weights(self, context, input):
        """
        The expert weights consider both prior and bayesian weights.
        """
        prior_weights = self.prior_weights_gate(input)
        expert_weights = super()._get_expert_weights(context)  # This method is first defined in MoEModel_Exp
        expert_weights = expert_weights * prior_weights
        expert_weights = expert_weights / torch.sum(expert_weights, dim=1, keepdim=True)

        return expert_weights
    
    @override
    def forward(self, context, input):
        expert_weights = self._get_expert_weights(context, input)
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)
        combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)
        
        # Update expert trace based on the frequency of activation
        with torch.no_grad():
            self.expert_trace = self.expert_trace + torch.sum(expert_weights, dim=0)
        
        return combined_output
    
    @override
    def evolve_experts(self, threshold=0.1):
        """
        Evolve experts based on their frequency of activation.
        This method can be called periodically to update the experts.
        """
        # Step 1: Prune
        # Calculate expert priority based on frequency
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        remove_mask = expert_priority < threshold
        remove_indices = torch.where(remove_mask)[0].tolist()
        print(f"Current expert frequencies: {expert_priority.detach()}")
        print(f"Experts to remove (freq < {threshold:.4f}): {remove_indices}")
        
        if len(remove_indices) > 0:
            # Remove experts with low frequency
            for idx in sorted(remove_indices, reverse=True):
                del self.experts[idx]
            self.expert_trace = self.expert_trace[~remove_mask]
            
            # 重建gate层
            old_layer = self.prior_weights_gate[2]
            new_layer = nn.Linear(old_layer.in_features, len(self.experts))
            with torch.no_grad():
                new_layer.weight.data = old_layer.weight[~remove_mask, :].clone()
                new_layer.bias.data = old_layer.bias[~remove_mask].clone()
            self.prior_weights_gate[2] = new_layer
        
        # Step 2: Add
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        add_mask = expert_priority > 0.3 / max(threshold, 0.001)  # The 1 / threshold
        add_indices = torch.where(add_mask)[0].tolist()
        print(f"Experts to add (freq > {(0.3 / max(threshold, 0.001)):.4f}): {add_indices}")
        
        # Get the prior weights from the gate
        if len(add_indices) > 0:
            for idx in sorted(add_indices, reverse=True):
                # Create a new expert by copying an existing one
                new_expert = copy.deepcopy(self.experts[idx])
                with torch.no_grad():
                    for param in new_expert.parameters():
                        noise = torch.randn_like(param) * 0.01
                        param.add_(noise)
                self.experts.append(new_expert)
                # Update the expert trace
                self.expert_trace[idx] /= 2
                self.expert_trace = torch.cat([self.expert_trace, torch.ones(1) * self.expert_trace[idx]])
            
            # 重建gate层以容纳新专家
            old_layer = self.prior_weights_gate[2]
            new_layer = nn.Linear(old_layer.in_features, len(self.experts))
            
            with torch.no_grad():
                # 复制现有权重
                new_layer.weight.data[:old_layer.out_features] = old_layer.weight.data
                new_layer.bias.data[:old_layer.out_features] = old_layer.bias.data
                
                # 为新专家添加权重（复制来源专家的权重）
                start_idx = old_layer.out_features
                for i, source_idx in enumerate(add_indices):
                    new_layer.weight.data[start_idx + i] = old_layer.weight.data[source_idx]
                    new_layer.bias.data[start_idx + i] = old_layer.bias.data[source_idx]
            
            self.prior_weights_gate[2] = new_layer
  
        # Update and print the priority
        expert_priority = len(self.experts) * self.expert_trace / torch.sum(self.expert_trace)
        print(f"Updated expert frequencies: {expert_priority.detach()}")
        