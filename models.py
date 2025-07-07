"""
models.py
This module defines the structure of two models: The MoEModel and the End2EndModel.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # self.gate = nn.Sequential(
        #     nn.Linear(context_size, num_experts),
        #     nn.ReLU(),
        #     nn.Linear(num_experts, num_experts)
        # )

    def forward(self, context, input):
        num_experts = len(self.experts)
        # context dimension is (B, 4, 2), separate into (B, 4) and (B, 4)
        context_x, context_y = context[:, 0::2], context[:, 1::2]  # (B, 4), (B, 4)
        flattened_context_x, flattened_context_y = context_x.view(-1, 1), context_y.view(-1, 1)  # (B*4, 1), (B*4, 1)
        
        # Use torch.stack instead of torch.tensor to preserve gradients
        expert_predictions = torch.stack([expert(flattened_context_x) for expert in self.experts], dim=0)  # (num_experts, B*4, 1)
        flattened_context_y_expanded = flattened_context_y.unsqueeze(0).expand(num_experts, -1, 1)  # (num_experts, B*4, 1)
        error_context_y = (expert_predictions - flattened_context_y_expanded).view(num_experts, -1, 4).transpose(0, 1)  # (B, num_experts, 4)
        
        error_context = torch.sum(torch.pow(error_context_y, 2), dim=2)  # (B, num_experts)
        weights_context = F.softmax(-error_context, dim=1)  # (B, num_experts)
        
        # Also fix the expert outputs calculation
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)  # (B, num_experts, output_size)
        combined_output = torch.sum(weights_context.unsqueeze(2) * expert_outputs, dim=1)  # (B, output_size)
        
        return combined_output
    
class SaMoEModel(nn.Module):
    """
    Scalable Mixture of Experts (SaMoE) model that processes context, input, and output_gt.
    """
    def __init__(self, num_experts=4, context_size=8, input_size=1, hidden_size=32, output_size=1):
        super(SaMoEModel, self).__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(context_size, num_experts)

    def forward(self, context, input):
        gate_output = F.softmax(self.gate(context), dim=1)
        expert_outputs = [expert(input) for expert in self.experts]
        combined_output = sum(gate_output[:, i] * expert_outputs[i] for i in range(len(self.experts)))
        
        return combined_output