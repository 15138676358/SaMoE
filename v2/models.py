"""
The moe models v2 for grasping dataset.
The data structure is as follows:
- context: dict({attempt: dict({img: np.ndarray, loc: (int, int), done: bool})})
- input: img: np.ndarray
- output_gt: done: bool
The experts and gate network are CNNs, and the baseline network is a flatten network with multiple CNN modules.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import override

class ImgModule(nn.Module):
    """
    A CNN module for processing images.
    - input: torch.FloatTensor(batch_size, 352, 352, 3) representing the image
    - output: torch.FloatTensor(batch_size, hidden_size) representing the processed image features
    """
    def __init__(self, hidden_size=32):
        super(ImgModule, self).__init__()
        self.model = nn.Sequential(
            # 352x352x3 -> 88x88x8
            nn.Conv2d(3, 8, kernel_size=7, stride=4, padding=3),  # stride=4大幅降维
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 88 -> 44

            # 44x44x8 -> 22x22x16
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 44 -> 22
            
            # 22x22x16 -> 11x11xhidden_size
            nn.Conv2d(16, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 22 -> 11
            
            # 11x11xhidden_size -> 1x1xhidden_size
            nn.AdaptiveAvgPool2d(1),
            nn.ReLU()
        )

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        output = self.model(img)
        
        return output.view(img.size(0), -1)


class InputExpert(nn.Module):
    """
    Expert network that processes input data and produces an output.
    - input: Dict{
                img: torch.FloatTensor(batch_size, 352, 352, 3) representing the image
                loc: torch.FloatTensor(batch_size, 2) representing the grasping location. "grasp_wrt_crop" in the json file.
    - output: torch.FloatTensor(batch_size, 1)
    """
    def __init__(self, hidden_size=32):
        super(InputExpert, self).__init__()
        self.img_module = ImgModule(hidden_size)
        # MLP
        self.decode_module = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        img, loc = input['img'], input['loc']
        img_features = self.img_module(img)  # (batch_size, hidden_size)
        combined_features = torch.cat([img_features, loc], dim=1)
        output = self.decode_module(combined_features)
        
        return output

class ContextExpert(nn.Module):
    """
    Transformer-based context expert that processes sequential context data.
    - input: context with shape (batch_size, seq_len, context_features)
            where each context item contains {img, loc, done}
    - output: torch.FloatTensor(batch_size, hidden_size)
    """
    def __init__(self, hidden_size=32, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=10):
        super(ContextExpert, self).__init__()  # 修复：原来错误地调用了InputExpert
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.img_module = ImgModule(hidden_size)
        self.loc_embedding = nn.Linear(2, hidden_size // 4)  # 位置坐标嵌入
        self.done_embedding = nn.Embedding(2, hidden_size // 4)  # done状态嵌入 (True/False)
        # 特征融合层：将img特征、loc特征、done特征融合
        self.feature_fusion = nn.Linear(hidden_size + hidden_size // 4 + hidden_size // 4, hidden_size)
        # 位置编码
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, hidden_size))
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True  # 使用batch_first=True，输入格式为(batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # 最终输出层：将序列压缩为单个向量
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        # 序列聚合方式
        self.aggregation_method = 'attention'  # 可选: 'mean', 'max', 'last', 'attention'
        if self.aggregation_method == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

    def forward(self, context):
        """
        Forward pass for context processing.
        
        Args:
            context: Dict containing:
                - 'imgs': (batch_size, seq_len, 3, 352, 352)
                - 'locs': (batch_size, seq_len, 2)
                - 'dones': (batch_size, seq_len) - boolean values
                
        Returns:
            output: (batch_size, hidden_size)
        """
        imgs = context['imgs']    # (batch_size, seq_len, 3, 352, 352)
        locs = context['locs']    # (batch_size, seq_len, 2)
        dones = context['dones']  # (batch_size, seq_len)
        batch_size, seq_len = imgs.shape[0], imgs.shape[1]
        # 重塑图像数据用于批量处理
        img_reshaped = imgs.view(batch_size * seq_len, 3, 352, 352)
        img_features = self.img_module(img_reshaped)  # (batch_size * seq_len, hidden_size)
        img_features = img_features.view(batch_size, seq_len, self.hidden_size)
        loc_features = self.loc_embedding(locs.float())  # (batch_size, seq_len, hidden_size//4)
        done_int = dones.long()  # (batch_size, seq_len, 1)
        done_features = self.done_embedding(done_int.squeeze(-1))  # (batch_size, seq_len, hidden_size//4)
        combined_features = torch.cat([img_features, loc_features, done_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # (batch_size, seq_len, hidden_size)
        seq_features = fused_features + self.positional_encoding[:seq_len].unsqueeze(0)
        
        # 创建padding mask（如果需要处理变长序列）
        # 这里假设所有序列都是相同长度，如果需要处理变长序列，需要传入mask
        src_key_padding_mask = None
        transformer_output = self.transformer_encoder(
            seq_features, 
            src_key_padding_mask=src_key_padding_mask
        )  # (batch_size, seq_len, hidden_size)
        
        # 序列聚合
        if self.aggregation_method == 'mean':
            aggregated = torch.mean(transformer_output, dim=1)
        elif self.aggregation_method == 'max':
            aggregated = torch.max(transformer_output, dim=1)[0]
        elif self.aggregation_method == 'last':
            aggregated = transformer_output[:, -1, :]
        elif self.aggregation_method == 'attention':
            # 注意力池化
            attention_weights = self.attention_pooling(transformer_output)  # (batch_size, seq_len, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            aggregated = torch.sum(transformer_output * attention_weights, dim=1)  # (batch_size, hidden_size)
        else:
            aggregated = torch.mean(transformer_output, dim=1)
        
        output = self.output_projection(aggregated)  # (batch_size, hidden_size)
        
        return output

class End2EndModel(nn.Module):
    def __init__(self, hidden_size=32):
        super(End2EndModel, self).__init__()
        context = ContextExpert(hidden_size=hidden_size)
        input = InputExpert(hidden_size=hidden_size)
        input.decode_module = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU()
        )  # In this module, the input outputs hidden_size instead of 1
        gate = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Ensure output is in the range [0, 1]
        )
        self.model = nn.ModuleDict({'context': context, 'input': input, 'gate': gate})

    def forward(self, context, input):
        context_features = self.model['context'](context)
        input_features = self.model['input'](input)
        combined_features = torch.cat([context_features, input_features], dim=1)
        output = self.model['gate'](combined_features)

        return output
    
class MoEModel(nn.Module):
    """
    Mixture of Experts (MoE) model that processes context, input, and output_gt.
    """
    def __init__(self, num_experts=4, hidden_size=32):
        super(MoEModel, self).__init__()
        self.experts = nn.ModuleList([InputExpert(hidden_size) for _ in range(num_experts)])

    @abstractmethod
    def get_expert_weights(self, context, input):
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, context, input):
        expert_weights = self.get_expert_weights(context, input)  # (batch_size, num_experts)
        expert_outputs = torch.stack([expert(input) for expert in self.experts], dim=1)  # (batch_size, num_experts, output_size)
        combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)

        return combined_output
    
class MoEModel_Imp(MoEModel):
    @override
    def __init__(self, num_experts=4, hidden_size=32):
        super(MoEModel_Imp, self).__init__(num_experts, hidden_size)
        context = ContextExpert(hidden_size=hidden_size)
        input = InputExpert(hidden_size=hidden_size)
        input.decode_module = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size),
            nn.ReLU()
        )  # In this module, the input outputs hidden_size instead of 1
        gate = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_experts),
            nn.Softmax(dim=-1)
        )
        self.expert_weights_gate = nn.ModuleDict({'context': context, 'input': input, 'gate': gate})

    @override
    def get_expert_weights(self, context, input):
        context_features = self.expert_weights_gate['context'](context)
        input_features = self.expert_weights_gate['input'](input)
        combined_features = torch.cat([context_features, input_features], dim=1)
        expert_weights = self.expert_weights_gate['gate'](combined_features)

        return expert_weights
    
class MoEModel_Exp(MoEModel):
    @override
    def __init__(self, num_experts=4, hidden_size=32):
        super(MoEModel_Exp, self).__init__(num_experts, hidden_size)
        self.prior_weights_gate = InputExpert(hidden_size=hidden_size)
        self.prior_weights_gate.decode_module = nn.Sequential(
            nn.Linear(hidden_size + 2, num_experts),
            nn.ReLU(),
            nn.Softmax(dim=-1)
        )  # In this module, the input outputs num_experts instead of 1
    
    @override
    def get_expert_weights(self, context, input):
        num_experts = len(self.experts)
        imgs, locs, dones = context['imgs'], context['locs'], context['dones']
        batch_size, seq_len = imgs.shape[0], imgs.shape[1]
        C, H, W = imgs.shape[2], imgs.shape[3], imgs.shape[4]
        imgs, locs, dones = imgs.view(batch_size * seq_len, C, H, W), locs.view(batch_size * seq_len, -1), dones.view(batch_size * seq_len, -1)
        context_input, context_output = {'img': imgs, 'loc': locs}, dones

        expert_predictions = torch.stack([expert(context_input) for expert in self.experts], dim=0)  # (num_experts, batch_size * seq_len, 1)
        context_output = context_output.unsqueeze(0).expand(num_experts, -1, 1)  # (num_experts, batch_size * seq_len, 1)
        expert_errors = (expert_predictions - context_output).view(num_experts, -1, 4).transpose(0, 1)  # (batch_size, num_experts, seq_len)
        
        expert_errors = torch.sum(torch.pow(expert_errors, 2), dim=2)  # (batch_size, num_experts)
        expert_weights = F.softmax(-expert_errors, dim=1)  # (batch_size, num_experts)

        prior_weights = self.prior_weights_gate(input)
        expert_weights = expert_weights * prior_weights
        expert_weights = expert_weights / torch.sum(expert_weights, dim=1, keepdim=True)  # Normalize weights

        return expert_weights
