import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import End2EndModel, MoEModel_Exp, MoEModel_Imp, SaMoEModel
import numpy as np
np.random.seed(42)  # For reproducibility
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, train_data_loader, test_data_loader, criterion, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader providing the training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
    """
    model.train()
    train_loss, test_loss = 0.0, 0.0
    
    for input_img, input_loc, context_imgs, context_locs, context_dones, output_done in train_data_loader:
        context = {'imgs': context_imgs.float().to(device) / 255.0,  # Normalize images to [0, 1]
                   'locs': context_locs.float().to(device), 
                   'dones': context_dones.float().to(device)}
        input = {'img': input_img.float().to(device) / 255.0, 'loc': input_loc.float().to(device)}
        output_done = output_done.float().to(device)

        optimizer.zero_grad()
        if isinstance(model, End2EndModel):
            combined_output = model(context, input)
            loss = criterion(combined_output, output_done)
        else:
            expert_weights, expert_outputs = model(context, input)
            combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)

            # outputs_sorted, _ = torch.sort(expert_outputs, dim=1)
            # gaps = outputs_sorted[:, 1:] - outputs_sorted[:, :-1]
            # expected_gaps = torch.ones_like(gaps).to(device) / gaps.size(1) # Assuming uniform distribution
            # expert_entropy = -torch.min(gaps, dim=1)[0].mean()  # Regularization term for expert entropy
            # weight_entropy = -torch.max(expert_weights, dim=1)[0].mean()

            expert_accuracy = F.softmax(-torch.pow(expert_outputs - output_done.unsqueeze(1).repeat(1, len(model.experts), 1), 2), dim=1)  # (batch_size, num_experts)
            accuracy_e = torch.transpose_copy(expert_accuracy.squeeze(-1), 0, 1)
            feats_e = accuracy_e - accuracy_e.mean(dim=1, keepdim=True)
            feats_e = feats_e / (feats_e.std(dim=1, keepdim=True) + 1e-8)
            # expert_errors = F.softmax(-expert_errors, dim=1).squeeze(-1)
            # batch_expert_consistency = -torch.pow(expert_weights - expert_errors, 2)  # Regularization term for expert consistency
            # batch_expert_consistency = F.softmax(batch_expert_consistency, dim=1)  # Normalize to sum to 1
            # batch_data_weights = F.softmax(batch_expert_consistency.detach(), dim=0)  # Normalize to sum to 1

            # output_done = batch_data_weights * output_done
            # combined_output = batch_data_weights * combined_output
            # if max(batch_data_weights) < 0.3 / batch_data_weights.size(0):
            #     print("Some data has been addressed properly: ", batch_data_weights)

            weights_t = torch.transpose_copy(expert_weights, 0, 1)
            feats_w = weights_t - weights_t.mean(dim=1, keepdim=True)
            feats_w = feats_w / (feats_w.std(dim=1, keepdim=True) + 1e-8)
            corr_matrix = torch.matmul(feats_w, feats_w.T) / feats_w.shape[1]  # shape: (num_experts, num_experts)
            # 只关注非对角线元素（即不同expert之间的相关性）
            mask_w = ~torch.eye(corr_matrix.size(0), dtype=torch.bool, device=corr_matrix.device)
            mean_corr = corr_matrix[mask_w].mean()
            max_corr = corr_matrix[mask_w].max()

            cons_matrix = torch.matmul(feats_e, feats_w.T) / feats_w.shape[1]
            mask_e = torch.eye(corr_matrix.size(0), dtype=torch.bool, device=corr_matrix.device)
            
            loss = 1.0 * criterion(combined_output, output_done) 
            + 5.0 * corr_matrix[mask_w].mean()
            # - 0.5 * torch.max(expert_weights, dim=1)[0].mean()
            # + 5.0 * expert_entropy
            - 5.0 * cons_matrix[mask_e].mean()
            # + 0.3 * batch_expert_consistency.min(dim=1)[0].mean()
            # - 0.3 * batch_expert_consistency.max(dim=0)[0].mean()
            # + 0.3 * batch_expert_consistency.min(dim=0)[0].mean()
            # + 0.2 * weight_entropy
            # Add entropy regularization

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        for input_img, input_loc, context_imgs, context_locs, context_dones, output_done in test_data_loader:
            context = {'imgs': context_imgs.float().to(device) / 255.0, 
                       'locs': context_locs.float().to(device), 
                       'dones': context_dones.float().to(device)}
            input = {'img': input_img.float().to(device) / 255.0, 'loc': input_loc.float().to(device)}
            output_done = output_done.float().to(device)
            
            if isinstance(model, End2EndModel):
                combined_output = model(context, input)
            else:
                expert_weights, expert_outputs = model(context, input)
                combined_output = torch.sum(expert_weights.unsqueeze(2) * expert_outputs, dim=1)
            loss = criterion(combined_output, output_done)
            test_loss += loss.item()
    
    return train_loss / len(train_data_loader), test_loss / len(test_data_loader)

def train(model, train_dataset, test_dataset, batch_size=32, num_epochs=100, learning_rate=0.001):
    """
    Train the model on the provided training data.
    
    Args:
        model (nn.Module): The model to train.
        train_data (tuple): Tuple containing context, input_data, and output_gt.
        batch_size (int): Size of each training batch.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
    """
    model.to(device)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss, test_loss = train_epoch(model, train_data_loader, test_data_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        if isinstance(model, SaMoEModel):
            current_num_experts = len(model.experts)
            model.evolve_experts(threshold=min(epoch / num_epochs, 0.25))

            # 检查是否需要更新optimizer
            if len(model.experts) != current_num_experts:
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                print(f"Optimizer updated: {current_num_experts} -> {len(model.experts)} experts")

def main():
    # Load dataset from .npz file
    print("Loading dataset...")
    data = np.load('v2/train_dataset.npz')
    input_img = torch.tensor(data['input_img'])
    input_loc = torch.tensor(data['input_loc'])
    context_imgs = torch.tensor(data['context_imgs'])
    context_locs = torch.tensor(data['context_locs'])
    context_dones = torch.tensor(data['context_dones'])
    output_done = torch.tensor(data['output_done'])

    train_dataset = TensorDataset(input_img, input_loc, context_imgs, context_locs, context_dones, output_done)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

    data = np.load('v2/test_dataset.npz')
    input_img = torch.tensor(data['input_img'])
    input_loc = torch.tensor(data['input_loc'])
    context_imgs = torch.tensor(data['context_imgs'])
    context_locs = torch.tensor(data['context_locs'])
    context_dones = torch.tensor(data['context_dones'])
    output_done = torch.tensor(data['output_done'])

    test_dataset = TensorDataset(input_img, input_loc, context_imgs, context_locs, context_dones, output_done)
    
    # Train the model
    # model = End2EndModel(hidden_size=32)
    # model = MoEModel_Exp(num_experts=16, hidden_size=16)
    model = SaMoEModel(num_experts=16, hidden_size=32)
    print("Model initialized.")
    train(model, train_dataset, test_dataset, batch_size=16, num_epochs=25, learning_rate=0.001)

    # Save the trained model
    torch.save(model.state_dict(), 'v2/model_sam.pth')
    print("Model saved to 'model_sam.pth'.")

if __name__ == "__main__":
    main()
