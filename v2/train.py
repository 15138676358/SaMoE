import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import End2EndModel, MoEModel_Exp, MoEModel_Imp
import numpy as np
from tqdm import tqdm

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
        context = {'imgs': context_imgs.float(), 
                   'locs': context_locs.float(), 
                   'dones': context_dones.float()}
        input = {'img': input_img.float(), 'loc': input_loc.float()}
        output_done = output_done.float()

        optimizer.zero_grad()
        output_pred = model(context, input)
        loss = criterion(output_pred, output_done)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        for input_img, input_loc, context_imgs, context_locs, context_dones, output_done in test_data_loader:
            context = {'imgs': context_imgs.float(), 
                       'locs': context_locs.float(), 
                       'dones': context_dones.float()}
            input = {'img': input_img.float(), 'loc': input_loc.float()}
            output_done = output_done.float()
            
            output_pred = model(context, input)
            loss = criterion(output_pred, output_done)
            test_loss += loss.item()
    
    return train_loss / len(train_data_loader), test_loss / len(test_data_loader)

def train(model, train_data, test_data, batch_size=32, num_epochs=100, learning_rate=0.001):
    """
    Train the model on the provided training data.
    
    Args:
        model (nn.Module): The model to train.
        train_data (tuple): Tuple containing context, input_data, and output_gt.
        batch_size (int): Size of each training batch.
        num_epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
    """
    input_img, input_loc, context_imgs, context_locs, context_dones, output_done = train_data
    train_dataset = TensorDataset(input_img, input_loc, context_imgs, context_locs, context_dones, output_done)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    input_img, input_loc, context_imgs, context_locs, context_dones, output_done = test_data
    test_dataset = TensorDataset(input_img, input_loc, context_imgs, context_locs, context_dones, output_done)
    test_data_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        train_loss, test_loss = train_epoch(model, train_data_loader, test_data_loader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # if isinstance(model, SaMoEModel):
        #     current_num_experts = len(model.experts)
        #     model.evolve_experts(threshold=0.2 * epoch / num_epochs)

        #     # 检查是否需要更新optimizer
        #     if len(model.experts) != current_num_experts:
        #         optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        #         print(f"Optimizer updated: {current_num_experts} -> {len(model.experts)} experts")

def main():
    # Load dataset from .npz file
    print("Loading dataset...")
    data = np.load('v2/dataset.npz')
    input_img = torch.tensor(data['input_img'])
    input_loc = torch.tensor(data['input_loc'])
    context_imgs = torch.tensor(data['context_imgs'])
    context_locs = torch.tensor(data['context_locs'])
    context_dones = torch.tensor(data['context_dones'])
    output_done = torch.tensor(data['output_done'])

    # Split data into training and testing sets
    train_size = int(0.8 * len(output_done))
    test_size = len(output_done) - train_size
    train_data = (input_img[:train_size], input_loc[:train_size],
                  context_imgs[:train_size], context_locs[:train_size],
                  context_dones[:train_size], output_done[:train_size])
    test_data = (input_img[train_size:], input_loc[train_size:],
                 context_imgs[train_size:], context_locs[train_size:],
                 context_dones[train_size:], output_done[train_size:])
    print(f"Training on {len(train_data[0])} samples, Testing on {len(test_data[0])} samples")
    
    # Train the model
    # model = End2EndModel()  # or MoEModel_Exp() or MoEModel_Imp()
    model = MoEModel_Exp(num_experts=16)  # Uncomment if using SaMoEModel
    print("Model initialized")
    train(model, train_data, test_data, batch_size=32, num_epochs=25, learning_rate=0.001)

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to 'model.pth'")

if __name__ == "__main__":
    main()
