import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .models.siamese_network import SiameseNetwork
from .data_loader import IrisDataset
from config.config import Config

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = SiameseNetwork(Config.EMBEDDING_DIM).to(device)
    criterion = nn.ContrastiveLoss(margin=Config.MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = IrisDataset(os.path.join(Config.DATA_ROOT, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output1, output2 = model(data[0], data[1])
            loss = criterion(output1, output2, target)
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train_model() 