from Training.loadDataset import flowfusionDataset
import torch
from torch.utils.data  import DataLoader
from Training.transformer import transformer
import torch.optim.lr_scheduler as lr_scheduler
from Training.loadDataset import dynamic_collate_fn
from  torch.nn.utils import clip_grad_norm_

def train(learning_rate, batch_size, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = flowfusionDataset('Dataset_UCF101/sorted_flow_poses/train')
    test_dataset = flowfusionDataset('Dataset_UCF101/sorted_flow_poses/test')

    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle = True, collate_fn=dynamic_collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle = True,  collate_fn=dynamic_collate_fn)

    model = transformer(d_model=512, num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, input_dim=1700, num_classes=len(train_dataset.classes)).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    
    for epoch in range (epochs):
        model.train()

        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs}, Loss: {loss}")

        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0 
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                val_loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)

                total_val_loss += val_loss.item()

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                avg_loss = total_val_loss/len(test_loader)

        print(f'Validation Loss: {avg_loss}')
        print(f"Test Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%")

    print("Training Complete")
    torch.save(model.state_dict(), 'model_XYZ.pth')
               
if __name__ == "__main__":
    train(1e-4, 32, 20)
    