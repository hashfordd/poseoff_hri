from Training.loadDataset import flowfusionDataset
import torch
from torch.utils.data  import DataLoader
from Training.transformer import transformer
import torch.optim.lr_scheduler as lr_scheduler
from Training.loadDataset import dynamic_collate_fn
from  torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import numpy as np

def train(learning_rate, batch_size, epochs):
    print("Running")

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
        loss_arr = []
        for batch_x, batch_y, batch_mask in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, mask=batch_mask)
            loss = criterion(outputs, batch_y)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()


        loss_arr.append(loss.item())
        print(f"Epoch {epoch}/{epochs}, Loss: {loss}")


        model.eval()
        correct = 0
        total = 0
        total_val_loss = 0 
        val_loss_arr = []
        with torch.no_grad():
            for batch_x, batch_y, batch_mask in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                batch_mask = batch_mask.to(device)

                outputs = model(batch_x, mask=batch_mask)
                val_loss = criterion(outputs, batch_y)
                _, predicted = torch.max(outputs.data, 1)

                total_val_loss += val_loss.item()

                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                avg_loss = total_val_loss/len(test_loader)
                
        val_loss_arr.append(avg_loss)

        print(f'Validation Loss: {avg_loss}')
        print(f"Test Accuracy after Epoch {epoch+1}: {100 * correct / total:.2f}%")

    
    loss_arr = np.array(loss_arr)
    val_loss_arr = np.array(val_loss_arr)

    print("Training Complete")
    plt.figure(figsize=(15, 5))
    plt.plot(loss_arr, label = "Training Loss", )
    plt.plot(val_loss_arr, label = "Validation Loss")

    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('loss_curve.png')
    plt.close()
    
    torch.save(model.state_dict(), 'model_XYZ.pth')
               
if __name__ == "__main__":
    train(1e-4, 32, 1)
    
