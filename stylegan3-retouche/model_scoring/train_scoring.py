import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from model_scoring.model import RatingPredictor
from model_scoring.dataset import RatingDataset
from model_scoring.data_preparation import load_and_prepare_data


def main(args):
    data = load_and_prepare_data(args.csv_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & Dataloaders
    train_set = RatingDataset(data["X_train"], data["y_train"], augment=True)
    val_set   = RatingDataset(data["X_val"], data["y_val"])
    test_set  = RatingDataset(data["X_test"], data["y_test"])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    model = RatingPredictor(input_dim=data["X_train"].shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(200):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch).squeeze()
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                out = model(X_batch).squeeze()
                loss = criterion(out, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds.extend((out.cpu().numpy() * data["y_std"] + data["y_mean"]))
                trues.extend((y_batch.cpu().numpy() * data["y_std"] + data["y_mean"]))
        val_loss /= len(val_loader.dataset)
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(trues))**2))

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSE={rmse:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
