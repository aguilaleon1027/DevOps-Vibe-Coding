import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import os
from sklearn.metrics import accuracy_score

from data_loader import get_dataloaders
from model import WorkoutRecommender
from data_generator import generate_dummy_data

def train_model(epochs=50, lr=0.001, batch_size=32):
    # Ensure data exists if not generate it
    if not os.path.exists("data/workout_data.csv"):
        print("Data not found. Generating dummy data...")
        generate_dummy_data(num_samples=2000)
        
    # Start MLflow run
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Workout_Recommendation")
    
    with mlflow.start_run():
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", batch_size)
        
        train_loader, test_loader, scaler = get_dataloaders(batch_size=batch_size)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WorkoutRecommender().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        print(f"Training on device: {device}")
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            avg_train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            
            # Auto-Validation loop
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
            avg_val_loss = val_loss / len(test_loader)
            val_acc = accuracy_score(all_labels, all_preds)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f} Val Acc: {val_acc:.4f}")
        
        # Save PyTorch local weights
        os.makedirs("weights", exist_ok=True)
        torch.save(model.state_dict(), "weights/best_model.pth")
        
        # Log model natively to MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Training complete. Best model weights saved at 'weights/best_model.pth'.")

if __name__ == "__main__":
    # Ensure working directory is correctly set to parent if executed from elsewhere
    current_dir = os.path.basename(os.getcwd())
    if current_dir == "ml_pipeline":
        os.chdir("..")
    train_model(epochs=30)
