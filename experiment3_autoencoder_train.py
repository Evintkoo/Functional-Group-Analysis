import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from torchinfo import summary  # Ensure torchinfo is installed using pip install torchinfo
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ClassicAutoencoder(nn.Module):
    """Classic Autoencoder with ReLU activations."""

    def __init__(self, input_size, hidden_sizes):
        super(ClassicAutoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            encoder_layers.append(nn.Tanh())  # Add ReLU after each linear layer
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        for i in range(len(hidden_sizes) - 1, -1, -1):
            if i == 0:
                decoder_layers.append(nn.Linear(hidden_sizes[i], input_size))
                decoder_layers.append(nn.Tanh())  # Add ReLU after each linear layer except the final output
            else:
                decoder_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i - 1]))
                decoder_layers.append(nn.Tanh())  # Add ReLU after each linear layer except the final output
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_and_preprocess_data(filepath: str, column_encoder: dict):
    """Load, preprocess, and normalize the dataset from the given file path."""
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please check the file path.")
        raise

    df['label'] = df['label'].map(column_encoder)
    X = df.drop('label', axis=1)
    y = df['label']

    # Normalize data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    logging.info(f"Data loaded and normalized successfully from {filepath}.")
    return X, y


def save_model_summary(autoencoder, config):
    """Save the model summary to a text file."""
    model_summary = summary(autoencoder, input_size=(config['batch_size'], config['input_size']), verbose=0)
    summary_path = os.path.join(config['output_dir'], "model_summary.txt")
    # Open the file with utf-8 encoding
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(str(model_summary))
    logging.info(f"Model summary saved to {summary_path}")


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, config: dict):
    """Train a Classic Autoencoder on the provided training data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = ClassicAutoencoder(config['input_size'], config['hidden_sizes']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Learning rate scheduler with stepper
    step_size = config.get('step_size', 10)  # Default step size of 10 epochs
    gamma = config.get('gamma', 0.5)  # Default reduction factor of 0.5
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Save model summary
    save_model_summary(autoencoder, config)

    # Prepare data loaders
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Initialize lists to store losses
    epoch_losses = []

    # CSV file for recording training progress
    loss_csv_path = os.path.join(config['output_dir'], "training_losses.csv")
    # Initialize CSV with headers
    pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss']).to_csv(loss_csv_path, index=False, mode='w', encoding='utf-8')

    best_loss = float('inf')
    for epoch in range(config['epochs']):
        autoencoder.train()
        running_loss = 0.0
        for inputs, _ in train_dataloader:
            optimizer.zero_grad()
            _, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_dataloader)

        # Validation phase
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_dataloader:
                _, outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)

        # Log epoch losses
        epoch_losses.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})
        logging.info(f'Epoch [{epoch + 1}/{config["epochs"]}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(autoencoder.state_dict(), config['autoencoder_path'])
            logging.info(f"Best model saved with validation loss: {best_loss:.4f}")

        # Adjust learning rate using the stepper
        scheduler.step()

        # Update CSV file with epoch results
        loss_df = pd.DataFrame([{'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss}])
        loss_df.to_csv(loss_csv_path, index=False, mode='a', header=False, encoding='utf-8')

    logging.info("Autoencoder training completed.")

    return autoencoder


def encode_data(autoencoder, X: np.ndarray):
    """Encode data using the trained Classic Autoencoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device).eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        encoded, _ = autoencoder(X_tensor)
    logging.info("Data encoding completed.")
    return encoded.cpu().numpy()


def run_train():
    config = {
        'output_dir': "results/classic_autoencoder",
        'test_size': 0.3,
        'filepath': os.getenv("TRAIN_DATASET_PATH"),
        'random_state': 42,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'epochs': 256,
        'input_size': None,
        'hidden_sizes': eval(os.getenv("AUTOENCODER_HIDDEN_SIZE")),
        'autoencoder_path': os.getenv('AUTOENCODER_MODEL_PATH'),
        'column_encoder': eval(os.getenv("COLUMN_ENCODER")),
        'step_size': 10,  # Step size for learning rate reduction
        'gamma': 0.5  # Reduction factor for learning rate
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    # Load and preprocess data
    X, y = load_and_preprocess_data(config['filepath'], config['column_encoder'])
    config['input_size'] = X.shape[1]

    # Split data
    X_train, X_val, _, _ = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_state'])

    # Train autoencoder
    autoencoder = train_autoencoder(X_train.values, X_val.values, config)

    # Encode data
    X_train_encoded = encode_data(autoencoder, X_train.values)
    X_val_encoded = encode_data(autoencoder, X_val.values)

    logging.info("Data encoding process completed successfully.")

def main():
    run_train()

if __name__ == "__main__":
    main()
