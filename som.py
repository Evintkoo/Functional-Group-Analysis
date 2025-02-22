import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from torchinfo import summary
from tqdm import tqdm
from SOM_plus_clustering.modules.som import SOM
from converter import convert_smiles_to_matrix
from variables import selected_feature

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("classic_autoencoder.log"),
        logging.StreamHandler()
    ]
)


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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def load_and_preprocess_data(filepath: str, column_encoder: dict):
    """Load, preprocess, and normalize the dataset."""
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}.")
    except FileNotFoundError as e:
        logging.error(f"Error: {e}. Please check the file path.")
        raise

    df['label'] = df['label'].map(column_encoder)
    X = df.drop('label', axis=1)
    y = df['label']

    # Normalize data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    logging.info("Data normalization completed.")

    return X.values, y.values


def encode_data(autoencoder, X: np.ndarray):
    """Encode data using the trained Classic Autoencoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device).eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        encoded, _ = autoencoder(X_tensor)
    logging.info("Data encoding completed.")
    
    return encoded.cpu().numpy()

def decode_data(autoencoder, X: np.ndarray):
    """Encode data using the trained Classic Autoencoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device).eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        encoded = autoencoder.decoder(X_tensor)
    logging.info("Data encoding completed.")
    
    return encoded.cpu().numpy()

def process_batch(batch_df, autoencoder, som_model, output_dir, batch_num):
    """Process each batch: vectorize, encode, cluster, and save the results."""
    batch_df = batch_df.drop("qed", axis=1)
    X = batch_df.values
    # Encode data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenized = encode_data(autoencoder=autoencoder, X=X)

    # Cluster data using SOM
    som_model.fit(x=tokenized, epoch=128)
    predictions = som_model.predict(tokenized)
    
    cluster_centers = som_model.cluster_center_
    
    decoded_cc = decode_data(autoencoder=autoencoder, X=cluster_centers)
    decoded_df = pd.DataFrame(decoded_cc, columns=selected_feature)
    
    batch_df['label'] = predictions

    # Save results
    os.makedirs(f"{output_dir}/group_{batch_num}", exist_ok=True)
    np.save(f"{output_dir}/group_{batch_num}/cluster_center.npy", np.array(som_model.cluster_center_))
    batch_df.to_csv(f"{output_dir}/group_{batch_num}/labeled_data.csv", index=False)
    decoded_df.to_csv(f"{output_dir}/group_{batch_num}/decoded_data.csv", index=False)
    pd.DataFrame(X).to_csv(f"{output_dir}/group_{batch_num}/matrix_data.csv", index=False)
    pd.DataFrame(tokenized).to_csv(f"{output_dir}/group_{batch_num}/tokenized_data.csv", index=False)
    logging.info(f"Batch {batch_num} processed and saved.")


def train_som():
    # Configuration
    config = {
        'output_dir': "results/clustering",
        'filepath': os.getenv("TRAIN_DATASET_PATH"),
        'test_size': 0.3,
        'random_state': 42,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'epochs': 128,
        'input_size': 28,
        'hidden_sizes': eval(os.getenv("AUTOENCODER_HIDDEN_SIZE")),
        'autoencoder_path': os.getenv('AUTOENCODER_MODEL_PATH'),
        'column_encoder': eval(os.getenv("COLUMN_ENCODER")),
        'step_size': 10,
        'gamma': 0.5
    }

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load autoencoder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = ClassicAutoencoder(config['input_size'], config['hidden_sizes']).to(device)
    
    try:
        autoencoder.load_state_dict(torch.load(config['autoencoder_path'], map_location=device))
        logging.info(f"Autoencoder model loaded successfully from {config['autoencoder_path']}.")
    except FileNotFoundError as e:
        logging.error(f"Error loading autoencoder model: {e}. Please check the model path.")
        raise

    # Read and process the data
    df = pd.read_csv("C:/Users/Evint/Documents/Projects/Functional-Group-Analysis/250k_rndm_zinc_drugs_clean_3.csv")
    df_x = pd.read_csv("data\experiment_3\clean_data.csv")
    df_x = df_x[selected_feature]
    df_x["qed"] = df["qed"]
    df = df_x
    labels = df["qed"]
    
    # Define edges for grouping druglikeness values
    edge_list = [0.39941986, 0.51981407, 0.69371681, 0.81411102]
    limits = [(0, edge_list[0])] + [(edge_list[i-1], edge_list[i]) for i in range(1, len(edge_list))] + [(edge_list[-1], 1)]
    
    som_model = SOM(m=10, n=10,
                    dim=config['hidden_sizes'][-1], 
                    initiate_method="kde", learning_rate=0.5, 
                    neighbour_rad=np.sqrt(6*6 + 5*5), distance_function="euclidean")

    # Process each group defined by the limits
    for counter, (lower_limit, upper_limit) in enumerate(tqdm(limits, desc="Processing Batches")):
        filtered_df = df[(df['qed'] >= lower_limit) & (df['qed'] < upper_limit)]
        logging.info(f"Processing batch {counter} with size {filtered_df.shape[0]}.")
        process_batch(filtered_df, autoencoder, som_model, config['output_dir'], counter)

def main():
    train_som()

if __name__ == "__main__":
    main()
