import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.pytorch
from src.model import FraudClassifier, save_model
# GAN training loop using modularized code
import numpy as np
from model import build_generator, build_discriminator, build_gan
from data import load_data, preprocess_data, get_fraud_genuine
from gan_utils import generate_synthetic_data, monitor_generator

import tensorflow as tf

def train_gan(
    data_path,
    num_epochs=1000,
    batch_size=64,
    generator_input_dim=29
):
    # Load and preprocess data
    df = load_data(data_path)
    df = preprocess_data(df)
    df_fraud, _ = get_fraud_genuine(df)

    # Build models
    generator = build_generator(generator_input_dim)
    discriminator = build_discriminator(generator_input_dim)
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer='adam', loss='binary_crossentropy')

    half_batch = int(batch_size / 2)

    for epoch in range(num_epochs):
        # Generate fake samples
        X_fake = generate_synthetic_data(generator, half_batch)
        y_fake = np.zeros((half_batch, 1))

        # Sample real fraud data
        X_real = df_fraud.drop("Class", axis=1).sample(half_batch)
        y_real = np.ones((half_batch, 1))

        discriminator.trainable = True
        discriminator.train_on_batch(X_real, y_real)
        discriminator.train_on_batch(X_fake, y_fake)

        noise = np.random.normal(0, 1, (batch_size, generator_input_dim))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}")
            monitor_generator(generator, df_fraud)

    return generator, discriminator, gan

def train_model(data_path="data/fraud_synthetic.csv", model_out="models/fraud_model.pkl"):
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Model
    model = FraudClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    with mlflow.start_run():
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        # Eval
        with torch.no_grad():
            preds = model(X_test).numpy()
            auc = roc_auc_score(y_test.numpy(), preds)

        mlflow.log_metric("AUC", auc)
        mlflow.pytorch.log_model(model, "fraud-model")

    save_model(model, model_out)
    print(f"✅ Training complete. Model saved at {model_out}, AUC={auc:.4f}")

if __name__ == "__main__":
    train_model()
