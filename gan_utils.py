import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def generate_synthetic_data(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
    fake_data = generator.predict(noise)
    return fake_data

def monitor_generator(generator, df_fraud):
    pca = PCA(n_components=2)
    real_fraud_data = df_fraud.drop("Class", axis=1)
    transformed_data_real = pca.fit_transform(real_fraud_data.values)
    df_real = pd.DataFrame(transformed_data_real)
    df_real['label'] = "real"
    synthetic_fraud_data = generate_synthetic_data(generator, len(df_fraud))
    transformed_data_fake = pca.fit_transform(synthetic_fraud_data)
    df_fake = pd.DataFrame(transformed_data_fake)
    df_fake['label'] = "fake"
    df_combined = pd.concat([df_real, df_fake])
    plt.figure()
    sns.scatterplot(data=df_combined, x=0, y=1, hue='label', s=10)
    plt.show()
