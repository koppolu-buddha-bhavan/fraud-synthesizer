from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from data import df_fraud
import matplotlib.pyplot as plt
import plotly.express as px
# Load the trained generator model
generator = load_model("generator_model.h5")

# Generate 1000 synthetic samples
num_samples = 1000
noise = np.random.normal(0, 1, (num_samples, generator.input_shape[1]))
synthetic_data = generator.predict(noise)
syn_df = pd.DataFrame(synthetic_data)
syn_df['label'] = 'fake'
print(syn_df.head())

old_df = df_fraud.drop('Class', axis=1)
old_df['label'] = 'real'
print(old_df.head())

old_df.columns = syn_df.columns
combined_df = pd.concat([syn_df, old_df])
print(combined_df.head())


# Save a PNG for each feature using Plotly in a dedicated folder
import os
img_dir = 'synthetic_feature_images'
os.makedirs(img_dir, exist_ok=True)
for col in combined_df.columns:
  if col == 'label':
    continue
  fig = px.histogram(combined_df, color='label', x=col, barmode="overlay", title=f'Feature {col}', width=640, height=500)
  img_path = os.path.join(img_dir, f'feature_{col}.png')
  fig.write_image(img_path)
  print(f'Saved {img_path}')

# Example plot: count of real vs fake
fig = px.histogram(combined_df, x='label', title='Feature label')
fig.write_image('synthetic_data_plot.png')
print('Plot saved to synthetic_data_plot.html')

# Save the synthetic data for use in the notebook
syn_df.to_csv('synthetic_data.csv', index=False)
print('Synthetic data saved to synthetic_data.csv')