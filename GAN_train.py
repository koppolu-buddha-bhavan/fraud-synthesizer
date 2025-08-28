from GAN_model import build_discriminator,build_generator,build_gan,generate_synthetic_data,monitor_generator
from data import df_fraud
import numpy as np
def train_gan():
    generator = build_generator()
    discriminator = build_discriminator()
    gan = build_gan(generator, discriminator)
    gan.compile(optimizer ='adam', loss = 'binary_crossentropy')

    num_epochs = 1000
    batch_size = 64
    half_batch = int(batch_size/2)

    for epoch in range(num_epochs):
        X_fake = generate_synthetic_data(generator, half_batch)
        y_fake = np.zeros((half_batch, 1))
    
        X_real = df_fraud.drop("Class", axis = 1).sample(half_batch)
        y_real = np.ones((half_batch, 1))
    
        discriminator.trainable = True
        discriminator.train_on_batch(X_real, y_real)
        discriminator.train_on_batch(X_fake, y_fake)
    
        noise = np.random.normal(0,1,(batch_size, 29))
        gan.train_on_batch(noise, np.ones((batch_size, 1)))
    
        if epoch%10 == 0:
            monitor_generator(generator, epoch)

    # Save the trained generator model
    generator.save("generator_model.h5")

train_gan()