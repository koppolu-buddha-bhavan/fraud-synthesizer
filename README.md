# 🚨 Fraud Data Generation using GANs  
![GAN Illustration](gan.png)

![Python](https://img.shields.io/badge/Python-3.7+-blue?logo=python)  
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)  
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

---

## 📖 Overview  
Fraudulent transactions are **rare, sensitive, and difficult to collect** — making fraud detection a huge challenge in finance & banking.  

This project leverages **Generative Adversarial Networks (GANs)** 🤖 to **generate synthetic fraud data** that mimics real-world fraudulent transactions.  
👉 The synthetic data can be used to **train and test fraud detection models** more effectively.  

---

## 📊 Dataset  
We used the **[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/sowmyakuruba/credit-card-fraud-detection/data)** from Kaggle.  

---

## 🎯 Motivation  
- Fraud data is **highly imbalanced** (fraud cases are extremely rare).  
- GANs can **augment data** by generating realistic fraud samples.  
- Helps build **robust fraud detection models** with better generalization.  

---

## ⚙️ Requirements  
Make sure you have the following dependencies installed:  

```bash
Python >= 3.7
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Scikit-learn
```

Or install directly:  
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage  

1️⃣ Clone this repo:  
```bash
git clone https://github.com/your-username/fraud-data-gan.git
cd fraud-data-gan
```

2️⃣ Start Jupyter Notebook:  
```bash
jupyter notebook
```

3️⃣ Open and run:  
```bash
Fraud_Data_Generation_GAN.ipynb
```

4️⃣ Train the GAN and generate synthetic fraud data.  
📁 The generated dataset will be saved as a **CSV file** for further ML experiments.  

---

## 📂 Project Structure  

```
📦 Fraud-Data-GAN
 ┣ 📜 Fraud_Data_Generation_GAN.ipynb   # Main Notebook
 ┣ 📜 requirements.txt                  # Dependencies
 ┣ 📜 LICENSE                           # MIT License
 ┗ 📜 README.md                         # Project Documentation
```

---

## 📈 Results  

✅ GAN training visualizations:  
- Loss curves for Generator & Discriminator  
- Real vs Synthetic fraud data comparison  

✅ Synthetic fraud data saved in `.csv` format.  

*(Tip: Add screenshots/plots from your notebook here for extra appeal!)*  

---

## 🤝 Contributing  

Contributions are always welcome 💡!  

- Fork this repo  
- Create a new branch (`feature/your-feature`)  
- Commit your changes  
- Open a Pull Request 🚀  

---

## 📜 License  
This project is licensed under the **MIT License**.  
Feel free to use and modify as per your needs.  

---

✨ *If you like this project, give it a ⭐ on GitHub!*  
