# 🩺 Disease Prediction Web Application

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Framework-black?logo=flask)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SVM-orange)
![Google Gemini API](https://img.shields.io/badge/AI-Gemini%20API-red?logo=google)
![License](https://img.shields.io/badge/License-MIT-green)

> A **Flask-based AI & ML-powered medical assistant** that predicts diseases, generates reports, and allows doctor-like consultations through a chatbot.

---

## ✨ Features

### 🔹 Machine Learning Disease Prediction
- Predicts diseases using **SVM** trained on a **preprocessed Kaggle dataset**.
- Uses checkbox-based symptom selection to avoid input errors.
- Outputs:
  - **Disease name**
  - **Description**
  - **Precautions**
  - **Medications**
  - **Diet plan**
  - **Workout suggestions**
- Generates **hospital-style reports** with a **Download** option.

### 🔹 AI Disease Prediction
- Powered by **Google Gemini API**.
- Predicts diseases and provides **real-time medical advice**.
- Displays results in a **professional medical report** format with **Download** option.

### 🔹 AI Chatbot
- Doctor-like chatbot for medical consultations.
- Uses **Google Gemini API** for intelligent responses.

### 🔹 Contact & Feedback
- Contact page includes:
  - Name, phone, email, social links (**GitHub**, **LinkedIn**, **X/Twitter**).
  - Feedback form using **Web3Forms**.

---

## 📂 Dataset Details
Dataset sourced from **Kaggle** – already preprocessed, so no extra cleaning required.  

**Files Used:**
- `training.csv` – Model training data.
- `description.csv` – Disease descriptions.
- `symptoms_df.csv` – Symptom names.
- `medications.csv` – Recommended medications.
- `diets.csv` – Diet plans.
- `precautions.csv` – Preventive measures.
- `workout.csv` – Workout recommendations.

---

## 🛠 Tech Stack

**Backend**  
- Python (Flask)  
- SVM for ML prediction  
- Google Gemini API for AI prediction & chatbot  

**Frontend**  
- HTML5, CSS3, JavaScript  
- Bootstrap for styling  
- Jinja2 for template rendering  

**Others**  
- Pandas, NumPy for data handling  
- Pickle for storing trained model  
- Web3Forms for feedback form  

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/disease-prediction-app.git
cd disease-prediction-app
