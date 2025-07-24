# 👁️ AI Eye Disease Analyzer

This project is a web-based application designed for the early detection of common eye diseases using artificial intelligence. It features a static landing page that explains the technology and a dynamic Streamlit tool for real-time analysis of retinal scans.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eyediseases.streamlit.app/)

### 🚀 Live Demo

-   **Main Landing Page:** [![Deployed on Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black?style=for-the-badge&logo=vercel)](https://eye-diseases.vercel.app/)
-   **Analyzer Tool:** [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eyediseases.streamlit.app/)

---

## 📋 Table of Contents

-   [Project Description](#-project-description)
-   [Key Features](#-key-features)
-   [How It Works](#-how-it-works)
-   [Technology Stack](#-technology-stack)
-   [Project Structure](#-project-structure)
-   [Setup and Installation](#-setup-and-installation)
-   [Disclaimer](#-disclaimer)

---

## 📝 Project Description

The AI Eye Disease Analyzer is a proof-of-concept tool that demonstrates the power of AI in healthcare. Users can upload a retinal scan image, and the application will use a machine learning model to classify it into one of four categories: **Normal, Cataract, Diabetic Retinopathy, or Glaucoma**. Following the classification, it leverages Google's Gemini AI to provide a detailed explanation, precautions, and recommended next steps in multiple languages, complete with audio narration.

---

## ✨ Key Features

-   **Interactive Landing Page:** A fully responsive HTML/CSS/JS page explaining the project's goals and technology.
-   **AI-Powered Classification:** Utilizes a fine-tuned ResNet-18 model (PyTorch) for accurate image classification.
-   **Generative AI Explanations:** Employs Google's Gemini AI to generate user-friendly and empathetic advice based on the classification result.
-   **Multi-Language Audio Support:** Converts the AI-generated advice into speech in several languages (including English, Hindi, Bengali, etc.) with different accent options.
-   **Modern User Interface:** A clean, intuitive, and visually appealing interface built with Streamlit.

---

## ⚙️ How It Works

The application uses a powerful dual-AI approach:

1.  **Classification AI:** A retinal image uploaded by the user is first processed by a fine-tuned **ResNet-18 Convolutional Neural Network (CNN)**. This model analyzes the image patterns and classifies the eye's condition.
2.  **Generative AI:** The classification result is then passed to **Google's Gemini AI**. A detailed prompt instructs Gemini to act as a health advisor, generating a comprehensive report that explains the condition, lists precautions, and suggests the next steps in a clear and empathetic tone.

---

## 🛠️ Technology Stack

-   **Frontend:** HTML5, CSS3, JavaScript
-   **Backend & AI:**
    -   **Framework:** Streamlit
    -   **ML Library:** PyTorch (for the ResNet-18 model)
    -   **Generative AI:** Google Generative AI (Gemini 1.5 Flash)
    -   **Text-to-Speech:** gTTS
    -   **Translation:** Deep-Translator
-   **Deployment:**
    -   **Landing Page:** Vercel
    -   **Analyzer Tool:** Streamlit Cloud

---

## 📂 Project Structure

```
.
├── app.py                  # The Streamlit analyzer tool
├── index.html              # The main landing page
├── retinopathy_model.pth   # The trained model (must be added)
├── api_key.py              # Your Google API key (must be created)
├── requirements.txt        # Project dependencies
└── README.md
```

---

## 🚀 Setup and Installation

To run this project locally, follow these steps:

**1. Prerequisites:**
-   Python 3.8 or higher
-   `pip` package manager

**2. Clone the Repository:**
```bash
git clone https://github.com/Venky8249/Eye_diseases.git
cd Eye_diseases
```

**3. Set Up a Virtual Environment (Recommended):**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**4. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**5. Add Required Files:**
You need to add two crucial files to the root directory:

-   **Trained Model:**
    -   Download or place your trained PyTorch model file and name it `retinopathy_model.pth`.

-   **Google API Key:**
    -   Create a new file named `api_key.py`.
    -   Inside this file, add your Google API key like this:
      ```python
      api_key = "YOUR_GOOGLE_API_KEY_HERE"
      ```

**6. Run the Application:**
-   **To view the landing page:** Open the `index.html` file directly in your web browser.
-   **To run the analyzer tool:** Execute the following command in your terminal:
  ```bash
  streamlit run app.py
  ```

---

## ⚠️ Disclaimer

This tool is for **educational and demonstrative purposes only**. It is **not a substitute for professional medical advice**, diagnosis, or treatment. Always seek the advice of your physician or other qualified health providers with any questions you may have regarding a medical condition.
