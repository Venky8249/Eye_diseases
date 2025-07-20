import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import google.generativeai as genai
from api_key import api_key
from gtts import gTTS
from deep_translator import GoogleTranslator
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Eye Image Analyzer",
    page_icon="🩺",
    layout="centered"
)

# --- Custom CSS for Styling ---
page_style = """
<style>
/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: #0c1445;
    background-image: linear-gradient(180deg, #0c1445 0%, #03045e 100%);
}

/* Main content area */
[data-testid="stVerticalBlock"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 2rem;
    backdrop-filter: blur(10px);
}

/* Style for the File Uploader container */
[data-testid="stFileUploader"] {
    background-image: linear-gradient(180deg, #0c1445 0%, #03045e 100%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 1rem;
}

/* --- NEW: Style for the INNER drop zone of the File Uploader --- */
[data-testid="stFileUploader"] section {
    background-color: transparent; /* Make the inner box transparent */
    border: 2px dashed rgba(255, 255, 255, 0.4); /* Add a dashed border */
    border-radius: 8px;
}
[data-testid="stFileUploader"] section small {
    color: rgba(255, 255, 255, 0.8) !important; /* Make the "Limit 200MB" text visible */
}
[data-testid="stFileUploader"] section svg {
    color: rgba(255, 255, 255, 0.8); /* Make the cloud icon visible */
}

/* Buttons */
[data-testid="stButton"] button {
    background-image: linear-gradient(45deg, #0077b6 0%, #00b4d8 100%);
    border: none;
    color: white;
    border-radius: 8px;
    transition: all 0.3s ease-in-out;
}
[data-testid="stButton"] button:hover {
    box-shadow: 0 0 15px #90e0ef;
    transform: scale(1.05);
}

/* Expander */
[data-testid="stExpander"] {
    background-color: rgba(0, 180, 216, 0.1);
    border-radius: 10px;
}

/* Headers and text */
h1, h2, h3, h4, h5, h6, p, div, label {
    color: #ffffff !important;
}

</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# --- Model Loading and Configuration ---
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")

@st.cache_resource
def load_classifier_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('retinopathy_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

classifier_model = load_classifier_model()

classifier_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
id2label = {0: 'Normal', 1: 'Cataract', 2: 'Diabetic Retinopathy', 3: 'Glaucoma'}

def classify_image(image, model, transform_fn, id2label_map):
    image_tensor = transform_fn(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        pred_index = torch.argmax(output, axis=1).item()
        return id2label_map[pred_index]

# --- Streamlit App UI ---
st.title("👁️ AI Eye Image Analyzer")
st.write("Upload a medical image to get a classification and AI-powered advice.")
st.warning("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.", icon="⚠️")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image first
    st.markdown("###### Uploaded Image")
    st.image(image, width=300)
    st.divider()

    # Display the analysis section below the image
    st.subheader("Analysis & Precautions")
    
    with st.expander("Advanced Audio Options"):
        languages = {"English": "en", "Hindi": "hi", "Bengali": "bn", "Korean": "ko", "Chinese": "zh-cn", "Japanese": "ja"}
        accents = {"Default": "com", "India": "co.in", "United Kingdom": "co.uk", "United States": "com"}
        out_lang_name = st.selectbox("Audio Language", list(languages.keys()))
        english_accent_name = st.selectbox("English Accent", list(accents.keys()))
    
    if st.button("Analyze Image"):
        with st.spinner('Analyzing... This may take a moment.'):
            predicted_class = classify_image(image, classifier_model, classifier_transform, id2label)
            st.success(f"**Predicted Disease: {predicted_class}**")

            prompt = f"""The user's retinal scan has been classified as '{predicted_class}'. Provide a helpful explanation of this condition, list detailed precautions, and recommend the next steps the user should take. Keep the tone empathetic and clear. Structure the response with clear headings. End with the mandatory disclaimer: 'This is AI-generated advice. Consult with a Doctor before making any decisions.'"""
            ai_response = gemini_model.generate_content(prompt).text.strip()
            
            output_language_code = languages[out_lang_name]
            tld = accents[english_accent_name]
            translated_text = GoogleTranslator(source='en', target=output_language_code).translate(ai_response)
            tts = gTTS(translated_text, lang=output_language_code, tld=tld, slow=False)
            os.makedirs("temp", exist_ok=True)
            file_path = "temp/analysis_audio.mp3"
            tts.save(file_path)

            st.markdown("#### AI Recommendations & Precautions:")
            st.write(ai_response)
            st.audio(file_path)