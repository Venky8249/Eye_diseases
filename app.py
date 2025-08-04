import streamlit as st
import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import google.generativeai as genai
from api_key import api_key  # Make sure you have this file with your API key
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
/* --- General App & Text Styling --- */
[data-testid="stAppViewContainer"] {
    background-color: #0c1445;
    background-image: linear-gradient(180deg, #0c1445 0%, #03045e 100%);
}

/* --- NEW: Make Header Transparent --- */
[data-testid="stHeader"] {
    background-color: transparent;
}

[data-testid="stVerticalBlock"] {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 15px;
    padding: 2rem;
    backdrop-filter: blur(10px);
}
h1, h2, h3, h4, h5, h6, p, div, label, li {
    color: #ffffff !important;
}

/* --- Sidebar Styling --- */
[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #0c1445 0%, #03045e 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] .stMarkdown {
     color: #ffffff !important;
}

/* --- File Uploader Styling --- */
[data-testid="stFileUploader"] {
    background-image: linear-gradient(180deg, #0c1445 0%, #03045e 100%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    padding: 1rem;
}
[data-testid="stFileUploader"] section {
    background-color: transparent;
    border: 2px dashed rgba(255, 255, 255, 0.4);
    border-radius: 8px;
}
[data-testid="stFileUploader"] section small,
[data-testid="stFileUploader"] section svg {
    color: rgba(255, 255, 255, 0.8) !important;
}

/* --- Button Styling --- */
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

/* --- Expander Styling --- */
[data-testid="stExpander"] {
    background-color: rgba(0, 180, 216, 0.1);
    border-radius: 10px;
    border: 1px solid rgba(0, 180, 216, 0.2);
}

/* --- Selectbox Styling (for visibility) --- */
[data-testid="stSelectbox"] > div {
    background-color: rgba(0, 119, 182, 0.3);
    border: 1px solid #00b4d8;
    border-radius: 8px;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: transparent !important;
}
[data-testid="stSelectbox"] svg {
    color: #ffffff !important;
}
div[data-baseweb="popover"] ul {
    background-color: #03045e;
    border: 1px solid #00b4d8;
}
div[data-baseweb="popover"] ul li:hover {
    background-color: #0077b6;
}

/* --- Modern Animated Home Button --- */
.home-button {
    position: relative;
    z-index: 1;
    display: inline-block;
    padding: 10px 25px;
    margin-bottom: 25px;
    border: 1px solid #90e0ef;
    border-radius: 50px;
    color: #ffffff !important;
    background: transparent;
    text-decoration: none;
    font-weight: 600;
    overflow: hidden;
    transition: all 0.4s ease-in-out;
    width: 100%; /* Make button fill sidebar width */
    text-align: center;
}
.home-button:hover {
    transform: scale(1.05);
    color: #ffffff !important;
    text-decoration: none;
}
.home-button::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: #0077b6;
    z-index: -1;
    transform-origin: left;
    transform: scaleX(0);
    transition: transform 0.4s ease-in-out;
}
.home-button:hover::before {
    transform: scaleX(1);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)


# --- Model Loading and Configuration ---
# Configure the Generative AI model
try:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
except Exception as e:
    st.error(f"Could not configure Google Gemini AI. Please check your API key. Error: {e}")
    st.stop()


# Load the pre-trained classifier model
@st.cache_resource
def load_classifier_model():
    """Loads the pre-trained ResNet model for eye disease classification."""
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    try:
        # Load the model state dictionary. Ensure the .pth file is in the same directory.
        model.load_state_dict(torch.load('retinopathy_model.pth', map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Model file 'retinopathy_model.pth' not found. Please make sure it's in the correct directory.")
        st.stop()
    model.eval()
    return model

classifier_model = load_classifier_model()

# Define the image transformations for the classifier
classifier_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
id2label = {0: 'Normal', 1: 'Cataract', 2: 'Diabetic Retinopathy', 3: 'Glaucoma'}

def classify_image(image, model, transform_fn, id2label_map):
    """Classifies an image using the loaded model."""
    image_tensor = transform_fn(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        pred_index = torch.argmax(output, axis=1).item()
        return id2label_map[pred_index]

# --- Sidebar Controls ---
with st.sidebar:
    st.title("Controls")
    st.markdown('<a href="https://eye-diseases.vercel.app/" target="_self" class="home-button">🏠 Home</a>', unsafe_allow_html=True)
    
    st.markdown("### Audio Options")
    languages = {"English": "en", "Hindi": "hi", "Bengali": "bn", "Korean": "ko", "Chinese": "zh-cn", "Japanese": "ja"}
    accents = {"Default": "com", "India": "co.in", "United Kingdom": "co.uk", "United States": "com"}
    out_lang_name = st.selectbox("Audio Language", list(languages.keys()))
    english_accent_name = st.selectbox("English Accent", list(accents.keys()))
    
# --- Main App UI ---
st.title("👁️ AI Eye Image Analyzer")

st.write("Upload a medical image of an eye to get a classification and AI-powered advice.")
st.warning("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical advice.", icon="⚠️")

uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.markdown("###### Uploaded Image")
    st.image(image, width=300)
    st.divider()

    st.subheader("Analysis & Precautions")
    
    # Initialize session state if not present
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'predicted_class' not in st.session_state:
        st.session_state.predicted_class = ""
    if 'ai_response' not in st.session_state:
        st.session_state.ai_response = ""
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None

    if st.button("Analyze Image"):
        with st.spinner('Analyzing... This may take a moment.'):
            # --- Step 1: Use Gemini to check if the image is a valid eye scan ---
            pre_classification_prompt = "Analyze this image. Is it a medical image of a human retina or eye, suitable for diagnosing conditions like diabetic retinopathy, glaucoma, normal or cataracts? Please answer with only 'Yes' or 'No'."
            
            try:
                # Send the prompt and the image to the Gemini model
                response = gemini_model.generate_content([pre_classification_prompt, image])
                is_eye_image = response.text.strip().lower()

                # --- Step 2: Branch logic based on the check ---
                if 'yes' in is_eye_image:
                    # --- IT IS AN EYE IMAGE: Proceed with original classification ---
                    st.session_state.predicted_class = classify_image(image, classifier_model, classifier_transform, id2label)
                    
                    # Generate detailed medical advice using Gemini
                    advice_prompt = f"""The user's retinal scan has been classified as '{st.session_state.predicted_class}'. Provide a helpful explanation of this condition, list detailed precautions, and recommend the next steps the user should take. Keep the tone empathetic and clear. Structure the response with clear headings. End with the mandatory disclaimer: 'This is AI-generated advice. Consult with a Doctor before making any decisions.'"""
                    st.session_state.ai_response = gemini_model.generate_content(advice_prompt).text.strip()

                else:
                    # --- IT IS NOT AN EYE IMAGE: Describe and instruct ---
                    st.session_state.predicted_class = "Not an Eye Image"
                    
                    # Generate a description and instruction using Gemini
                    not_eye_prompt = """The user has uploaded an image that is not a medical eye scan. 
                    First, briefly describe what is in this image. 
                    Then, in a new paragraph, add a clear and polite message telling the user that for the app to work, they must upload a retinal or fundus image of an eye."""
                    st.session_state.ai_response = gemini_model.generate_content([not_eye_prompt, image]).text.strip()

                # --- Step 3: Generate audio for the AI response (for both cases) ---
                output_language_code = languages[out_lang_name]
                tld = accents[english_accent_name]
                
                translated_text = GoogleTranslator(source='en', target=output_language_code).translate(st.session_state.ai_response)
                
                # Check if translation returned a non-empty string
                if translated_text:
                    tts = gTTS(translated_text, lang=output_language_code, tld=tld, slow=False)
                    os.makedirs("temp", exist_ok=True)
                    file_path = "temp/analysis_audio.mp3"
                    tts.save(file_path)
                    st.session_state.audio_file = file_path
                else:
                     st.session_state.audio_file = None

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.session_state.ai_response = "Sorry, I couldn't process the image. Please try again."
                st.session_state.predicted_class = "Error"
            
            st.session_state.analysis_done = True
            st.rerun()

    # --- Step 4: Display results based on the analysis outcome ---
    if st.session_state.analysis_done:
        if st.session_state.predicted_class == "Not an Eye Image":
            st.warning(f"**Analysis Result: {st.session_state.predicted_class}**")
            st.markdown("#### AI Response:")
            st.write(st.session_state.ai_response)
        
        elif st.session_state.predicted_class == "Error":
            st.error(st.session_state.ai_response)

        else: # This is a successful disease prediction
            st.success(f"**Predicted Disease: {st.session_state.predicted_class}**")
            st.markdown("#### AI Recommendations & Precautions:")
            st.write(st.session_state.ai_response)

        # Play audio if it was generated successfully
        if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
            st.audio(st.session_state.audio_file)