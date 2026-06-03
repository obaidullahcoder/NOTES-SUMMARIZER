import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader
import easyocr
from PIL import Image
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client safely
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key not found. Please set GROQ_API_KEY in your environment or Streamlit Secrets.")
    st.stop() 
else:
    client = Groq(api_key=groq_api_key)

# Page Configuration
st.set_page_config(page_title="🤖 Advanced Note Summarizer", layout="wide")

# 🎨 CUSTOM CSS: Forces orange background, fixes text colors, and hides Streamlit/GitHub menus
st.markdown(
    """
    <style>
    /* 1. Make the whole app background orange */
    .stApp {
        background-color: #FF9224;
    }
    
    /* 2. Force text colors to black for readability */
    h1, h2, h3, p, span, label {
        color: #000000 !important;
    }
    .stTextArea textarea {
        background-color: #F0F2F6 !important;
        color: #000000 !important;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #000000 !important;
    }
    
    /* 3. HIDE STREAMLIT HEADER, DEPLOY BUTTON, AND FOOTER */
    header {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- TOP HEADER SECTION (Title on Left, Profile Pic & Tagline on Right) ---
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.title("Advanced AI Notes Summarizer")
    st.markdown("**Upload a PDF document, an image/screenshot, or paste text directly to generate an intelligent summary.**")

with header_col2:
    # Try to load your profile photo if it exists in the folder
    if os.path.exists("profile.png"):
        img = Image.open("profile.png")
        st.image(img, width=100)
    else:
        st.caption("(Place 'profile.png' here)")
        
    st.markdown("<p style='font-size: 11px; font-weight: bold; margin-top: -5px;'>made by obaid for nalaik students</p>", unsafe_allow_html=True)

st.markdown("---")

# --- MAIN LAYOUT COLUMNS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Input")
    
    # 1. File Uploader widget
    uploaded_file = st.file_uploader("Upload your notes (PDF, PNG, JPG, JPEG)", type=["pdf", "png", "jpg", "jpeg"])

    extracted_text = ""

    # 2. Process the file if a user uploads one
    if uploaded_file is not None:
        # --- IF THE FILE IS A PDF ---
        if uploaded_file.type == "application/pdf":
            with st.spinner("Extracting text from PDF..."):
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
            st.success("PDF Text Extracted Successfully!")

        # --- IF THE FILE IS AN IMAGE (PNG/JPG) ---
        elif uploaded_file.type in ["image/png", "image/jpeg"]:
            with st.spinner("Reading text from image (OCR)..."):
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                
                # Initialize the OCR reader (english)
                reader = easyocr.Reader(['en'])
                results = reader.readtext(image_np)
                
                # Combine found text chunks
                extracted_text = " ".join([res[1] for res in results])
            st.success("Image Text Extracted Successfully!")

    # 3. Text area displays extracted text OR allows manual typing
    if extracted_text:
        note_input = st.text_area("Review/Edit your extracted notes below:", value=extracted_text, height=300)
    else:
        note_input = st.text_area("Or type/paste your notes manually here:", height=300)

    submit_button = st.button("Summarize Note", type="primary")

with col2:
    st.subheader("AI Summary")
    
    # 4. Running the Groq LLM completion block when button clicked
    if submit_button:
        if not note_input.strip():
            st.warning("Please upload a file or enter some text first.")
        else:
            with st.spinner("Analyzing your notes..."):
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system", 
                                "content": "You are an expert academic assistant. Summarize the user's notes cleanly using simple core bullet points, key concepts, and easy formulas if applicable."
                            },
                            {"role": "user", "content": f"Please summarize these notes:\n\n{note_input}"}
                        ],
                        # Using the stable, active free-tier model
                        model="llama-3.1-8b-instant",
                        temperature=0.3,
                    )
                    summary = chat_completion.choices[0].message.content
                    st.success("Generation Complete!")
                    st.markdown(summary)
                except Exception as e:
                    st.error(f"An error occurred while connecting to Groq: {e}")