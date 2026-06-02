import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq client safely
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API Key not found. Please set GROQ_API_KEY in your environment or Streamlit Secrets.")
    st.stop()  # Instantly halts execution cleanly so it doesn't try to use a missing 'client'
else:
    client = Groq(api_key=groq_api_key)

# Page Configuration
st.set_page_config(
    page_title="⚡ Groq Note Summarizer",
    page_icon="📝",
    layout="wide"
)

# App Header
st.title("⚡ Groq Note Summarizer")
st.caption("Paste your messy notes and get a clean, AI-generated summary instantly powered by Llama 3.1 on Groq.")
st.markdown("---")

# Create two columns for the side-by-side workspace
col1, col2 = st.columns(2)

with col1:
    st.subheader("Your Notes")
    # Text area for user input
    note_input = st.text_area(
        label="Enter or paste your notes here:",
        placeholder="Type or paste your lecture notes, meeting transcripts, or brainstorms here...",
        height=400,
        label_visibility="collapsed"
    )
    
    # Trigger button
    submit_button = st.button("Summarize Note", type="primary", use_container_width=True)

with col2:
    st.subheader("AI Summary")
    
    # Logic when the user clicks the button
    if submit_button:
        if not note_input.strip():
            st.warning("Please enter some text to summarize!")
        else:
            # Spinner placeholder while waiting for Groq
            with st.spinner("Analyzing your notes and generating summary..."):
                try:
                    # Request to Groq API
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are an expert executive assistant. Summarize the user's notes cleanly. Use concise bullet points for key takeaways and a 1-2 sentence overview at the top."
                            },
                            {
                                "role": "user",
                                "content": f"Please summarize these notes:\n\n{note_input}"
                            }
                        ],
                        model="llama-3.1-8b-instant", # Updated to the active, supported model
                        temperature=0.3,              # Low temperature keeps summaries focused and factual
                    )
                    
                    # Extract the text answer
                    summary = chat_completion.choices[0].message.content
                    
                    # Render the output neatly inside a markdown box
                    st.success("Generation complete!")
                    st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"An error occurred while connecting to Groq: {e}")
    else:
        # Placeholder text before the user hits submit
        st.info("Your summary will appear here once you hit 'Summarize Note'.")