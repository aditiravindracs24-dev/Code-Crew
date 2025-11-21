import streamlit as st
import fitz              
from PIL import Image     
import pytesseract        
import docx               

st.title("SpecCraft – SRS Generator")
st.write("Upload a requirements document to extract raw text.")

uploaded_file = st.file_uploader(
    "Upload PDF, DOCX, Image (PNG/JPG) or Text file",
    type=["pdf", "docx", "png", "jpg", "jpeg", "txt"]
)

def extract_pdf(file):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

def extract_docx(file):
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])

def extract_image(file):
    img = Image.open(file)
    return pytesseract.image_to_string(img)

text = ""

if uploaded_file:
    st.subheader("Extracted Text")

    if uploaded_file.type == "application/pdf":
        text = extract_pdf(uploaded_file)

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_docx(uploaded_file)

    elif uploaded_file.type in ["image/png", "image/jpeg"]:
        text = extract_image(uploaded_file)

    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    else:
        st.error("Unsupported file format.")

    st.text_area("Raw Extracted Text", text, height=350)

