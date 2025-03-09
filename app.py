import streamlit as st
import requests
import os
import time
import PyPDF2
from retrieval import upsert_screenplay_vectors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Backend URL
BACKEND_URL = "http://127.0.0.1:5000/generate"

# Streamlit UI
st.set_page_config(page_title="Screenplay AI", layout="wide")

st.sidebar.title("📜 Upload Screenplay File")
uploaded_file = st.sidebar.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

st.sidebar.title("🎭 Choose Genre")
genre = st.sidebar.selectbox(
    "Select a genre:",
    ["Sci-Fi", "Drama", "Comedy", "Thriller", "Horror", "Fantasy", "Action", "Romance"],
    index=0
)

st.title("🎬 RICKY - AI Screenplay Assistant")
st.write("Generate high-quality screenplays with AI.")

# Function to extract text from uploaded files
def extract_text(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text().replace("\n", " ") for page in reader.pages if page.extract_text()])
    else:
        text = file.read().decode("utf-8")
    return text

# Function to generate embeddings and upsert to Pinecone
def process_and_store(file):
    text = extract_text(file)
    if not text:
        st.error("No text found in file.")
        return None
    
    upsert_screenplay_vectors(str(hash(text)), text, genre)
    return text

if uploaded_file:
    screenplay_text = process_and_store(uploaded_file)
    if screenplay_text:
        st.sidebar.success("File uploaded and stored successfully! 🎉")

# Chat Interface
st.sidebar.button("Clear History", on_click=lambda: st.session_state.update({"messages": []}))

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your screenplay idea...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    response = requests.post(
        BACKEND_URL,
        json={"text": user_input, "genre": genre}
    )
    ai_response = response.json().get("screenplay", "Error generating screenplay.")

    st.session_state["messages"].append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)

    # Save screenplay as PDF
    def save_as_pdf(text, filename="generated_screenplay.pdf"):
        file_path = os.path.join("data/screenplays", filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        c = canvas.Canvas(file_path, pagesize=letter)
        y_position = 750
        for line in text.split(" "):
            c.drawString(100, y_position, line)
            y_position -= 15
            if y_position < 50:
                c.showPage()
                y_position = 750
        c.save()
        return file_path

    pdf_path = save_as_pdf(ai_response)
    with open(pdf_path, "rb") as pdf_file:
        st.sidebar.download_button("Download PDF", data=pdf_file, file_name="screenplay.pdf", mime="application/pdf")
