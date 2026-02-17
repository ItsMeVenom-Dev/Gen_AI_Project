# 🚀 RAG Application (Gemini)

## ✨ Overview

**RAG Application (Gemini)** is an interactive Streamlit-based web app that allows users to upload PDFs and ask questions directly from the document.

It uses **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini**, **LangChain**, and **FAISS** to generate accurate, context-aware answers strictly from the uploaded document.

💡 If the answer is not present in the PDF, the model clearly responds:

> "Answer not available in the provided PDF."

---

## 🎯 Features

- 📂 Upload any PDF document
- ✂️ Automatic text chunking
- 🧠 HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- 🔎 FAISS Vector Database
- 🤖 Gemini 2.5 Flash LLM
- 📚 Context-based answering only
- ❌ Proper fallback response when answer not found

---

## 🧰 Technologies Used

- **Python**
- **Streamlit**
- **LangChain**
- **FAISS**
- **HuggingFace Sentence Transformers**
- **Google Gemini API**
- **PyPDF2**

---

# ⚙️ Setup Instructions

Follow the steps below to run the project locally.

---

## 1 Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```
## 2 Install Requirements

```bash
pip install -r requirements.txt
```

## 3 Add Gemini API Key

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

## 4 Run the Application

```bash
streamlit run app.py
```


## 🧠 How It Works (RAG Flow)

Here’s what happens internally:

1️⃣ User uploads a PDF
2️⃣ Text is extracted using PyPDF2
3️⃣ Text is split into chunks
4️⃣ Embeddings are generated using HuggingFace
5️⃣ FAISS stores vector embeddings
6️⃣ User asks a question
7️⃣ Relevant chunks are retrieved
8️⃣ Gemini generates an answer using ONLY the retrieved context





Author
Lucky
