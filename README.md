# 🔐 Secure PDF RAG Chatbot

A secure local RAG (Retrieval Augmented Generation) application that allows employees to ask questions about company PDF documents using Ollama and LangChain. The application provides role-based access, allowing admins to upload and manage company documents while employees can securely query the available PDFs. All document processing, retrieval, and LLM inference are performed locally using Ollama and ChromaDB.

## 📌 Project Overview

Companies often store important information in internal documents such as:
- HR policies
- Leave policies
- Work-from-home policies
- IT security guidelines
- Employee benefits
- Company procedures and SOPs

Searching through multiple documents manually can be time-consuming.

This project provides a chatbot where:

1. An **Admin** can upload company PDF documents.
2. An **Employee** can log in to the chatbot.
3. The employee asks questions in natural language.
4. The system searches the relevant company documents.
5. Relevant document sections are retrieved using RAG.
6. **Ollama runs the LLM locally** to generate the answer.
7. The application displays the answer along with the relevant document sources.

---

## ✨ Features

- 🔒 **Local AI Processing** – Documents and LLM processing remain on the local machine.
- 👤 **Role-Based Login** – Separate Admin and Employee access.
- 📄 **Multi-PDF Support** – Admin can upload multiple company documents.
- 🧠 **RAG-Based Question Answering** – Answers are generated using relevant document content.
- 🤖 **Ollama** – Runs the LLM locally without cloud LLM APIs.
- 🗂️ **ChromaDB** – Provides local vector storage and document retrieval.
- 📚 **Source References** – Shows the documents used to answer questions.
- 🔍 **Document-Based Answers** – The chatbot is designed to answer using the available company documents.
- ⚡ **Direct Document Retrieval** – Uses focused document retrieval to improve response time.
- 🖥️ **Streamlit Interface** – Simple web-based interface for Admin and Employee users.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │      Admin       │
                    │      Login       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Upload Company   │
                    │      PDFs        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   PDF Processing │
                    │   & Text Split   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │ Local Retrieval  │
                    └────────┬─────────┘
                             │
                             │
┌──────────────────┐         ▼
│    Employee      │   ┌──────────────────┐
│      Login       │──▶│ Relevant Document│
└──────────────────┘   │     Retrieval     │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Ollama Local LLM │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Answer + Sources │
                       └──────────────────┘


## 🔄 How the RAG Pipeline Works

Company PDF
     ↓
PDF Text Extraction
     ↓
Text Chunking
     ↓
Local Embeddings
     ↓
ChromaDB
     ↓
User Question
     ↓
Relevant Document Retrieval
     ↓
Relevant Context
     ↓
Ollama Local LLM
     ↓
Final Answer
     ↓
Source References


## 👥 User Roles

1) 👨‍💼 Admin
The Admin can:
Log in using Admin credentials.
Upload company PDF documents.
Add multiple documents to the document collection.
Manage the documents available to the chatbot.

2) 👩‍💻 Employee
The Employee can:
Log in using Employee credentials.
Access the chatbot.
Ask questions about company documents.
Receive answers based on the available PDFs.
View the document sources used for the answer.

##🔐 Local Privacy

One of the main goals of this project is to demonstrate a local/private RAG architecture.

The application uses Ollama to run the LLM locally.

Therefore, the application does not require sending company PDF content to a cloud LLM API for normal question answering.

For a real company implementation, the same architecture could be hosted on private company-controlled infrastructure, allowing employees to access the application through an internal web interface.


## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web interface |
| LangChain | RAG pipeline and LLM integration |
| Ollama | Local LLM runtime |
| ChromaDB | Local vector database |
| PyMuPDF | PDF text extraction |
| RecursiveCharacterTextSplitter | Document chunking |
| Git & GitHub | Version control |


## 📁 Project Structure

PDF-RAG-Chatbot/
│
├── src/
│   ├── app/
│   │   └── main.py
│   │
│   └── core/
│       └── document.py
│
├── data/
│   ├── documents/
│   │   └── company PDFs
│   │
│   └── chroma_db/
│       └── local vector database
│
├── run.py
├── requirements.txt
├── .gitignore
└── README.md

The ragenv virtual environment is used locally but is excluded from GitHub.


## ⚙️ Installation

1. Clone the repository
git clone https://github.com/aditinikalje123/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot

2. Create a Python virtual environment
Python 3.10 is recommended for this project.

python -m venv ragenv
Activate it on Windows:
.\ragenv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

## 🤖 Install Ollama

Install Ollama on your system and download the required models.

Example:
ollama pull llama3.2:3b
For document embeddings:
ollama pull nomic-embed-text
Make sure Ollama is running before starting the application.


## ▶️ Run the Application

From the project directory:
streamlit run src/app/main.py

Or use:

python run.py
The Streamlit application will open in your browser.


## 🔑 Demo Login

For demonstration purposes, the application contains separate Admin and Employee credentials.

1) Admin
Username: admin
Password: admin123

2) Employee
Username: employee
Password: employee123

These credentials are intended only for local demonstration. A production implementation should use secure password hashing, a proper authentication system, and organization-managed user accounts.


##💬 Example Questions

After logging in as an employee, you can ask questions such as:

How many days of annual leave are available?

How much advance notice is required for annual leave?

How many work-from-home days can an employee take per month?

What are the company password requirements?

What should I do if I receive a suspicious email?

What should I do if my company laptop is lost?

How can I submit a reimbursement request?

How many sick leave days are available?

The chatbot retrieves relevant information from the uploaded company PDFs and generates an answer using the local LLM.

## 📚 Example Company Documents

The project can work with documents such as:

HR Leave Policy
Employee Work From Home Policy
IT Security Guidelines
Employee Benefits Policy
Company SOPs
Internal HR guidelines
IT policies

The PDFs are examples for demonstrating the RAG workflow.


## 🚀 Future Improvements

Possible improvements for a production-level implementation include:

Secure password hashing
Database-backed user authentication
Multiple employee accounts
Fine-grained document permissions
Department-based document access
Document deletion and management
Improved conversation history
Faster retrieval and response generation
Audit logs for document access
Internal/private server deployment
HTTPS and enterprise authentication

## 🎯 Real-World Use Case

This project can be adapted for organizations that need employees to query internal documentation without manually searching through large collections of PDFs.

For example:

Employee
   ↓
Secure Login
   ↓
Internal Company Chatbot
   ↓
Authorized Company Documents
   ↓
RAG Retrieval
   ↓
Private / Self-Hosted LLM
   ↓
Answer + Source

The LLM and document-processing components can be hosted on infrastructure controlled by the organization rather than requiring each employee to run the model locally.

## 📌 Project Highlights

Built a local PDF-based RAG chatbot.
Implemented Admin and Employee role separation.
Added multi-document support.
Integrated Ollama for local LLM inference.
Added local document retrieval using ChromaDB.
Used PyMuPDF for PDF text extraction.
Added source references to chatbot responses.
Focused on privacy for confidential company documents.

## 📄 License

This project is intended for educational and portfolio purposes.
