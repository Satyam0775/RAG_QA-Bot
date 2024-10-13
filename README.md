# RAG-Based QA Bot with PDF Support

This project is a **Question Answering (QA) bot** built using **Gradio** for the frontend interface, **Pinecone** for storing document embeddings, and **Cohere** for generating answers. The bot allows users to **upload PDF documents** and **ask questions** based on the document content. It returns the **relevant segment** from the document along with the **answer**.

---

## **Features**

- **Upload PDF documents** and generate embeddings for efficient search.
- **Ask questions** based on the uploaded content.
- **Retrieve relevant segments** and display them with answers.
- **Handles multiple queries** efficiently.
- **Containerized with Docker** for easy deployment.
- **Public Gradio link for live interaction**.

---

## **URLs**

- **Local URL**: [http://127.0.0.1:7860](http://127.0.0.1:7860)  
- **Public Gradio Link**: [https://e18ce42117606df9dd.gradio.live](https://e18ce42117606df9dd.gradio.live)

---

## **Technologies Used**

- **Python**: Programming language used to build the application.
- **Gradio**: For building the user interface.
- **Pinecone**: Vector database for storing and retrieving document embeddings.
- **Cohere**: Generative AI platform to generate answers.
- **PyPDF2**: For extracting text from uploaded PDFs.
- **Docker**: For containerizing the application for easy deployment.

RAG_QA-Bot.zip/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
├── colab_notebook.ipynb
└── sample.pdf Image
<img width="941" alt="Screenshot 2024-10-13 143859" src="https://github.com/user-attachments/assets/f9767af3-e412-4a3c-910a-f1aa4673d56f">






