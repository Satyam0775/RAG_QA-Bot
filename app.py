import gradio as gr
import PyPDF2
import cohere
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import io

# Initialize Pinecone and connect to the index
pc = Pinecone(api_key="0f78bc1b-81f7-4a15-9af3-0fbcf0acdb4e")
index = pc.Index("quickstart")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Cohere with your API key
co = cohere.Client("CxIrucBVA8NNJJOBUnxwRWq488MVydBku1DlqP1u")

def extract_text_from_pdf(pdf_file):
    """Extracts text from the uploaded PDF, with error handling."""
    try:
        if pdf_file is None:
            return "No file uploaded."

        # Read the PDF content as bytes using a file-like object
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() or ""

        if not text.strip():
            return "The uploaded PDF is empty or has no readable content."
        return text

    except PyPDF2.errors.PdfReadError:
        return "The uploaded PDF is encrypted or unreadable."
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def store_pdf_embeddings(pdf_text):
    """Generate and store embeddings for the uploaded PDF content."""
    segments = [pdf_text[i:i + 512] for i in range(0, len(pdf_text), 512)]
    embeddings = model.encode(segments)
    vectors = [(f"seg-{i}", embed.tolist()) for i, embed in enumerate(embeddings)]
    index.upsert(vectors=vectors)
    return "PDF uploaded and stored successfully!"

def ask_question(query):
    """Handle user questions and generate answers based on the PDF content."""
    query_embedding = model.encode(query).tolist()

    # Retrieve the most relevant segment from Pinecone
    result = index.query(top_k=1, vector=query_embedding)
    retrieved_seg_id = result['matches'][0]['id']
    segment_text = f"Segment: {retrieved_seg_id}"

    # Generate the answer using the retrieved segment as context
    prompt = f"{segment_text}\nQuestion: {query}\nAnswer:"
    response = co.generate(
        model="command-xlarge-nightly",
        prompt=prompt,
        max_tokens=50
    )

    # Return both the segment and the answer
    answer = response.generations[0].text.strip()
    return segment_text, answer

# Gradio Interface Setup
with gr.Blocks() as demo:
    gr.Markdown("# Interactive QA Bot with PDF Support")

    # PDF Upload Section
    pdf_input = gr.File(label="Upload PDF", type="file", file_types=[".pdf"])
    upload_status = gr.Textbox(label="Upload Status", interactive=False)
    upload_button = gr.Button("Upload and Store")

    # Handle PDF Upload
    upload_button.click(
        lambda pdf: store_pdf_embeddings(extract_text_from_pdf(pdf))
        if pdf is not None else "Please upload a valid PDF.",
        inputs=pdf_input, outputs=upload_status
    )

    # Question and Answer Section
    query_input = gr.Textbox(label="Enter your question")
    segment_output = gr.Textbox(label="Retrieved Segment", interactive=False)
    answer_output = gr.Textbox(label="Answer", interactive=False)
    query_button = gr.Button("Ask")

    # Handle User Questions
    query_button.click(
        ask_question, inputs=query_input, outputs=[segment_output, answer_output]
    )

demo.launch(share=True)  # Set share=True if you want a public link