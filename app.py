import gradio as gr
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from groq import Groq
import faiss
import os
import time
# Adjust the paths to your uploaded PDF files
pdf_paths = [
    "/data/financepdf.pdf",
    "/dataFINANCIAL REGULATIONS.pdf",
    "dataINTERNATIONAL FINANCIAL REPORTING STANDARDS (UPDATED).docx.pdf"

    # Add more paths here as needed
]

# Load PDF files and extract text
document_texts = []
for pdf_path in pdf_paths:
    try:
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() for page in reader.pages if page.extract_text() is not None]
        document_text = "\n".join(pages)
        document_texts.append(document_text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")

# Split the documents into smaller sections for vectorization
sections = []
for document_text in document_texts:
    sections.extend(document_text.split("\n\n"))

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)

# Initialize FAISS vector store
vector_store = FAISS.from_texts(
    texts=sections,
    embedding=embeddings_model
)

prompt = """
You are a chatbot designed to answer questions about the Financials:
{retrieved_data}

User Query: {user_query}
"""

# Initialize Groq API client
client = Groq(api_key="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")

def process_query(user_query):
    try:
        if not user_query.strip():
            return "Please enter a valid question."

        print(f"User Query: {user_query}")

        # Retrieve relevant documents from FAISS vector store
        retrieved_docs = vector_store.similarity_search(user_query, k=3)

        if not retrieved_docs:
            return "No relevant information found in the data I have."

        retrieved_data = "\n".join([doc.page_content for doc in retrieved_docs])

        # Limit the length of the data if it exceeds token limits
        if len(retrieved_data.split()) > 1000:  # Adjust the threshold accordingly
            retrieved_data = " ".join(retrieved_data.split()[:1000])  # Truncate data

        print(f"Retrieved Data: {retrieved_data[:1000]}")  # Show first 1000 characters of the retrieved data

        # Format the prompt for the model
        formatted_prompt = prompt.format(retrieved_data=retrieved_data, user_query=user_query)

        print(f"Formatted Prompt: {formatted_prompt[:1000]}")  # Print first 1000 chars

        # Make the API call to Groq model
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.5,
            max_tokens=500,
            top_p=0.85,
            stream=False,
            stop=None,
        )

        if not completion.choices:
            return "No response from the model."

        response = completion.choices[0].message.content
        print(f"Response: {response}")
        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"

# Gradio app

with gr.Blocks("Light") as app:
    gr.Markdown("""<h1>Financial Chatbot</h1>
    Use this chatbot to ask questions related to finance.
    """)

    query_input = gr.Textbox(
        label="Ask a Question",
        placeholder="Enter your question about finance...",
        lines=3
    )
    query_output = gr.Textbox(
        label="Response",
        placeholder="The chatbot's response will appear here...",
        lines=10,
        interactive=False
    )
    ask_button = gr.Button("Ask")
    ask_button.click(
        fn=process_query,
        inputs=query_input,
        outputs=query_output
    )

    gr.Markdown("""<h3>Disclaimer</h3>
    The chatbot provides Finance related Responses. Please verify critical details independently.
    """)

# Launch Gradio interface
app.launch()
