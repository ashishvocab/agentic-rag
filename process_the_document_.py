import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyDdlpRoOhsmj8KPWiX5krNFOfTCwSMj8LY"
if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is missing. Please set it in the environment variables."
    )

genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)


def load_and_chunk_pdf(pdf_path):
    """Load PDF and split into text chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []


def get_embeddings(texts):
    """Generate embeddings for text chunks."""
    embeddings = []
    for doc in texts:
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=doc.page_content,
                task_type="retrieval_document",
            )
            if response and "embedding" in response:
                embeddings.append(response["embedding"])
        except Exception as e:
            print(f"Error generating embedding for text chunk: {e}")
            continue
    return embeddings


def initialize_faiss(pdf_file_path):
    """Initialize FAISS index with document embeddings."""
    text_chunks = load_and_chunk_pdf(pdf_file_path)

    if not text_chunks:
        print("Warning: No text chunks found. Check the PDF file.")
        return None, None

    embeddings = get_embeddings(text_chunks)
    if not embeddings:
        print("Error: No embeddings generated. Check the document content.")
        return None, None

    embeddings_np = np.array(embeddings, dtype=np.float32)
    faiss_index = faiss.IndexFlatL2(embeddings_np.shape[1])
    faiss_index.add(embeddings_np)
    print(f"FAISS index initialized successfully for {pdf_file_path}!")
    return faiss_index, text_chunks


def retrieve_relevant_chunks(query, faiss_index, text_chunks, top_k=5):
    """Retrieve the most relevant text chunks for a given query."""
    try:
        query_embedding = genai.embed_content(
            model="models/embedding-001", content=query, task_type="retrieval_query"
        )["embedding"]

        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        distances, indices = faiss_index.search(query_embedding_np, top_k)

        relevant_chunks = [
            text_chunks[i].page_content for i in indices[0] if i < len(text_chunks)
        ]
        return relevant_chunks
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []


def generate_response(
    query, guidelines_index, guidelines_chunks, llm, conversation_history, top_k=5
):
    """Generate a response to the user query using the guidelines and example PDFs."""
    relevant_guidelines = retrieve_relevant_chunks(
        query, guidelines_index, guidelines_chunks, top_k
    )

    if not relevant_guidelines:
        return "Sorry, I couldn't find any relevant information in the guidelines."

    guidelines_context = "\n\n".join(relevant_guidelines)

    is_first_interaction = len(conversation_history) <= 1

    history = "\n".join(
        [
            f"User: {entry['user']}\nAgent: {entry['agent']}"
            for entry in conversation_history
        ]
    )

    try:
        input_text = f"""
            You are a customer care agent for CRED-Help responding directly to a customer.
            The customer has come to you specifically for an issue related to CRED and not just a normal query. 
            Use these guidelines to inform your response:
            {guidelines_context}

            User's query: {query}

            Previous conversation history:
            {history}

            {"" if is_first_interaction else "IMPORTANT: DO NOT use greetings like Good Morning/Afternoon/Evening again as you've already greeted the customer."}

            IMPORTANT INSTRUCTIONS:
            1. Respond DIRECTLY to the user as if in a conversation
            2. DO NOT include meta-commentary like "Here's my response" or "Following the guidelines" 
            3. DO NOT narrate your actions or thought process
            4. Keep responses helpful and concise
            5. Follow the guidelines provided
            6. Do not repeat information from previous messages
            """

        response = llm.invoke(input=input_text)

        agent_response = response.content

        conversation_history.append({"user": query, "agent": agent_response})
        return agent_response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, there was an error generating a response."


if __name__ == "__main__":
    guidelines_pdf_path = "guidelines.pdf"
    guidelines_index, guidelines_chunks = initialize_faiss(guidelines_pdf_path)

    if guidelines_index:
        conversation_history = []
        print(
            "Customer Care Agent is ready to assist you. Type 'exit' to end the conversation."
        )

        first_agent_response = "Good morning, thank you for contacting CRED-Help. How can I assist you today?"
        print(f"Agent: {first_agent_response}")
        conversation_history.append({"user": "Hello", "agent": first_agent_response})

        while True:
            user_query = input("You: ")
            if user_query.lower() == "exit":
                print("Agent: Thank you for reaching out to CRED. Have a great day!")
                break

            response = generate_response(
                query=user_query,
                guidelines_index=guidelines_index,
                guidelines_chunks=guidelines_chunks,
                llm=llm,
                conversation_history=conversation_history,
            )
            print(f"Agent: {response}")
