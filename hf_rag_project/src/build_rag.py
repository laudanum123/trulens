import datasets
import time
import torch
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
import os 
import pickle 
import openai 
from dotenv import load_dotenv 

# Define data directory
DATA_DIR = "hf_rag_project/data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.idx")
DOCUMENTS_PATH = os.path.join(DATA_DIR, "documents.pkl")


def load_wikipedia_dataset():
    """Loads the rag-mini-wikipedia dataset from Hugging Face."""
    print("Loading dataset...")
    # Ensure data dir exists
    os.makedirs(DATA_DIR, exist_ok=True)
    # Using the 'text-corpus' configuration which contains the passages
    dataset = datasets.load_dataset("rag-datasets/rag-mini-wikipedia", name='text-corpus', trust_remote_code=True)
    print("Dataset loaded.")
    return dataset


def prepare_documents(dataset):
    """Prepares documents for indexing from the dataset passages."""
    print("Preparing documents...")
    start_time = time.time()
    split_name = list(dataset.keys())[0]
    original_passages = dataset[split_name]

    documents = []
    for record in original_passages:
        # Ensure text is not empty or just whitespace
        text = record['passage'].strip()
        if text:
            documents.append({
                'id': str(record['id']), # Use original ID, ensure string type
                'text': text
            })
        else:
             print(f"Warning: Skipping empty passage with ID {record['id']}")


    # Save documents list
    with open(DOCUMENTS_PATH, 'wb') as f:
        pickle.dump(documents, f)
    print(f"Saved {len(documents)} documents to {DOCUMENTS_PATH}")


    end_time = time.time()
    print(f"Document preparation complete in {end_time - start_time:.2f} seconds.")
    print(f"Prepared {len(documents)} documents.")
    return documents

def generate_embeddings(documents, model_name='all-MiniLM-L6-v2'):
    """Generates embeddings for the prepared documents."""
    print(f"Generating embeddings using '{model_name}'...")
    start_time = time.time()

    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = SentenceTransformer(model_name, device=device)

    # Extract text to be embedded
    texts = [doc['text'] for doc in documents]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True, device=device, convert_to_numpy=True) # Ensure numpy array

    end_time = time.time()
    print(f"Embedding generation complete in {end_time - start_time:.2f} seconds.")
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}.")
    return embeddings

def build_and_save_faiss_index(embeddings):
    """Builds a FAISS index from embeddings and saves it to disk."""
    print("Building FAISS index...")
    start_time = time.time()

    # Dimension of embeddings
    d = embeddings.shape[1]

    # Build the index
    index = faiss.IndexFlatL2(d)  # Using L2 distance
    # Ensure embeddings are float32 numpy arrays, which FAISS expects
    embeddings_np = np.array(embeddings).astype('float32')
    index.add(embeddings_np)         # Add vectors to the index

    end_time = time.time()
    print(f"FAISS index built in {end_time - start_time:.2f} seconds.")
    print(f"Index contains {index.ntotal} vectors.")

    # Save the index
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved to {INDEX_PATH}")

    return index

def load_rag_components(embedding_model_name='all-MiniLM-L6-v2'):
    """Loads the FAISS index, documents, and embedding model."""
    print("Loading RAG components...")
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCUMENTS_PATH):
        print(f"Error: Index ({INDEX_PATH}) or documents ({DOCUMENTS_PATH}) not found.")
        print("Please run the script without skipping generation first.")
        return None, None, None 

    # --- Load Retriever Components ---
    index = faiss.read_index(INDEX_PATH)
    print(f"Loaded FAISS index from {INDEX_PATH} with {index.ntotal} vectors.")
    with open(DOCUMENTS_PATH, 'rb') as f:
        documents = pickle.load(f)
    print(f"Loaded {len(documents)} documents from {DOCUMENTS_PATH}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading embedding model '{embedding_model_name}' on device '{device}'...")
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    print("Embedding model loaded.")

    return index, documents, embedding_model 

def search_index(query, index, documents, model, k=3):
    """Searches the FAISS index for relevant documents."""
    print(f"\nSearching for top {k} documents related to: '{query}'")
    start_time = time.time()

    # Embed the query
    query_embedding = model.encode([query], convert_to_numpy=True, device=model.device)
    query_embedding_np = np.array(query_embedding).astype('float32')

    # Search the index
    distances, indices = index.search(query_embedding_np, k) # D = distances, I = indices

    end_time = time.time()
    print(f"Search complete in {end_time - start_time:.4f} seconds.")

    # Retrieve documents
    retrieved_docs = []
    print("Retrieved documents:")
    for i in range(k):
        doc_index = indices[0][i]
        distance = distances[0][i]
        if doc_index < len(documents):
             doc = documents[doc_index]
             retrieved_docs.append(doc)
             print(f"  - Index: {doc_index}, Distance: {distance:.4f}, ID: {doc['id']}")
             # print(f"    Text: {doc['text'][:150]}...") # Optionally print snippet
        else:
            print(f"  - Warning: Retrieved index {doc_index} is out of bounds for documents list (size {len(documents)}).")


    return retrieved_docs, distances, indices

def generate_answer(query, retrieved_docs, client: openai.OpenAI):
    """Generates an answer using the LM Studio endpoint based on query and retrieved context."""
    print("\nGenerating answer using LM Studio...")
    start_time = time.time()

    # Format the prompt (similar structure, can be adjusted)
    context = "\n".join([f"Document {i+1} (ID: {doc['id']}): {doc['text']}" for i, doc in enumerate(retrieved_docs)])
    prompt = f"""Answer the following question based *only* on the provided context. If the context does not contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

    #print(f"\n--- Prompt ---\n{prompt}\n-------------\n") # Optional: print the prompt

    try:
        # Call LM Studio endpoint
        response = client.chat.completions.create(
            model="local-model", # Model name specified in LM Studio (can be anything)
            messages=[
                {"role": "system", "content": "You are a helpful assistant answering questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Adjust temperature as needed
            max_tokens=100 # Limit response length
        )
        answer = response.choices[0].message.content.strip()

    except openai.APIConnectionError as e:
        print(f"Error connecting to LM Studio API: {e}")
        print("Please ensure LM Studio is running and the server is started.")
        answer = "[Error connecting to LM Studio]"
    except Exception as e:
        print(f"An unexpected error occurred during generation: {e}")
        answer = f"[Error during generation: {e}]"


    end_time = time.time()
    print(f"Answer generation complete in {end_time - start_time:.2f} seconds.")

    return answer

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    # Configure OpenAI client for LM Studio
    # Assumes LM Studio is running on default port localhost:1234
    # The API key is ignored by LM Studio but required by the openai library
    try:
        lm_studio_client = openai.OpenAI(base_url="http://192.168.178.20:1234/v1", api_key="not-needed")
        # Test connection (optional, but good practice)
        lm_studio_client.models.list() # This will fail if server not running
        print("Successfully connected to LM Studio endpoint.")
    except openai.APIConnectionError as e:
         print(f"Failed to connect to LM Studio at http://192.168.178.20:1234/v1: {e}")
         print("Please ensure LM Studio is running and the server tab is active.")
         lm_studio_client = None # Set client to None if connection fails
    except Exception as e:
        print(f"An unexpected error occurred connecting to LM Studio: {e}")
        lm_studio_client = None


    # Check if index and documents already exist
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
        print("Index and documents found on disk.")
        # Load retriever components
        index, documents, embedding_model = load_rag_components()

        # Proceed only if retriever components loaded and LM Studio client is available
        if all([index, documents, embedding_model]) and lm_studio_client:
            # --- Test the full RAG pipeline ---
            sample_query = "What is the capital of Uruguay?"
            retrieved_docs, _, _ = search_index(sample_query, index, documents, embedding_model, k=3)

            # Generate the answer using LM Studio
            final_answer = generate_answer(sample_query, retrieved_docs, lm_studio_client)

            print("\n--- Final Answer (from LM Studio) ---")
            print(final_answer)
        elif not lm_studio_client:
            print("\nCannot run RAG test because LM Studio client is not available.")
        else:
             print("\nRetriever components failed to load. Cannot run RAG test.")


    else:
        print("Index or documents not found. Generating...")
        wiki_dataset = load_wikipedia_dataset()
        documents = prepare_documents(wiki_dataset)
        embeddings = generate_embeddings(documents)
        index = build_and_save_faiss_index(embeddings)

        print("\n--- Index and Documents Generation Complete ---")
        print(f"FAISS index saved to: {INDEX_PATH}")
        print(f"Documents list saved to: {DOCUMENTS_PATH}")
        print("\nRun the script again to test the full RAG pipeline with LM Studio.")
        print("Ensure LM Studio is running with the server started before the next run.")
