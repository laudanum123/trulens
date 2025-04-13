# hf_rag_project/src/evaluate_rag.py
import datasets
import openai
from dotenv import load_dotenv
import os
import time
import argparse 
import numpy as np

# Import functions from our build_rag script
from build_rag import load_rag_components, search_index, generate_answer

# TruLens imports
from trulens.core import TruSession, Feedback, Select # Core components
from trulens.apps.app import instrument
from trulens.apps.custom import TruCustomApp # Keep TruCustomApp for wrapping
from trulens.providers.openai import OpenAI as fOpenAI # Provider path

# Constants
DATASET_NAME = "wiki_dpr" 
DATASET_SPLIT = "test" 
NUM_QUESTIONS_TO_EVALUATE = 10

def load_questions(dataset_path="rag-datasets/rag-mini-wikipedia", config_name='question-answer', split=DATASET_SPLIT):
    """Loads the questions from the specified Hugging Face dataset."""
    print(f"Loading questions from dataset '{dataset_path}', config '{config_name}', split '{split}'...")
    try:
        qa_dataset_split = datasets.load_dataset(dataset_path, name=config_name, split=split)
        questions = qa_dataset_split['question'] 
        print(f"Loaded {len(questions)} questions.")
        return questions
    except KeyError:
        print(f"Error: Could not find 'question' column in the loaded dataset split.")
        try:
            print("Available features:", qa_dataset_split.features)
        except Exception:
            pass
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

class RAGApp:
    def __init__(self, index, documents, embedding_model, llm_client):
        self.index = index
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    @instrument # Add decorator
    def retrieve_context(self, query: str) -> list:
        """Retrieves context documents for a given query."""
        retrieved_docs, _, _ = search_index(query, self.index, self.documents, self.embedding_model, k=3)
        return [doc['text'] for doc in retrieved_docs]

    @instrument # Add decorator
    def generate_response(self, query: str, contexts: list) -> str:
        """Generates an answer based on the query and retrieved contexts."""
        # generate_answer expects docs with 'id' and 'text'
        dummy_retrieved_docs = [{'id': f'context_{i}', 'text': ctx} for i, ctx in enumerate(contexts)]
        answer = generate_answer(query, dummy_retrieved_docs, self.llm_client)
        return answer

    @instrument # Add decorator
    def query(self, query: str) -> str:
        """Main query method combining retrieval and generation."""
        contexts = self.retrieve_context(query)
        answer = self.generate_response(query, contexts)
        return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with TruLens")
    parser.add_argument("--reset-db", action="store_true", help="Reset the TruLens database before evaluation.")
    args = parser.parse_args()

    print("--- Starting RAG Evaluation with TruLens ---")
    start_time = time.time()

    load_dotenv()
    lm_studio_client = None
    try:
        lm_studio_base_url = os.getenv("LM_STUDIO_URL", "http://192.168.178.20:1234/v1") 
        lm_studio_client = openai.OpenAI(base_url=lm_studio_base_url, api_key="not-needed")
        lm_studio_client.models.list()
        print(f"Successfully connected to LM Studio endpoint at {lm_studio_base_url}.")
    except Exception as e:
        print(f"Failed to connect to LM Studio: {e}. Please ensure server is running.")
        exit()

    print("Initializing TruLens...")
    tru_session = TruSession(database_url="sqlite:///trulens_eval.db") # Use TruSession with database_url
    if args.reset_db:
        print("Resetting TruLens database.")
        tru_session.reset_database() # Use TruSession method
    else:
        print("Appending to existing TruLens database (if any). Use --reset-db to clear.")

    provider = fOpenAI(model_engine="local-model", client=lm_studio_client)
    print(f"Configured TruLens feedback provider to use LM Studio.")

    # Updated Selectors for instrumented methods
    select_context = Select.RecordCalls.retrieve_context.rets # Context from retrieve_context
    select_query = Select.RecordInput # Input to the main 'query' method
    select_answer = Select.RecordOutput # Output of the main 'query' method

    # Feedback functions using updated selectors
    f_groundedness = Feedback(
        provider.groundedness_measure_with_cot_reasons,
        name="Groundedness"
    ).on(source=select_context, statement=select_answer) # Use context and final answer

    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons,
        name="Answer Relevance"
    ).on(prompt=select_query, response=select_answer) # Use main query input and final answer

    f_context_relevance = Feedback(
        provider.context_relevance_with_cot_reasons,
        name="Context Relevance"
    ).on(question=select_query, context=select_context).aggregate(np.mean) # Use main query input and context

    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
    print("Defined TruLens feedbacks: Groundedness, Answer Relevance, Context Relevance")

    print("Loading RAG components...")
    index, documents, embedding_model = load_rag_components()
    if not all([index, documents, embedding_model]):
        print("Failed to load RAG components. Exiting.")
        exit()

    rag_app_instance = RAGApp(index, documents, embedding_model, lm_studio_client)

    # Automatic version tracking setup
    from datetime import datetime
    app_base_name = "HF_RAG_LMStudio"
    version = datetime.now().strftime("%Y%m%d_%H%M%S")  # Timestamp as version

    # Deprecation warnings noted, using TruCustomApp as per current script structure
    tru_rag_recorder = TruCustomApp(
        rag_app_instance,
        app_name = app_base_name,
        app_version = version,
        feedbacks=feedbacks
    )
    print(f"Instrumented RAG pipeline with TruLens (version {version}).")

    questions = load_questions()
    if not questions:
        print("Failed to load questions. Exiting.")
        exit()

    # Randomly sample questions
    import random
    random.seed(42)  # For reproducibility
    sampled_questions = random.sample(questions, min(NUM_QUESTIONS_TO_EVALUATE, len(questions)))
    print(f"\n--- Evaluating {len(sampled_questions)} randomly sampled questions with TruLens ---")
    evaluation_records = []
    for i, question in enumerate(sampled_questions):
        print(f"\n--- Question {i+1}/{len(sampled_questions)} ---")
        print(f"Q: {question}")

        # Call the instrumented app directly using the main 'query' method
        # The 'with' block is no longer needed for recording decorated methods
        llm_response = None
        record_data = None
        try:
            # Use the context manager and call the original app instance
            with tru_rag_recorder as recording:
                llm_response = rag_app_instance.query(question)

            # Get the record after the 'with' block completes
            record_data = recording.get()
            if record_data: # Check if record exists
                 evaluation_records.append(record_data) # Store the record

                 # Access record data for printing
                 try:
                     # Find the retrieve_context call in the record
                     retrieve_call = next(call for call in record_data.calls if call.stack[-1].method.name == 'retrieve_context')
                     context_texts = retrieve_call.rets
                     print("Retrieved Contexts (Top 3):")
                     for ctx_idx, ctx in enumerate(context_texts):
                         text_preview = ctx[:100] + "..." if len(ctx) > 100 else ctx
                         print(f"  - Context {ctx_idx+1}: '{text_preview}'")
                 except (StopIteration, AttributeError, IndexError, TypeError) as e: # Added TypeError
                      print(f"Could not retrieve context details from record: {e}")
                      print("Record structure might have changed or call not found.")

                 if llm_response:
                     print(f"A: {llm_response}") # Print the direct response
                 else:
                      print("A: No response generated.")
            else:
                 print("Error: Failed to get record data.")
                 evaluation_records.append(None) # Indicate failure

        except Exception as e:
            print(f"Error during RAG query execution or recording: {e}")
            # Optionally add placeholder record or skip
            evaluation_records.append(None) # Indicate failure for this question

        time.sleep(0.5) # Keep the delay

    end_time = time.time()
    print(f"\n--- TruLens Evaluation Complete ---")
    successful_evals = sum(1 for rec in evaluation_records if rec is not None)
    print(f"Successfully evaluated {successful_evals}/{len(sampled_questions)} questions in {end_time - start_time:.2f} seconds.")
    print("Evaluation records saved to TruLens database (if successful).")
