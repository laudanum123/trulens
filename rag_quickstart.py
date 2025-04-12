import os
from dotenv import load_dotenv
load_dotenv()
from trulens.apps.app import instrument
from trulens.core import TruSession, Feedback, Select
from trulens.providers.openai import OpenAI as TruLensOpenAI  # TruLens provider

from openai import OpenAI as OfficialOpenAI  # Official OpenAI client
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import numpy as np
import chromadb

# Initialize session
session = TruSession()
session.reset_database()

# Set up OpenAI client for LM Studio
# Point to your LM Studio server – adjust host and port as needed.
lm_studio_client = OfficialOpenAI(base_url="http://192.168.178.20:1234/v1", api_key="lm-studio")

# Create Vector Store (uses official OpenAI API for embeddings)
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(
    name="Washington", embedding_function=embedding_function
)

# Add sample data
data = {
    "uw_info": """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
""",
    "wsu_info": """
Washington State University, commonly known as WSU, founded in 1890, is a public research university in Pullman, Washington.
With multiple campuses across the state, it is the state's second largest institution of higher education.
WSU is known for its programs in veterinary medicine, agriculture, engineering, architecture, and pharmacy.
""",
    "seattle_info": """
Seattle, a city on Puget Sound in the Pacific Northwest, is surrounded by water, mountains and evergreen forests, and contains thousands of acres of parkland.
It's home to a large tech industry, with Microsoft and Amazon headquartered in its metropolitan area.
The futuristic Space Needle, a legacy of the 1962 World's Fair, is its most iconic landmark.
""",
    "starbucks_info": """
Starbucks Corporation is an American multinational chain of coffeehouses and roastery reserves headquartered in Seattle, Washington.
As the world's largest coffeehouse chain, Starbucks is seen to be the main representation of the United States' second wave of coffee culture.
""",
    "newzealand_info": """
New Zealand is an island country located in the southwestern Pacific Ocean. It comprises two main landmasses—the North Island and the South Island—and over 700 smaller islands.
The country is known for its stunning landscapes, ranging from lush forests and mountains to beaches and lakes. New Zealand has a rich cultural heritage, with influences from 
both the indigenous Māori people and European settlers. The capital city is Wellington, while the largest city is Auckland. New Zealand is also famous for its adventure tourism,
including activities like bungee jumping, skiing, and hiking.
"""
}

for key, doc in data.items():
    print(f"Adding {key} to vector store...")
    vector_store.add(key, documents=doc)

# RAG Implementation
class RAG:
    @instrument
    def retrieve(self, query: str) -> list:
        results = vector_store.query(query_texts=query, n_results=2)
        print(f"Retrieved {len(results['documents'])} documents for query: {query}")
        return [doc for sublist in results["documents"] for doc in sublist]

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        if not context_str:
            return "Sorry, I couldn't find an answer."
        
        completion = lm_studio_client.chat.completions.create(
            model="my-generation-model", 
            temperature=0,
            messages=[{
                        "role": "user",
                        "content": f"We have provided context information below. \n"
                        f"---------------------\n"
                        f"{context_str}"
                        f"\n---------------------\n"
                        f"First, say hello and that you're happy to help. \n"
                        f"\n---------------------\n"
                        f"Then, given this information, please answer the question: {query}",
                    }
                ],
            ).choices[0].message.content
        
        return completion or "Did not find an answer."

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        return self.generate_completion(query, context_str)

# Initialize RAG and TruLens
rag = RAG()
provider = TruLensOpenAI(
    client=lm_studio_client, 
    model_engine="my-critique-model" 
)

# Feedback functions
f_groundedness = Feedback(
    provider.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on(Select.RecordCalls.retrieve.rets.collect()).on_output()

f_answer_relevance = Feedback(
    provider.relevance_with_cot_reasons,
    name="Answer Relevance"
).on_input().on_output()

f_context_relevance = Feedback(
    provider.context_relevance_with_cot_reasons,
    name="Context Relevance"
).on_input().on(Select.RecordCalls.retrieve.rets[:]).aggregate(np.mean)


# Run the application
if __name__ == "__main__":
    from trulens.apps.custom import TruCustomApp
    
    tru_rag = TruCustomApp(
        rag,
        app_name="RAG",
        app_version="base",
        feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
    )
    
    with tru_rag as recording:
        print(rag.query("What wave of coffee culture does Starbucks represent?"))
        print(rag.query("Does Washington State University have Starbucks?"))
        from trulens.dashboard import run_dashboard

        run_dashboard(session)
