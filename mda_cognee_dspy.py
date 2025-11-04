import logging
import re
import os
import uuid
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import dspy
import cognee
import weaviate
from transformers import AutoTokenizer, AutoModel
from llama_index.readers.file import UnstructuredReader
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.vector_stores.types import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_PERSIST_FNAME,
    MetadataFilters,
    FilterCondition,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from qdrant_client import QdrantClient
from dspy.retrieve.qdrant_rm import QdrantRM
from unstructured.partition.auto import partition

# Initialize tokenizer and model for encoding queries
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def encode_query(query: str) -> np.ndarray:
    """Encode a query string into a vector embedding.

    Args:
        query: The query string to encode

    Returns:
        A numpy array representing the query embedding
    """
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    # Use mean pooling to convert token embeddings to a single sentence embedding
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


# Set up logging to file with UTF-8 encoding to handle Unicode characters
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# Initialize the Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Collection name
COLLECTION_NAME = "llama_index_doc"

# Initialize the Qdrant retrieval model
qdrant_retriever_model = QdrantRM(COLLECTION_NAME, qdrant_client, k=10)

# Check if the collection exists or needs to be created
try:
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    logging.info(f"Collection '{COLLECTION_NAME}' already exists with configuration: {collection_info.config}")
except Exception as e:
    if "not found" in str(e):
        logging.info(f"Collection '{COLLECTION_NAME}' does not exist. Attempting to create without specifying vector size.")
        qdrant_client.create_collection(collection_name=COLLECTION_NAME)
        logging.info(f"Created collection '{COLLECTION_NAME}' with automatic vector sizing.")
    else:
        logging.error(f"An error occurred while accessing collection info: {str(e)}")

# Configure DSPy settings with Claude and the Qdrant retrieval model
api_key = os.environ.get("ANTHROPIC_API_KEY")
claude = dspy.Claude(model="claude-3-haiku-20240307", api_key=api_key)
dspy.settings.configure(lm=claude, rm=qdrant_retriever_model)

# Set up the Weaviate client
weaviate_client = weaviate.Client(
    url=os.environ["WEAVIATE_URL"],
    auth_client_secret=weaviate.AuthClientSecret(
        api_key=os.environ["WEAVIATE_API_KEY"]
    )
)

# Configure Cognee - API key should be set in environment, not hardcoded
if "OPENAI_API_KEY" not in os.environ:
    logging.warning("OPENAI_API_KEY not set in environment variables. Please set it before running.")

# Function to load documents
def load_documents(file_path: str) -> List[Document]:
    """Load documents from a file using unstructured partition.

    Args:
        file_path: Path to the file to load

    Returns:
        List of Document objects with text and metadata
    """
    logging.info(f"Loading documents from: {file_path}")
    elements = partition(filename=file_path)
    docs = [Document(text=str(el), metadata={"source": file_path}) for el in elements]
    logging.info(f"Loaded {len(docs)} documents from {file_path}")
    return docs

from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.data_structs.data_structs import IndexDict

# Initialize the VectorStoreIndex with the OpenAIEmbedding model
vector_store = VectorStoreIndex(
    embed_model=OpenAIEmbedding(),
    store_nodes_override=True,
    index_struct=IndexDict()
)

def add_documents_to_collection(documents, qdrant_client, collection_name, vector_store):
    nodes = [TextNode(text=doc.text) for doc in documents]  # Create TextNode instances for each document
    embedded_nodes = vector_store._get_node_with_embedding(nodes)  # Use the vector store to process nodes and retrieve embeddings

    points = []
    for node in embedded_nodes:
        if hasattr(node, 'embedding') and node.embedding is not None:
            embedding = node.embedding
            if isinstance(embedding, list):
                # If embedding is already a list, use it directly
                vector = embedding
            else:
                # If embedding is a numpy array, convert it to a list
                vector = embedding.tolist()

            point = {
                "id": str(uuid.uuid4()),
                "payload": {"document": node.text},
                "vector": vector
            }
            points.append(point)

    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        logging.info("Added documents successfully.")
    except Exception as e:
        logging.error(f"Failed to add documents: {e}")

# Example usage - This will be executed in __main__ block
# Documents should be loaded and processed in the main execution block

# Use Cognee with the configured Weaviate client
def add_documents_to_weaviate(documents):
    # Implement the logic to add documents and embeddings to Weaviate
    for doc in documents:
        weaviate_client.data_object.create(
            data_object=doc.text,
            class_name="Document"
        )

def search_weaviate(query):
    # Implement the logic to search for relevant documents in Weaviate based on the query
    search_results = weaviate_client.query.get(
        class_name="Document",
        properties=["text"],
        where={
            "path": ["text"],
            "operator": "Contains",
            "valueString": query
        },
        limit=5
    )
    return [result["text"] for result in search_results["data"]["Get"]["Document"]]


class RerankingSignature(dspy.Signature):
    document_id = dspy.InputField(desc="ID of the document", type=str)
    initial_score = dspy.InputField(desc="Initial score from the first retrieval phase", type=float)
    query = dspy.InputField(desc="User query for contextual relevance", type=str)
    features = dspy.InputField(desc="Features extracted for reranking", type=list)
    reranked_score = dspy.OutputField(desc="Recomputed score after reranking", type=float)

class RerankModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=10)  # Utilizing QdrantRM via global settings

    def forward(self, document_id, query, initial_score):
        context = self.retrieve(query).passages
        reranked_score = initial_score + len(context)  # Simplistic reranking logic
        return reranked_score


def calculate_ndcg(predicted_relevance, true_relevance, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
    
    Args:
        predicted_relevance (list): List of predicted relevance scores.
        true_relevance (list): List of true relevance scores.
        k (int): The rank position to calculate NDCG for (default: 10).
    
    Returns:
        float: NDCG score at rank k.
    """
    if len(predicted_relevance) == 0 or len(true_relevance) == 0:
        return 0.0
    
    # Sort predicted relevance scores in descending order
    sorted_indices = np.argsort(predicted_relevance)[::-1]
    
    # Calculate Discounted Cumulative Gain (DCG) at rank k
    dcg = 0.0
    for i in range(min(k, len(sorted_indices))):
        idx = sorted_indices[i]
        relevance = true_relevance[idx]
        dcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Calculate Ideal Discounted Cumulative Gain (IDCG) at rank k
    ideal_relevance = sorted(true_relevance, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_relevance))):
        relevance = ideal_relevance[i]
        idcg += (2 ** relevance - 1) / np.log2(i + 2)
    
    # Calculate NDCG
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

class RerankingOptimizer(dspy.Module):
    def __init__(self, rerank_module):
        super().__init__()
        self.rerank_module = rerank_module
        self.lm = dspy.settings.lm  # Get the language model from global settings
        self.teleprompter = BootstrapFewShotWithRandomSearch(
            metric=self.custom_metric,
            teacher_settings={'lm': self.lm},  # Use the explicitly passed LM
            max_bootstrapped_demos=2,  # Reduce the number of bootstrapped demos
            max_labeled_demos=8,  # Reduce the number of labeled demos
            num_candidate_programs=4,  # Reduce the number of candidate programs
            num_threads=4
        )

    def custom_metric(self, predictions, labels, extra_arg=None):
        logging.debug(f"custom_metric called with predictions: {predictions}, labels: {labels}")
        if len(predictions) == 0 or len(labels) == 0:
            logging.warning("Empty predictions or labels")
            return 0

        predicted_scores = []
        true_scores = []

        for pred in predictions:
            try:
                score = float(pred.split('reranked_score:')[1].split()[0])
                predicted_scores.append(score)
            except (IndexError, ValueError):
                logging.warning(f"Error extracting predicted score from: {pred}")
                pass

        for label in labels:
            try:
                score = float(label.split('reranked_score:')[1].split()[0])
                true_scores.append(score)
            except (IndexError, ValueError):
                logging.warning(f"Error extracting true score from: {label}")
                pass

        if len(predicted_scores) == 0 or len(true_scores) == 0:
            logging.warning("Empty predicted_scores or true_scores")
            return 0

        if len(predicted_scores) != len(true_scores):
            logging.warning("Mismatch in lengths of predicted_scores and true_scores")
            return 0

        logging.debug(f"Predicted scores: {predicted_scores}")
        logging.debug(f"True scores: {true_scores}")

        squared_errors = [(pred_score - true_score) ** 2 for pred_score, true_score in zip(predicted_scores, true_scores)]
        
        if len(squared_errors) == 0:
            logging.warning("Empty squared_errors")
            return 0
        
        logging.debug(f"Squared errors: {squared_errors}")
        
        mse = np.mean(squared_errors)
        logging.debug(f"MSE: {mse}")
        
        return mse

    def optimize_reranking(self, document_ids, initial_scores, query):
        logging.debug(f"optimize_reranking called with document_ids: {document_ids}, initial_scores: {initial_scores}, query: {query}")
        if len(document_ids) == 0 or len(initial_scores) == 0:
            logging.error("Empty training set.")
            return None

        def trainset_generator():
            logging.debug("trainset_generator called")
            for i, (doc_id, score) in enumerate(zip(document_ids, initial_scores)):
                logging.debug(f"Generating example {i+1}/{len(document_ids)}")
                logging.debug(f"Document ID: {doc_id}")
                logging.debug(f"Initial Score: {score}")
                logging.debug(f"Query: {query}")
                example = dspy.Example(
                    document_id=doc_id,
                    initial_score=score,
                    query=query
                ).with_inputs("document_id", "initial_score", "query")
                logging.debug(f"Generated example: {example}")
                yield example

        try:
            logging.info("Starting optimization...")
            optimized_program = self.teleprompter.compile(
                student=self.rerank_module,
                trainset=trainset_generator()
            )
            logging.info("Optimization completed.")
            return optimized_program
        except ZeroDivisionError as e:
            logging.error(f"Division by zero error during optimization: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Failed to optimize reranking: {str(e)}")
            return None



class QueryPlanningSignature(dspy.Signature):
    query = dspy.InputField(desc="User query")
    agent_ids = dspy.InputField(desc="Available agent IDs")
    historical_data = dspy.InputField(desc="Historical performance data", optional=True)
    selected_agents = dspy.OutputField(desc="Agents selected for the query")

class QueryPlanner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.process_query = dspy.ChainOfThought(QueryPlanningSignature)

    def forward(self, query, agent_ids, historical_data=None):
        context = f"Query: {query}\nAgents: {agent_ids}\nHistorical Data: {historical_data if historical_data else 'No historical data'}"
        prediction = self.process_query(query=query, agent_ids=agent_ids, historical_data=historical_data)
        return prediction.selected_agents if hasattr(prediction, 'selected_agents') else []

class DocumentAgent(dspy.Module):
   def __init__(self, document_id, content, qdrant_client, collection_name):
       super().__init__()
       self.document_id = document_id
       self.content = content
       self.qdrant_client = qdrant_client
       self.collection_name = collection_name
       self.lm = dspy.settings.lm  # Assuming Claude is configured globally

       # Add the document content to Cognee's knowledge base
       cognee.add(content)

   def request(self, prompt):
       """Makes a request to the Anthropic API using the provided prompt."""
       try:
           response = self.lm(prompt)

           # Check if the response is a string
           if isinstance(response, str):
               # If the response is a string, return it as is
               return response
           elif isinstance(response, list):
               # If the response is a list, join the elements into a string
               return " ".join(response)
           elif isinstance(response, dict):
               # If the response is a dictionary, check for a 'response' key
               if 'response' in response:
                   return response['response']
               else:
                   logging.warning("'response' key not found in response dictionary")
           else:
               # If the response is neither a string, list, nor a dictionary, log a warning
               logging.warning(f"Unexpected response format: {type(response)}")

       except Exception as e:
           logging.error(f"Error during Anthropic API request: {str(e)}")

       # If any of the above cases fail, return None
       return None

   def encode_query(self, query):
       inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
       outputs = model(**inputs)
       # Use mean pooling to convert token embeddings to a single sentence embedding
       return outputs.last_hidden_state.mean(dim=1).detach().numpy()

   def fetch_updated_data(self, query):
       """ Fetches updated or additional data relevant to the query from Qdrant. """
       try:
           batch_results = self.qdrant_client.query_batch(
               self.collection_name,
               query_texts=[query],
               limit=3  # Fetch the top 3 relevant documents
           )
           logging.debug(f"Batch results: {batch_results}")
           additional_data = " ".join([result.payload["document"] for batch in batch_results for result in batch])
       except Exception as e:
           logging.error(f"Error during Qdrant search: {str(e)}")
           additional_data = ""
       
       return additional_data

   def evaluate(self, query):
       """ Evaluates the query by fetching data based on the query context and returns a score. """
       if "update" in query.lower():  # Check if the query involves updating data
           updated_content = self.fetch_updated_data(query)
           content_to_use = f"{self.content}\n{updated_content}"
       else:
           content_to_use = self.content

       logging.debug(f"Content to use: {content_to_use}")
       
       # Retrieve relevant information from Cognee's knowledge base
       cognee_info = cognee.search("SIMILARITY", f"document_id: {self.document_id}, query: {query}")
       
       prompt = f"Evaluate the following content based on the query: {query}\nContent: {content_to_use}\nCognee Information: {cognee_info}"
       logging.debug(f"Prompt: {prompt}")
       
       try:
           response = self.request(prompt)  # Use the request method to make the API call
           logging.debug(f"Raw API response: {response}")
           
           if isinstance(response, str):
               if "does not directly answer" in response.lower() or "not relevant" in response.lower():
                   score = 0.0  # Assign a score of 0 if the content does not answer the query
               elif "provides some information" in response.lower() or "partially relevant" in response.lower():
                   score = 0.5  # Assign a score of 0.5 if the content provides some information but not a complete answer
               else:
                   score = 1.0  # Assign a score of 1 if the content directly answers the query
           else:
               logging.warning("Unexpected response format")
               score = 0.0  # Default score if the response format is unexpected
       except Exception as e:
           logging.error(f"Error during Anthropic API request: {str(e)}")
           score = 0.0  # Handle any exceptions and assign a score of 0
       
       logging.debug(f"Evaluation score: {score}")
       return score

   def answer_query(self, query):
       """ Uses the evaluate method to process the query and fetch the final answer from the LM """
       # Break down the query into sub-queries
       sub_queries = self.break_down_query(query)
       
       # Initialize an empty list to store the answers for each sub-query
       sub_answers = []
       cited_documents = []  # Initialize a list to store cited documents
       
       for sub_query in sub_queries:
           score = self.evaluate(sub_query)
           logging.debug(f"Sub-query score: {score}")
           
           if score > 0:
               # Extract the relevant information from the content for the sub-query
               relevant_parts = self.extract_answer(sub_query)
               
               # Generate an answer for the sub-query using the language model
               sub_answer = self.generate_answer(sub_query, relevant_parts)
               sub_answers.append(sub_answer)
               
               # Add the current document to the cited_documents list
               cited_documents.append(self.document_id)
       
       # Combine the answers from all sub-queries
       combined_answer = " ".join(sub_answers)
       
       # Retrieve relevant information from Cognee's knowledge base
       cognee_info = cognee.search("SIMILARITY", query)
       
       # Refine the combined answer using the language model and Cognee's information
       refined_answer = self.refine_answer(query, combined_answer, cognee_info)
       
       # Add citations to the final answer
       cited_docs_str = ", ".join([f"Document {doc_id}" for doc_id in cited_documents])
       final_answer = f"{refined_answer}\n\nCited documents: {cited_docs_str}"
       
       return final_answer

   def break_down_query(self, query):
       """ Breaks down a complex query into smaller sub-queries """
       # Use a pre-trained question decomposition model or rule-based approach
       # to break down the query into sub-queries
       sub_queries = []
       
       # Example: Split the query based on keywords like "and", "or", "additionally", etc.
       sub_queries = re.split(r"\b(and|or|additionally)\b", query, flags=re.IGNORECASE)
       sub_queries = [q.strip() for q in sub_queries if q.strip()]
       
       return sub_queries

   def generate_answer(self, query, relevant_parts):
       """ Generates an answer using the language model based on the query and relevant parts """
       prompt = f"Query: {query}\nRelevant information: {' '.join(relevant_parts)}\nAnswer:"
       response = self.request(prompt)
       
       if response:
           return response.strip()
       else:
           return "I don't have enough information to answer this query."

   def refine_answer(self, query, answer, cognee_info):
       """ Refines the generated answer using the language model and Cognee's information """
       prompt = f"Query: {query}\nGenerated answer: {answer}\nCognee Information: {cognee_info}\nRefined answer:"
       response = self.request(prompt)
       
       if response:
           return response.strip()
       else:
           return answer
       
   def extract_answer(self, query):
       """ Extracts the relevant information from the document content to construct an answer """
       # Preprocess the query and content
       processed_query = self.preprocess_text(query)
       processed_content = self.preprocess_text(self.content)

       # Perform relevance scoring or information extraction techniques
       # to identify the most relevant parts of the content
       relevant_parts = self.find_relevant_parts(processed_query, processed_content)

       # Construct the answer based on the relevant parts
       answer = self.construct_answer(relevant_parts)

       return answer

   def preprocess_text(self, text):
       """ Preprocesses the text by lowercasing, removing punctuation, etc. """
       # Implement text preprocessing steps here
       processed_text = text.lower()
       # Add more preprocessing steps as needed
       return processed_text

   def find_relevant_parts(self, query, content):
       """ Finds the most relevant parts of the content based on the query """
       # Convert the content into sentences
       sentences = self.split_into_sentences(content)
       
       # Calculate the similarity between the query and each sentence
       similarities = []
       for sentence in sentences:
           similarity = self.calculate_similarity(query, sentence)
           similarities.append(similarity)
       
       # Sort the sentences based on their similarity scores
       sorted_sentences = [x for _, x in sorted(zip(similarities, sentences), reverse=True)]
       
       # Return the top N most relevant sentences
       top_n = 3  # Adjust the number of relevant sentences to return
       relevant_parts = sorted_sentences[:top_n]
       
       return relevant_parts

   def split_into_sentences(self, text):
       """ Splits the text into sentences """
       # You can use a library like NLTK or spaCy for more accurate sentence splitting
       # For simplicity, we'll use a basic approach here
       sentences = text.split(". ")
       return sentences

   def calculate_similarity(self, query, sentence):
       """ Calculates the similarity between the query and a sentence """
       # You can use more advanced similarity metrics like cosine similarity or TF-IDF
       # For simplicity, we'll use the Jaccard similarity here
       query_words = set(query.split())
       sentence_words = set(sentence.split())
       intersection = query_words.intersection(sentence_words)
       union = query_words.union(sentence_words)
       similarity = len(intersection) / len(union)
       return similarity

   def construct_answer(self, relevant_parts):
       """ Constructs the answer based on the relevant parts """
       # Join the relevant parts into a coherent answer
       answer = " ".join(relevant_parts)
       
       # Perform any necessary post-processing or formatting
       answer = answer.capitalize()
       
       return answer




class MasterAgent(dspy.Module):
   def __init__(self, document_agents, reranker, query_planner):
       super().__init__()
       self.document_agents = document_agents
       self.reranker = reranker
       self.query_planner = query_planner

   def process_query(self, query):
       # Use the query planner to determine which agents to involve in the query process
       selected_agents = self.query_planner.forward(query, list(self.document_agents.keys()))
       
       # Print the selected agents
       selected_agents_str = ", ".join([f"Document {agent_id}" for agent_id in selected_agents])
       logging.info(f"Selected agents for query '{query}': {selected_agents_str}")

       # Evaluate the query using the selected agents, generating initial scores
       initial_scores = {agent_id: agent.evaluate(query) for agent_id, agent in self.document_agents.items() if agent_id in selected_agents}

       # Rerank the results based on the initial scores
       results = {doc_id: self.reranker.forward(doc_id, query, score) for doc_id, score in initial_scores.items()}

       # Handle cases where no valid results are found
       if not results:
           return "No documents found."

       # Retrieve relevant information from Cognee's knowledge base
       cognee_info = cognee.search("SIMILARITY", query)

       # Identify the top document based on the reranked scores and get the final answer
       top_doc_id = max(results, key=results.get)
       agent_response = self.document_agents[top_doc_id].answer_query(query)

       # Combine Cognee's information with the agent response
       final_answer = f"Cognee Information: {cognee_info}\n\nAgent Response: {agent_response}"
       
       return final_answer



if __name__ == "__main__":
   logging.basicConfig(filename='app.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
   logging.info("Starting the document processing application.")

   try:
       # Load file path from environment variable or use default
       file_path = os.environ.get("DOCUMENT_PATH", "docs/latest.md")
       documents = load_documents(file_path)
       
       if not documents:
           logging.error("No documents found. Exiting.")
           exit()

       logging.info(f"Loaded documents: {[doc.metadata['source'] for doc in documents]}")
       add_documents_to_collection(documents, qdrant_client, COLLECTION_NAME, vector_store)

       # Add documents to Cognee's knowledge base
       for doc in documents:
           cognee.add(doc.text)
       cognee.cognify()

       # Add documents to Weaviate
       add_documents_to_weaviate(documents)

       # Update DocumentAgent initialization to include qdrant_client and COLLECTION_NAME
       document_agents = {str(idx): DocumentAgent(document_id=idx, content=doc.text, qdrant_client=qdrant_client, collection_name=COLLECTION_NAME) for idx, doc in enumerate(documents)}
       logging.info(f"Created {len(document_agents)} document agents.")

       reranker = RerankModule()
       optimizer = RerankingOptimizer(reranker)
       query_planner = QueryPlanner()
       master_agent = MasterAgent(document_agents, reranker, query_planner)

       query = "what is class VectorStoreIndex(BaseIndex[IndexDict]):?"
       logging.info(f"Processing query: {query}")
       
       # Search for relevant documents in Weaviate
       weaviate_results = search_weaviate(query)
       logging.info(f"Weaviate search results: {weaviate_results}")

       response = master_agent.process_query(query)  # Directly process the query without optimization
       logging.info(f"Response: {response}")

   except Exception as e:
       logging.error(f"An error occurred during application execution: {str(e)}")
       logging.error(traceback.format_exc())  # Provides a stack trace
