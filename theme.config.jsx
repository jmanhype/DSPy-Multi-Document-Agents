export default {
      logo: <span>Multi-Document Agent Q&A System - made with Lumentis</span>,
      editLink: {
        component: null,
      },
      project: {
        link: "https://github.com/hrishioa/lumentis",
      },
      feedback: {
        content: null,
      },
      footer: {
        text: (
          <>
            Made with 🫶 by&nbsp;
            <a href="https://twitter.com/hrishioa" target="_blank">
              Hrishi - say hi!
            </a>
          </>
        ),
      },
      head: (
        <>
          <meta property="og:title" content="Multi-Document Agent Q&A System" />
          <meta property="og:description" content="This is a Python script that implements a document processing and question-answering system using various libraries and modules. The script imports several libraries, including logging, re, numpy, dspy (Divination Synthesis Python), llama_index, qdrant_client, unstructured, transformers, and torch.The script sets up logging configuration, initializes a Qdrant vector database client, and configures the Dspy settings with the Claude language model and the Qdrant retrieval model. It defines functions to load documents, encode queries, and add documents to the Qdrant collection.The script also includes several custom classes:1. RerankingSignature and RerankModule: Classes for reranking document scores based on contextual relevance.2. RerankingOptimizer: A class for optimizing the reranking process using a custom metric and the BootstrapFewShotWithRandomSearch technique from Dspy.3. QueryPlanningSignature and QueryPlanner: Classes for planning which document agents to involve in answering a query.4. DocumentAgent: A class representing an individual document agent that can evaluate queries, fetch updated data, and generate answers using the Claude language model.5. MasterAgent: A class that orchestrates the overall query processing by utilizing the query planner, document agents, and reranker.The script includes a main entry point that loads documents, creates document agents, initializes the reranker, optimizer, query planner, and the master agent. It then processes a sample query using the master_agent.process_query() method, which involves selecting relevant document agents, evaluating the query, reranking the results, and generating the final answer with citations.The output of the script includes logging information about the process, the final answer to the query, and citations for the document agents used to generate the answer. Based on the code, it appears to be processing technical documentation related to the LlamaIndex library and the VectorStoreIndex class.A few clues point to this:1. It loads a document from the path "C:/Users/strau/storm/docs.llamaindex.ai/en/latest.md", which suggests it is processing documentation for the LlamaIndex project.2. In the main execution part, it processes the query "what is class VectorStoreIndex(BaseIndex[IndexDict]):?", which is asking about the VectorStoreIndex class from LlamaIndex.3. It imports several modules and classes from the llama_index library, such as Document, TextNode, VectorStoreIndex, etc.So this script seems to be designed to take technical documentation content related to the LlamaIndex library, specifically focusing on understanding and explaining the VectorStoreIndex class. It utilizes natural language processing techniques, vector embeddings, and a retrieval system (Qdrant) to process the documentation and answer queries about the content." />
          <meta name="robots" content="noindex, nofollow" />
          <link rel="icon" type="image/x-icon" href="https://raw.githubusercontent.com/jmanhype/DSPy-Multi-Document-Agents/main/favicon.ico" />
        </>
      ),
    };
    