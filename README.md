# Multi-Document Agent System (MDAS)

## Overview

MDAS leverages a distributed agent-based architecture to enhance document processing through a smart partitioning of textual documents. By encapsulating the semantic meaning of document partitions in semi-autonomous agents, it offers parallel querying and reasoning across a comprehensive knowledge space.

## System Architecture

Detailed documentation for the system architecture can be found in the following sections:

- [Document Agents](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/system-architecture/document-agents.mdx)
- [Qdrant Vector Database](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/system-architecture/qdrant-vector-database.mdx)
- [Vector Embeddings](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/system-architecture/vector-embeddings.mdx)

## Query Processing

Explore how MDAS processes queries through these detailed documents:

- [Master Agent](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/query-processing/master-agent.mdx)
- [Query Planner](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/query-processing/query-planner.mdx)
- [Reranking Module](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/query-processing/reranking-module.mdx)

## Optimization Techniques

Understand the optimization techniques used in MDAS:

- [Bootstrapped Few-Shot Learning](https://github.com/jmanhype/DSPy-Multi-Document-Agents/blob/main/pages/optimization-techniques/bootstrapped-few-shot-learning.mdx)

## Getting Started

To set up the MDAS:

```bash
git clone https://github.com/jmanhype/DSPy-Multi-Document-Agents.git
cd DSPy-Multi-Document-Agents
pip install -r requirements.txt
```

## Usage

Run the system with:

```bash
python main.py
```

## Contributing

Contributions are welcome! Please fork the project, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
