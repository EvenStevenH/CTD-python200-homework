from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# ---------------------------------------------------------------------------- #
# Step 1: Setup
if load_dotenv():
    print("API key loaded successfully!")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("../../06_AI_augmentation/resources/groundwork_docs")
# docs_dir = Path("./groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"
print(f"Document directory found!\n")

# Step 2: Load the Documents
docs = SimpleDirectoryReader(docs_dir).load_data()
print(f"Loaded {len(docs)} documents:")
for doc in docs:
    print(f"- {doc.metadata.get('file_name', 'unknown file')}")

# Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(docs)  # in-memory pipeline using docs
engine = index.as_query_engine(similarity_top_k=3)  # query engine with setting
print("Index built successfully. Ready to answer questions.\n")

# ---------------------------------------------------------------------------- #
# Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for i, q in enumerate(questions, start=1):
    response = engine.query(q)
    top_node = response.source_nodes[0]
    doc_name = top_node.node.metadata.get("file_name", "unknown")
    score = round(top_node.score, 4) if top_node.score else "N/A"
    print(f"--- Question {i} of {len(questions)}")
    print(f"Question: {q}")
    print(f"Answer: {response}")
    print(f"Top Retrieved Document Name: {doc_name}")
    print(f"Similarity Score: {score}")
    print(f"Preview (first 200 chars): {top_node.node.text[:200]}\n")

# The model performed very well. For all five responses, the assistant sounded confident and accurate to the source material, each with high similarity scores (in the 0.76–0.90 range). That is, there was strong consistency between questions, retrieved context, and final answers.

# ---------------------------------------------------------------------------- #
# Step 5: Find a Failure

query = "What is the bathroom policy?"
response = engine.query(query)
print(f"Query: {query}")
print(f"Answer: {response}")
print(f"All {len(response.source_nodes)} retrieved source nodes:")
for i, node in enumerate(response.source_nodes, start=1):
    name = node.node.metadata.get("file_name", "unknown file")
    score = round(node.score, 4) if node.score else "N/A"
    print(f"Source Node {i} of {len(response.source_nodes)}")
    print(f"Document Name: {name}")
    print(f"Similarity Score: {score}")
    print(f"Preview (first 200 chars): {node.node.text[:200]}\n")

# I wanted to ask about the bathroom policy, expecting it to be hard because it is a relatively valid question while knowing that the answer is simply not in the documents.

# The retrieval failed, and the model states that the documents do not mention a bathroom policy. The model had became less certain (with similarity scores of 0.70–0.75), but this response is fairly acceptable for this specific question. It acknowledges the failure over hallucinating, but I cannot say the same about trusting other AI-generated responses on more specific questions that nuanced, highly accurate details. To improve the system, I might request more documents and implement fallbacks (for, say, a message if similarity scores fall below a threshold) to improve accuracy and avoid hallucinations.

# ---------------------------------------------------------------------------- #
# Step 6: Reflection

# 1. Framework value (lines of code):
# The manual semantic RAG implementation can require dozens of lines to handle loading, chunking, embedding, indexing, and retrieval. In my project, the equivalent LlamaIndex implementation only required me to use 3 lines to build the index and query engine. This demonstrates how frameworks greatly reduce boilerplate while still providing the same core functionality.

# A hospital's HR and compliance team could use this same approach to let staff ask natural-language questions against internal policy manuals, clinical guideline PDFs, and shift-scheduling documents. Instead of searching through dozens of long documentation by hand, a nurse or administrator could ask "What is the protocol for reporting a needle stick injury?" and get an answer grounded in the actual hospital documents, with the source policy cited (allowing for manual review), saving time and reducing the risk of staff relying on outdated memory of a policy that has since changed.

# One failure mode that RAG cannot fully prevent is incorrect reasoning over correctly retrieved information. Even when the relevant documents are retrieved, the language model can misunderstand the context, combine facts incorrectly, or draw an unsupported conclusion. Retrieval improves grounding but cannot guarantee perfect reasoning.
