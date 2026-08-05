from dotenv import load_dotenv
from pathlib import Path
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# ---------------------------------------------------------------------------- #
# Step 1: Setup
if load_dotenv():
    print("API key loaded successfully!")
else:
    print("Warning: could not load API key. Check your .env file.")

docs_dir = Path("./groundwork_docs")
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
for q in questions:
    response = engine.query(q)
    print(f"Question: {q}")
    print(f"Answer: {response}")
    for i, node in enumerate(response.source_nodes):
        top_node = response.source_nodes[0]
        name = top_node.node.metadata.get("file_name", "unknown file")
        score = round(node.score, 4) if node.score else "N/A"
        print(f"Top Source: {name} | Similarity Score: {score}")
        print(f"Preview: {top_node.node.text[:200]}\n")

# For all five responses, the assistant sounded confident, was accurate to the source material, and had high similarity scores (in the 0.76–0.90 range). There was also consistency between questions, retrieved context, and final answers. There was surprisingly little to no signs of hallucinations, meaning the model performed very well.

# ---------------------------------------------------------------------------- #
# Step 5: Find a Failure

query = "What is the bathroom policy?"
response = engine.query(query)
print(f"Query: {query}")
print(f"Answer: {response}")
print("All retrieved source nodes:")
for i, node in enumerate(response.source_nodes):
    top_node = response.source_nodes[0]
    name = top_node.node.metadata.get("file_name", "unknown file")
    score = round(node.score, 4) if node.score else "N/A"
    print(f"Node {i+1}: {name} | Similarity Score: {score}")
    print(f"Preview: {top_node.node.text[:200]}\n")

# I wanted to ask about the bathroom policy. I expected it to be hard because it is a relatively valid question, but the answer is simply not in the documents. When the retrieval failed, the model became less certain (with similarity scores of 0.70–0.75) and mentioned that it could not find the answer. This response is fairly acceptable for this specific question by acknowledging the failure without jumping to hallucinating, but I cannot say the same about trusting other AI-generated responses on more specific questions that nuanced, highly accurate details. To improve the system, I might request more documents and implement fallbacks (for, say, a message if similarity scores fall below a threshold) to improve accuracy and avoid hallucinations.

# ---------------------------------------------------------------------------- #
# Step 6: Reflection

# For my project, the LlamaIndex implementation took less than 5 lines of code to creating the index and query engine, compared to building the semantic rag manually. A framework like LlamaIndex saves much more time and reduces complexity so the developer can focus on the product.

# A different use case using LlamaIndex in a system for answering question from real documents could be for hospitals. That is, staff can use to prompt responses for medical guidelines, clinical standards, and hospital-specific guidelines, saving them time from reading long documents so they can spend it on patients that need them.

# Ine failure mode that RAG cannot fully prevent, even when retrieval is working correctly, is possibly overgeneralization or misinterpretation of information. In other words, it can give a confident but possibly inaccurate answer after confusing context between sources (ex. subtle differences in phrasing) and fabricating answers purely based on the retrieval step.
