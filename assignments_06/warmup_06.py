from dotenv import load_dotenv
from pathlib import Path
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully!\n")
else:
    print("Warning: could not load API key. Check your .env file.\n")

# ---------------------------------------------------------------------------- #
# RAG Concepts

# Concepts Question 1
# For Scenario A, I would use RAG because the legal team needs to provide up-to-date answers from a large internal policy library. RAG allows retrieval of specific, current information from the PDFs without needing to fine-tune on them.
# For Scenario B, I would use fine-tuning because the startup wants their model to write in a very specific, possibly uncommon brand voice. Fine-tuning will help the model learn and replicate this unique style effectively using their vast library of examples.
# For Scenario C, I would use prompt engineering because the data analyst needs answers about a single two-page report. Since it's a one-time use case, prompt engineering to provide the document content directly in the query or using a temporary knowledge base is sufficient/efficient enough.

# Concepts Question 2
# A confidently wrong answer can be more harmful than one that says "I am not sure" because it may mislead users into accepting false information without questioning it. For example, if a chatbot confidently provides incorrect medical advice to a patient, it may lead to potential physical harm or poor health decisions. The way the model expresses an answer also affects trust because confidence implies a degree of reliability and authority, even when the content may be inaccurate.

# Concepts Question 3
steps = [
    "Receive the user's query",  # process gets the input question or request.
    "Embed the user's query",  # the query is converted into an embedding to find similar text in documents.
    "Extract text from source documents",  # text is pulled from relevant sources (e.g., PDFs, web pages).
    "Split text into chunks",  # large texts are divided into smaller segments for efficient processing.
    "Convert text chunks into embeddings",  # each chunk is turned into an embedding vector.
    "Retrieve the most relevant chunks",  # the system finds and selects the best-matching text based on similarity to the query embedding.
    "Inject retrieved chunks into the prompt",  # the selected information is added to the input for the LLM.
    "Generate a response from the LLM",  # the final answer is produced, incorporating the retrieved context.
]

# ---------------------------------------------------------------------------- #


# Keyword RAG
print("====== Keyword RAG ======")
documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}


def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "in",
        "on",
        "of",
        "for",
        "to",
        "is",
        "are",
        "was",
        "were",
        "by",
        "with",
        "at",
        "from",
        "that",
        "this",
        "as",
        "be",
        "it",
        "its",
        "their",
        "they",
        "we",
        "you",
        "our",
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator) for w in query.lower().split() if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


def rag_answer(query, documents):
    result = simple_keyword_retrieval(query, documents)
    selected_name = result[0][0]
    print(f"Selected document: {selected_name}\n")


# Keyword Question 1
query = "What are your hours on weekends?"
rag_answer(query, documents)
# hours.txt was selected because the query matches with keywords from the document, such as the word "weekends".

# Keyword Question 2
query = "Do you have anything without caffeine?"
rag_answer(query, documents)
# The model returns "No overlapping keywords found", and thus no documents were selected. Keyword RAG got this technically right because it couldn't find exact matches with any of the filtered query tokens, especially with 'caffeine'.
# I think semantic RAG might be more appropriate, as an embedding model can recognize semantic similarities to a word like "caffeine" without needing an exact match (for instance, inferences through document terms like "espresso" and "lattes").

# Keyword Question 3
# I predict that loyalty.txt would be selected.
query = "How do I sign up for rewards?"
rag_answer(query, documents)
# After running the code, I found my prediction to be incorrect. The model returns "No overlapping keywords found", and thus no documents were selected. I believe this result happened because the model couldn't find exact matches with any of the filtered query tokens like 'do', 'how', 'i', 'rewards', 'sign', and 'up'.

# ---------------------------------------------------------------------------- #

# Semantic RAG Concepts

# Semantic Question 1
# A vector embedding represents text as a numerical vector in multi=dimensional space. In other words, it's a way to convert data (like words and sentences) into numbers that can capture their meaning and relationships.
# The chunk with cosine similarity of 0.85 is more relevant because this higher score indicates stronger semantic alignment with the query compared to the 0.30 score. Cosine similarity measures the angle between vectors, so a value closer to 1 means the texts are conceptually similar.
# Semantic search can find relevant chunks even when exact words don't match because it captures meaning through vector embeddings rather than relying on surface-level keyword matching.

# Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG |
# |----------------------------|-----------------------------------|--------------|
# | What is compared?          | Exact word overlap                | Semantic similarity (cosine distance) |
# | What is retrieved?         | Full document                     | Relevant segments based on semantic meaning |
# | Can it handle synonyms?    | No                                | Yes - captures conceptual relationships |
# | Storage format             | Plain text dictionary             | Numerical vectors in embedding space |
# | Relevance score            | Number of overlapping keywords    | Semantic alignment score (cosine similarity) |

# ---------------------------------------------------------------------------- #
# LlamaIndex

brightleaf_dir = Path("./brightleaf_pdfs")
assert brightleaf_dir.exists(), f"Directory not found: {brightleaf_dir}"

docs = SimpleDirectoryReader(brightleaf_dir).load_data()  # load docs
index = VectorStoreIndex.from_documents(docs)  # in-memory pipeline using docs
engine = index.as_query_engine(similarity_top_k=3)  # query engine with setting

# LlamaIndex Question 1
print("====== LlamaIndex Q1 ======\n")
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for q in questions:
    response = engine.query(q)
    print(f"Q: {q}")
    print(f"A: {response}\n")
    print(f"Retrieved {len(response.source_nodes)} source nodes:")
    for i, node in enumerate(response.source_nodes, start=1):
        score = round(node.score, 4) if node.score else "N/A"
        print(f"Node {i} | Similarity Score: {score}")
        print(f"Chunk Preview:\n{node.text[:150]}\n")

# The retrieved chunks for query 1 looked relevant overall. The employee benefits guide was the top source, and the company overview plus remote work policy appeared as weaker supporting context. The model's tone sounded confident and specific because it listed concrete benefits rather than using hedging language.

# The retrieved chunks for query 2 also looked relevant overall. The dedicated security policy file was the top source, while the company overview and remote work policy showed up as broader supporting material. The model's tone sounded confident and specific because it listed policies rather than using hedging language. Nothing particularly unexpected was retrieved, but it did seem that the high-level company overview appeared in both queries because it shares general company language.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 2
print("====== LlamaIndex Q2 ======\n")
query2 = "What employee benefits does BrightLeaf offer?"
print(f"Question: {query2}")
for k in [1, 5]:  # reruns query twice with top_k=1 and top_k=5
    engine_k = index.as_query_engine(similarity_top_k=k)
    response = engine_k.query(query2)
    print(f"Answer (top_k={k}): {response}")
    print(f"Retrieved {len(response.source_nodes)} source nodes:")
    for i, node in enumerate(response.source_nodes, start=1):
        score = round(node.score, 4) if node.score else "N/A"
        print(f"Node {i} | Similarity Score: {score}")
        print(f"Chunk Preview:\n{node.text[:150]}\n")


# The response changed between both top_k=1 and top_k=5. At top_k=1 (with similarly scores of 0.80) the model will say that employee benefits exist (but won't go into detail), stating, "BrightLeaf offers a variety of employee benefits." At top_k=5 with (similarly scores ranging form 0.77–0.80), the model will say that employee benefits can't be found, stating, "The employee benefits offered by BrightLeaf are not explicitly mentioned in the provided context information."

# Using five chunks provided more supporting context (as shown through top_k=5), but it did not substantially improve the final answer;more retrieved context is not always better because additional chunks can be redundant or less relevant.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 3
print("====== LlamaIndex Q3 ======\n")
query3 = "For remote contractors working overseas, what is BrightLeaf's parental leave policy?"
response = engine.query(query3)
print(f"Question: {query3}")
print(f"Answer: {response}\n")
print("All retrieved chunks:")
for i, node in enumerate(response.source_nodes):
    score = round(node.score, 4) if node.score else "N/A"
    print(f"Chunk {i} | Similarity Score: {score}")
    print(f"Preview:\n{node.text[:150]}\n")

# I expected the model to fabricate an answer. After retrieving context on remote work and benefits, the model states that the documents do not provide a parent leave policy for remote contractors working overseas. This is an acceptable response, as the documents focused on full-time employees and remote workers based in the US. To handle this kind of query better, I might seek additional documents or metadata to provide more context on employees (such as roles, location, and contract status).

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 4
print("====== LlamaIndex Q4 ======\n")
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

print("=== Evaluation of target query ===")
q1 = "What employee benefits does BrightLeaf offer?"
response1 = engine.query(q1)
faith_result1 = faithfulness_evaluator.evaluate_response(response=response1)
rel_result1 = relevancy_evaluator.evaluate_response(query=q1, response=response1)
print(f"Faithfulness Evaluation: {str(faith_result1.score)}")
print(f"Relevancy Result: {str(rel_result1.score)}\n")

print("=== Evaluation of low quality query ===")
q2 = "Does BrightLeaf have dogs working in any positions?"
response2 = engine.query(q2)
faith_result2 = faithfulness_evaluator.evaluate_response(response=response2)
rel_result2 = relevancy_evaluator.evaluate_response(query=q2, response=response2)
print(f"Faithfulness Evaluation: {str(faith_result2.score)}")
print(f"Relevancy Result: {str(rel_result2.score)}\n")

# Faithfulness score of 1.0 means the answer is accurately supported by the retrieved context, while score of 0.0 indicates the answer may include inaccuracies or hallucinated details not present in the original context. Relevancy checks how closely the produced answer addresses/relates to the question, while faithfulness checks if it is supported by the provided documents.

# The scores did not change for both queries: both the target query and the low quality query received 0.0 for faithfulness and 0.0 for relevancy. I think this happened because the response was not faithful to the retrieved contexts and may contain hallucinations or inaccuracies.

# The "LLM-as-a-judge" approach is an evaluation method where a large language model (LLM) itself is used to assess the quality of responses generated by another system. This approach is particularly useful in evaluating Retrieval-Augmented Generation (RAG) systems because it provides more nuanced and context-aware evaluations compared to simple accuracy metrics.
