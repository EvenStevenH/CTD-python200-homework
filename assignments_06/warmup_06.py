from dotenv import load_dotenv
from pathlib import Path
import string
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# ---------------------------------------------------------------------------- #
# RAG Concepts

# Concepts Question 1
# For Scenario A, I would use RAG because the legal team needs to provide up-to-date answers from a large internal policy library. RAG allows retrieval of specific, current information from the PDFs without needing to fine-tune on them.
# For Scenario B, I would use fine-tuning because the startup has 3,000 examples of the exact brand voice, so the model can learn and replicate that style from examples rather than relying on a generic prompt.
# For Scenario C, I would use RAG because it allows retrieval of specific, current information from a single two-page report.

# Concepts Question 2
# A confidently wrong answer can be more harmful than one that says "I am not sure" because it may create misplaced trust and mislead users into accepting false information without questioning it. For example, if a chatbot confidently provides incorrect medical advice to a patient, it may lead to potential physical harm or poor health decisions. The way the model expresses an answer also affects trust because confidence implies a degree of reliability and authority, even when the content may be inaccurate.

# Concepts Question 3
steps = [
    "Extract text from source documents",  # Text is pulled from relevant sources, such as PDF files.
    "Split text into chunks",  # Large texts are divided into smaller segments for efficient processing.
    "Convert text chunks into embeddings",  # Each chunk is turned into a vector that represents its meaning.
    "Receive the user's query",  # The user asks a question.
    "Embed the user's query",  # The query is converted into an embedding to find similar text in chunks.
    "Retrieve the most relevant chunks",  # The system finds the best-matching text based on similarity to the query embedding.
    "Inject retrieved chunks into the prompt",  # The selected information is added to the input for the LLM.
    "Generate a response from the LLM",  # The final answer is produced using the retrieved context.
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


# Keyword Question 1
query = "What are your hours on weekends?"
simple_keyword_retrieval(query, documents)
# "loyalty.txt" was the selected document. This is actually a tie: hours.txt, hiring.txt, and loyalty.txt each overlap with exactly one query token ("weekends" for hours.txt, "your" for the other two; "your" only survives filtering because "you" is a stopword but "your" is not). The tie-break in scores.sort(reverse=True) falls back to sorting by document name, so "loyalty.txt" wins alphabetically even though it isn't the most relevant document for this query. This is a good illustration of a limitation of pure keyword overlap: word-count matching is a weak relevance signal and can be decided by ties that have nothing to do with actual meaning.

# Keyword retrieval pointed to "loyalty.txt", even though "hours.txt" is the correct answer. This happened because there was an overlap with exactly one token in three documents: "weekends" in "hours.txt", "your" in "hiring.txt", and "your" in "loyalty.txt". The "scores.sort(reverse=True)" in "simple_keyword_retrieval" then puts "loyalty.txt" at the top of these three.

# Keyword Question 2
query = "Do you have anything without caffeine?"
simple_keyword_retrieval(query, documents)
# This prints "No overlapping keywords found." because all documents have zero token overlap with the filtered query tokens ('anything', 'caffeine', 'do', 'have', 'without'). Keyword RAG technically doesn't produce a wrong answer here (it doesn't hallucinate a document), but it still fails the user: menu.txt is the relevant document (it lists drinks and milk options), but it never mentions the literal word "caffeine" or "without," so exact keyword overlap can't find it.

# Semantic RAG would do better here because an embedding model can recognize that "anything without caffeine" is conceptually related to menu items like "decaf," "herbal tea," or milk options, even without any exact word match.

# Keyword Question 3
# I predict that "loyalty.txt" would be selected.
query = "How do I sign up for rewards?"
simple_keyword_retrieval(query, documents)
# After running the code, I found my prediction to be incorrect; the model returns "No overlapping keywords found". I believe this result happened because the model couldn't find exact matches with any of the filtered query tokens like 'do', 'how', 'i', 'rewards', 'sign', and 'up'.

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

docs_dir = Path("./brightleaf_pdfs")
assert docs_dir.exists(), f"Directory not found: {docs_dir}"

docs = SimpleDirectoryReader(docs_dir).load_data()  # load docs
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
        doc_name = node.metadata.get("file_name", "unknown")
        print(f"Node {i} | Document: {doc_name} | Similarity Score: {node.score:.4f}")
        print(f"Chunk Preview: {node.text[:150]}\n")

# The retrieved chunks for query 1 appear relevant, with "security_policy.pdf" as the top source node. The model's tone sounded confident and specific when saying that some benefits include health insurance, vision benefits, wellness programs, and financial security benefits. Nothing unexpected was retrieved.

# The retrieved chunks for query 2 also looked relevant overall, with "security_policy.pdf" as the top source node. The model's tone sounded confident and specific, where it states that some of Brightleaf's security policies include maintaining layered defenses for all networks, requiring multi-factor authentication and VPN with device certificates for access to critical systems, and encrypting customer data in transit and at rest. Nothing unexpected was retrieved.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 2
print("====== LlamaIndex Q2 ======\n")
query2 = "What employee benefits does BrightLeaf offer?"
print(f"Question: {query2}")
for k in [1, 5]:  # rerun query twice; top_k=1 and top_k=5
    engine_k = index.as_query_engine(similarity_top_k=k)
    response = engine_k.query(query2)
    print(f"Answer (top_k={k}): {response}")
    print(f"Retrieved {len(response.source_nodes)} source nodes:")
    for i, node in enumerate(response.source_nodes, start=1):
        doc_name = node.metadata.get("file_name", "unknown")
        print(f"Node {i} | Document: {doc_name} | Similarity Score: {node.score:.4f}")
        print(f"Chunk Preview: {node.text[:150]}\n")


# The response did not change between top_k=1 and top_k=5 on the same query. At top_k=1, the top source was "employee_benefits.pdf" (with similarly score of 0.8893), and the model states that employee benefits include health, vision, wellness benefits, financial security, and retirement benefits. At top_k=5, the top source was also "employee_benefits.pdf" (with the same similarly score of 0.8893), and the model states near-similar benefits.
# Using five chunks provided more supporting context (as shown through top_k=5), but it did not substantially improve the final answer.Additional chunks (like chunks from "earnings_report.pdf" and "partnerships.pdf") can be redundant or irrelevant.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 3
print("====== LlamaIndex Q3 ======\n")
query3 = "For remote contractors working overseas, what is BrightLeaf's parental leave policy?"
response = engine.query(query3)
print(f"Question: {query3}")
print(f"Answer: {response}\n")
print("All retrieved chunks:")
for i, node in enumerate(response.source_nodes, start=1):
    doc_name = node.metadata.get("file_name", "unknown")
    print(f"  Chunk {i} | Document: {doc_name} | Similarity Score: {node.score:.4f}")
    print(f"  Preview: {node.text[:150]}\n")

# I expected the model to fabricate or hallucinate an answer. After retrieving context on remote work and benefits, the top source was "employee_benefits.pdf" (with a similarity score of 0.8243). The model describes the parental leave policy (twelve weeks of paid parental leave to all new parents) and says that unpaid leave can be arranged as needed. While this a fairly acceptable response (it doesn't explicitly address the overseas part of the query), I would improve handling of this kind of query better by seeking additional company documents to help provide more context on different types of employees (such as roles, location, and contract status).

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 4
print("====== LlamaIndex Q4 ======\n")
llm_q4 = OpenAI(model="gpt-4o-mini", temperature=0.2)
faithfulness = FaithfulnessEvaluator(llm=llm_q4)
relevancy = RelevancyEvaluator(llm=llm_q4)

print("=== Evaluation of target query ===")
q1 = "What employee benefits does BrightLeaf offer?"
response1 = engine.query(q1)
faith_result1 = faithfulness.evaluate_response(query=q1, response=response1)
rel_result1 = relevancy.evaluate_response(query=q1, response=response1)
print(f"Faithfulness Evaluation: {faith_result1.score}")
print(f"Relevancy Result: {rel_result1.score}\n")

print("=== Evaluation of low quality query ===")
q2 = "Does BrightLeaf have dogs working in any positions?"
response2 = engine.query(q2)
faith_result2 = faithfulness.evaluate_response(query=q2, response=response2)
rel_result2 = relevancy.evaluate_response(query=q2, response=response2)
print(f"Faithfulness Evaluation: {faith_result2.score}")
print(f"Relevancy Result: {rel_result2.score}\n")

# Faithfulness score of 1 means the answer is accurately supported by the retrieved context, while score of 0 indicates the answer may include inaccuracies or hallucinated details not present in the original context. A relevancy score checks how closely the produced answer addresses/relates to the question, while faithfulness checks if it is supported by the provided documents.

# For my evaluations, only the faithfulness scores changed; both the target query and the low quality query received 1 for relevancy. However, the target query received 1 for faithfulness while the low quality query received 0 for faithfulness. I think this happened because the response for the second query was not faithful to the retrieved contexts and may contain hallucinations or inaccuracies.

# The "LLM-as-a-judge" approach is an evaluation method where a large language model (LLM) itself is used to assess the quality of responses generated by another system. This approach is particularly useful in evaluating Retrieval-Augmented Generation (RAG) systems because it provides more nuanced and context-aware evaluations compared to simple accuracy metrics.
