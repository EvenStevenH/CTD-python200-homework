from dotenv import load_dotenv
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

# Scenario A: A legal team wants an assistant that can answer questions about their internal policy library — hundreds of PDFs that are updated every quarter.
# I would use RAG because the legal team needs to provide up-to-date answers from a large internal policy library. RAG allows retrieval of specific, current information from the PDFs without needing to fine-tune on them.

# Scenario B: A startup wants their model to write product copy in a very specific brand voice — a dry, minimalist style that does not appear much online. They have 3,000 examples their in-house writers produced over the years.
# I would use fine-tuning because the startup wants their model to write in a very specific, possibly uncommon brand voice. Fine-tuning will help the model learn and replicate this unique style effectively using their vast library of examples.

# Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just received. She does not need this to work for any other document.
# I would use prompt engineering because the data analyst needs answers about a single two-page report. Since it's a one-time use case, prompt engineering to provide the document content directly in the query or using a temporary knowledge base is sufficient/efficient enough.

# ---------------------------------------------------------------------------- #

# Concepts Question 2

# A confidently wrong answer can be more harmful than one that says "I am not sure" because it may mislead users into accepting false information without questioning it. For example, if a chatbot confidently provides incorrect medical advice to a patient, it may lead to potential physical harm or poor health decisions. The way the model expresses an answer also affects trust because confidence implies a degree of reliability and authority, even when the content may be inaccurate.

# ---------------------------------------------------------------------------- #

# Concepts Question 3

# steps = [
#     "Receive the user's query",
#     "Embed the user's query",
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]

# Correct order and description:
# 1. The process starts with getting the input question or request.
# 2. The query is converted into an embedding to find similar text in documents.
# 3. Text is pulled from relevant sources (e.g., PDFs, web pages).
# 4. Large texts are divided into smaller segments for efficient processing.
# 5. Each chunk is turned into an embedding vector.
# 6. The system finds and selects the best-matching text based on similarity to the query embedding.
# 7. The selected information is added to the input for the LLM.
# 8. The final answer is produced, incorporating the retrieved context.

# ---------------------------------------------------------------------------- #


# Keyword RAG
print("====== Keyword RAG ======")


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


documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

# # Keyword Question 1
query = "What are your hours on weekends?"
rag_answer(query, documents)
# loyalty.txt was the selected document, which is incorrect (it would realistically be hours.txt). The hours, hiring, and loyalty documents all have an overlap of 1. However, simple_keyword_retrieval returns whichever one was added to the list first (loyalty.txt in this case) due to next() in its logic.

# Keyword Question 2
query = "Do you have anything without caffeine?"
rag_answer(query, documents)
# As no overlapping keywords were found, no documents were selected. This is false, as the context of the documents imply a cafe setting of sorts (through terms like espresso, lattes, and baristas), but this retrieval method is unable to infer that. I think semantic RAG might be more appropriate, as an embedding model can recognize semantic similarities to a word like "caffeine" without needing an exact match from the document.

# Keyword Question 3
query = "How do I sign up for rewards?"
rag_answer(query, documents)
# I predicted that loyalty.txt would be selected, but no documents. This happened because after stopwords were removed, the code was unable to find any overlapping keywords.

# ---------------------------------------------------------------------------- #

# Semantic RAG Concepts

# Semantic Question 1

# A vector embedding represents text as a numerical vector in multi=dimensional space. In other words, it's a way to convert data (like words and sentences) into numbers that can capture their meaning and relationships.

# The chunk with cosine similarity of 0.85 is more relevant because this higher score indicates stronger semantic alignment with the query compared to the 0.30 score. Cosine similarity measures the angle between vectors, so a value closer to 1 means the texts are conceptually similar.

# Semantic search can find relevant chunks even when exact words don't match because it captures meaning through vector embeddings rather than relying on surface-level keyword matching.

# ---------------------------------------------------------------------------- #
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

brightleaf_dir = "../../06_AI_augmentation/brightleaf_pdfs"
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
    print("All retrieved chunks:")
    for i, node in enumerate(response.source_nodes):
        score = round(node.score, 4) if node.score else "N/A"
        print(f"Source Node {i}")
        print(f"Similarity Score: {score}")
        print(f"Chunk Preview: {node.text[:150]}")

# Retrieval worked well for both queries, and the model provided a relevant answer to the questions. The model's response tone is confident and specific, with little to no hedging language For the first query, it was able to list actual benefit names like the Wellness Reimbursement Plan, Learning Hub, and 401(k) match. For the second query, it was able to cover things like credential rotation, encryption, and ISO 27001 alignment. Both responses had a irrelevant chunk/node, but the model was able to correctly ignore it and focus on the context it needs in each case.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 2
print("====== LlamaIndex Q2 ======\n")
query2 = "What employee benefits does BrightLeaf offer?"
for k in [1, 5]: # reruns query twice with top_k=1 and top_k=5
    engine_k = index.as_query_engine(similarity_top_k=k)
    response = engine_k.query(query2)
    print(f"Question (top_k={k}): {query2}")
    print(f"Answer: {response}\n")
    for i, node in enumerate(response.source_nodes):
        score = round(node.score, 4) if node.score else "N/A"
        print(f"Source Node {i}")
        print(f"Similarity Score: {score}")
        print(f"Chunk Preview: {node.text[:150]}")


# The response remained insufficient for both top_k=1 and top_k=5, where the model will vaguely say that employee benefits exist (but won't go into detail) or say that they are not mentioned in the provided context information.

# At top_k=1, only one chunk was used; its low quality provided no meaningful context. At top_k=5, 5 chunks were used for more potential context, but all of them were still unreadable/corrupted and thus did not improve the output.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 3
print("====== LlamaIndex Q3 ======\n")
query3 = "What new products or services is BrightLeaf planning to launch next year?"
response = engine.query(query3)
print(f"Question: {query3}")
print(f"Answer: {response}\n")
print("All retrieved chunks:")
for i, node in enumerate(response.source_nodes):
    score = round(node.score, 4) if node.score else "N/A"
    print(f"Chunk {i} | Similarity Score: {score}")
    print(f"Preview:\n{node.text[:150]}\n")

# On querying on something vague, something that spans multiple documents, or something where the information might not be in the documents at all, I expected the model to be unable to answer, be vague, or confidently provide a fabricated answer. What happened is the second one: the model was unable to answer, since since there is no relevant information on my query. To change the system to handle this kind of query better, I might set up a minimum relevance threshold or use a system prompt to say that it was unable to answer the query to prevent the possibility of hallucinations.

# ---------------------------------------------------------------------------- #
# LlamaIndex Question 4
print("====== LlamaIndex Q4 ======\n")
llm = OpenAI(model="gpt-4o-mini")
faithfulness = FaithfulnessEvaluator(llm=llm)
relevancy = RelevancyEvaluator(llm=llm)


def evaluate_query(query, type):
    response = engine.query(query)
    faith_result = faithfulness.evaluate_response(response=response)
    rel_result = relevancy.evaluate_response(query=query, response=response)
    print(f"=== Evaluation of an Expected {type} Query ===")
    print(f"Query: {query}")
    print(f"Response: {response}")
    print("Evaluator Results:")
    print(f"Faithfulness score: {faith_result.score}")
    print(f"Relevancy score: {rel_result.score}\n")


evaluate_query("What employee benefits does BrightLeaf offer?", "Good")
evaluate_query("Does BrightLeaf have dogs working in any positions?", "Poor")

# Faithfulness score of 1.0 means the answer is accurately supported by the retrieved context, while score of 0.0 indicates the answer may include inaccuracies or hallucinated details not present in the original context. Relevancy checks how closely the produced answer addresses/relates to the question, while faithfulness checks if it is supported by the provided documents.

# Both faithfulness and relevancy scores changed between the queries, which happens because the first query is generally well-supported by the documents while the second one is not clearly detailed in them. The "LLM-as-a-judge" approach uses a language model to evaluate another model's output. It's used instead of a simple accuracy metric because simple text-match rules via RAG evaluations difficult to score.
