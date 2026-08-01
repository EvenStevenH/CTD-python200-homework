from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key!\n")
client = OpenAI()


def get_response(message, sys_message=None):
    messages = [{"role": "user", "content": message}]
    if sys_message:  # add to messages
        messages.insert(0, {"role": "system", "content": sys_message})

    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return response.choices[0].message.content


# The Chat Completions API
# ---------------------------------------------------------------------------- #
print("=== API Q1 ===")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "What is one thing that makes Python a good language for beginners?",
        }
    ],
)

response_txt = response.choices[0].message.content
print(
    f"Model response: {response_txt}\n"
    f"Model Name: {response.model}\n"
    f"Tokens Used: {response.usage.total_tokens}\n"
)

# ---------------------------------------------------------------------------- #
print("=== API Q2 ===")

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=temp,
    )
    print(
        f"Response with temperature {temp}:\n"
        f"{response.choices[0].message.content}\n"
    )

# The output for temperature=0 is almost always the same ("DataForge Solutions"). The output for temperature=0.7 and temperature=1.5 varies more ("Data Catalyst Collective", for instance),  sometimes returning a list of ideas or including reasoning in the response. For more consistent, reproducible outputs, I would use temperature=0.

# ---------------------------------------------------------------------------- #
print("=== API Q3 ===")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Give me a one-sentence fun fact about pandas (the animal, not the library).",
        }
    ],
    n=3,
    temperature=1.0,
)
for i, choice in enumerate(response.choices):
    print(f"Response {i + 1}:\n" f"{choice.message.content}\n")

# ---------------------------------------------------------------------------- #
print("=== API Q4 ===")

prompt = "Explain how neural networks work."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=15,
)
print(f"Prompt: '{prompt}'\n" f"Response: {response.choices[0].message.content}\n")

# The response is cut off at 15 tokens, rendering it incomplete. I would use `max_tokens` in real applications to control costs, speed up inference, and prevent excessive generation.

# ---------------------------------------------------------------------------- #
# System Messages and Personas
print("=== System Q1 ===")

messages = [
    {
        "role": "system",
        "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"Response 1: {response.choices[0].message.content} \n")

messages = [
    {
        "role": "system",
        "content": "You are a boisterous, unhelpful Python teacher. You talk like a pirate and would rather share sea tales instead of teaching.",
    },
    {"role": "user", "content": "I don't understand what a list comprehension is."},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"Response 2: {response.choices[0].message.content} \n")

# While maintaining the system's role as a Python teacher, the personality and response tone is drastically different. The first response is more clear and encouraging, while the second speaks like a pirate and is not overly focused on helping the user. Both still offers an example of list comprehension in Python.

# ---------------------------------------------------------------------------- #
print("=== System Q2 ===")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {
        "role": "assistant",
        "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?",
    },
    {"role": "user", "content": "Can you remind me what my name is?"},
]
response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
print(f"System Question 2 response: {response.choices[0].message.content}\n")

# The Chat Completions API is stateless. The model still knows Jordan's name because messages included previous responses that has Jordan's name in them, thus providing context.

# ---------------------------------------------------------------------------- #
# Prompt Engineering
print("=== Prompt Q1 — Zero-Shot ===")

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow.",
]
for i, review in enumerate(reviews):
    response = get_response(
        f"""Review: "{review}"

        Classify the sentiment of this review as positive, negative, or mixed.""",
    )
    print(f"Review {i + 1}: {response}")

# ---------------------------------------------------------------------------- #
print("\n=== Prompt Q2 — One-Shot ===")

example = """
        Example:
        Review: "Fast shipping but the item arrived damaged."
        Sentiment: mixed
        """
for i, review in enumerate(reviews):
    prompt = f"""
        Example:
        Review: "Fast shipping but the item arrived damaged."
        Sentiment: mixed

        Classify the sentiment of this review as positive, negative, or mixed.

        Review: "{review}"
        Sentiment:
        """
    response = get_response(prompt)
    print(f"Review {i + 1}: {response}")

# By adding an example, the model was able to respond in the same format as the example for each review (a "Sentiment" label and a value in lowercase).

# ---------------------------------------------------------------------------- #
print("\n=== Prompt Q3 — Few-Shot ===")

example = """
        Example 1:
        Review: "The order was received two days early, and the high quality was exactly as expected."
        Sentiment: positive

        Example 2:
        Review: "The item came in just in time, but in the wrong color."
        Sentiment: mixed

        Example 3:
        Review: "The item never arrived, and the seller refused a refund."
        Sentiment: negative

        """
for i, review in enumerate(reviews):
    prompt = f"""
        Classify the sentiment of this review as positive, negative, or mixed.
        {example}

        Review: "{review}"
        Sentiment:
        """
    response = get_response(prompt)
    print(f"Review {i + 1}: {response}")

# For zero-shot, the model only uses pre-trained data to guess the output and format you wanted. One-shot and few-shot prompts provide examples to the model to reference, allow the model to recognize patterns and handle the task more reliably. As such, I would use zero-shot for completing simple tasks fast (like math), and one-shot or few-shot when when the task is nuanced or I need a specific output format.

# ---------------------------------------------------------------------------- #
print("\n=== Prompt Q4 — Chain of Thought ===")

prompt = f"""
    A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later takes a new job that pays $7,500 more per year than her post-raise salary. What is her final annual salary?
    """
response = get_response(
    prompt,
    sys_message="""
    Solve this problem and show your reasoning step by step before giving a final answer. Wrap the final answer in <answer></answer>.
    """,
)
print(f"Response: {response}\n")

# Asking the model to break down the problem into smaller, individual predictions tend to improve accuracy and reduce model hallucination.

# ---------------------------------------------------------------------------- #
print("=== Prompt Q5 — Structured Output ===")

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."
response = get_response(
    review,
    sys_message="Analyze this customer review and respond only with valid JSON. Return three keys: sentiment (positive/negative/mixed), confidence (0–1, float), reason (one sentence).",
)
print(f"Raw response: {response}\n")

try:
    response_json = json.loads(response)
    print(
        f"Sentiment: {response_json['sentiment']}\n"
        f"Confidence: {response_json['confidence']}\n"
        f"Reason: {response_json['reason']}\n"
    )
except json.JSONDecodeError:
    print(f"Not a valid JSON format. Raw response: {response}\n")
except KeyError:
    print(f"Incorrect JSON key. Raw response: {response}\n")

# ---------------------------------------------------------------------------- #
print("=== Prompt Q6 — Delimiters ===")

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."
```{user_text}```
"""
response = get_response(prompt)
print(f"Response 1:\n{response}\n")  # numbered list

user_text = "There's a crazy wolverine in the kitchen! He look mean and voracious."
prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."
```{user_text}```
"""
response = get_response(prompt)
print(f"Response 2:\n{response}\n")  # "No steps provided."

# Delimiters help clearly separate user instructions from data, reducing the risk of prompt injection and misinterpretation.

# ---------------------------------------------------------------------------- #
# Local Models with Ollama
print("=== Ollama Q1 ===")

prompt = "Explain what a large language model is in two sentences."
response = get_response(prompt)
print(f"OpenAI response: {response}\n")

"""
Ollama output:

A large language model is an AI system designed to understand and generate
human language, trained on vast datasets to learn patterns and improve
accuracy over time. It can comprehend complex contexts and produce coherent
text for tasks such as writing, translation, and content creation.
"""

"""
OpenAI output:

A large language model is an advanced artificial intelligence system designed to understand and generate human language by analyzing vast amounts of text data. It uses machine learning techniques, particularly deep learning, to predict and produce coherent and contextually relevant text based on the input it receives.
"""

# Both Ollama's and OpenAI's responses are very similar, with Ollama's having a more broader explanation and using less technical terms (like "machine learning"). I think the main advantage of running a model locally is more privacy, offline access, and your data not being used for unauthorized training. Conversely, local models do not update (their training data may only contain knowledge up to a certain date) and cannot search the internet for additional information.
