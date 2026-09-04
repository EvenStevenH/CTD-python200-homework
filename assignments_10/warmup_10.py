# Part 1: Warmup

# ML vs. LLM in Pipelines

# ML/LLM Question 1
# The ML classifier (joblib model) produces binary predictions and probability scores, while the LLM (OpenAI GPT) produces a natural-language recommendation. If swapped: using the LLM to predict binary outcome would reduce reliability since it's not trained on the prediction task, while using the ML model to write text would produce less natural, more formulaic outputs.

# ML/LLM Question 2
# For converting date string to day-of-week, I'd use deterministic code (datetime library) for simple mapping with no ambiguity. For classifying job posting by experience level, I'd use a trained ML model because classification benefits from training on specific examples like labeled text data. For predicting customer churn, I'd use a trained ML model on historical behavioral data because predictions requires statistical modeling of complex patterns over time. For normalizing city names, I'd use deterministic code with canonical mapping because consistent formatting doesn't require learning; just rule-based transformations. For summing a column of revenue, I'd use deterministic code because simple arithmetic is best done by straightforward computation, such as via numpy/pandas operations.

# ML/LLM Question 3
# Incremental processing ensures a pipeline only does new/unprocessed data on each run. It's important because it avoid unnecessary reprocessing of all records, which could be costly and risky (overwriting previous enrichment results). If 365 records were re-processed every time, it could cause unnecessary costs, latency, and possible inconsistency in regenerated outputs.

# ---------------------------------------------------------------------------- #
# Prompt Design

# Prompt Question 1
SYSTEM_PROMPT_2 = (
    "You are providing a two-sentence running recommendation for a daily weather summary app. "
    "Sentence 1: State the prediction (good/skip) with clear intent. "
    "Sentence 2: Explain 1-2 key factors that support your recommendation. "
    "Do not use bullet points or phrases like 'Based on the data'."
)
# To accommodate two sentences instead of one, I might update the logic to check for line breaks or multiple lines in the response, as well as splitting and validating each sentence separately.


# Prompt Question 2
# This call_with_retry function would be used in production pipelines to handle transient/temporary API issues without failing completely and help improve service availability by reattempting operations.
import time


def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                return None
