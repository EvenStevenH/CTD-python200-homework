from dotenv import load_dotenv
from openai import OpenAI
import json

if load_dotenv():
    print("Successfully loaded api key!\n")
client = OpenAI()


# Task 1: Setup and System Prompt
def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400,
    )
    return response.choices[0].message.content


# I specifically ask the system to use strong action verbs and results-oriented language in order to help the user stand out more to recruiters and people who are reviewing the materials. I also gave them a specific role and specific user to keep responses focused and to avoid tangents.
SYSTEM_PROMPT = """
    You are a professional resume coach and career changer mentor. Your goal is to help users improve their job application materials by providing specific, actionable feedback.

    Constraints:
    - Focus only on resume bullet points and cover letter content
    - Always remind users to review and edit your output before submitting
    - Acknowledge that you may not know the user's specific industry norms
    - Be specific in your improvements rather than just rearranging words
    - Use strong action verbs and results-oriented language

    For resumes, emphasize:
    - Specificity: Include metrics or concrete examples when possible
    - Results orientation: Focus on outcomes rather than just duties
    - Industry-specific terminology: Research the target industry to use appropriate terms
    - Strong action verbs: Replace generic verbs with powerful alternatives

    For cover letters, encourage users to:
    - Connect their background to the job requirements
    - Show enthusiasm for the specific company and role
    - Address any potential concerns the employer might have
    """


# ---------------------------------------------------------------------------- #
# Task 2: Bullet Point Rewriter
# Weak bullet points tend to use vague/passive verbs, give no metrics or context, and describe activities rather than results. The model suggested changes like swapping weak verbs for specific ones (ex. "worked" to "collaborated") and reframed tasks as outcomes.


def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Respond ONLY with valid JSON, no other text (such as the word "json"). Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```{bullet_text}```
    """

    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages)
    clean = (
        response_text.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        improved_bullets = json.loads(clean)  # parse JSON
        for bullet in improved_bullets:  # both versions side by side
            print(f"Original: {bullet['original']}")
            print(f"Improved: {bullet['improved']}\n")
        return improved_bullets
    except json.JSONDecodeError:
        print("Error parsing JSON response.\n" f"Response:\n{clean}\n")
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time",
]
print("Test starter bullets:")
rewrite_bullets(bullets)

# ---------------------------------------------------------------------------- #
# Task 3: Cover Letter Generator
# I chose examples that feature career changers with relevant knowledge transitioning into technical roles. A few-shot pattern helps maintain a confident tone, control structure (3-5 sentences that ends with a clear "why this company" line), and avoid cliches.


def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""You write strong cover letter opening paragraphs for career changers.
        The paragraph should be 3-5 sentences: confident, specific, and free of cliches.

        Here are two examples of the style and tone you should match:

        Example 1:
        Role: Data Analyst at a healthcare nonprofit
        Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
        Opening: After seven years as a registered nurse, I've spent my career making decisions
        under pressure using incomplete information, which turns out to be excellent training for
        data analysis. I recently completed a data analytics program where I built dashboards
        tracking patient outcomes across departments. I'm excited to bring that combination of
        clinical context and technical skill to [Company]'s mission-driven work.

        Example 2:
        Role: Junior Software Engineer at a fintech startup
        Background: Ten years in retail banking operations, self-taught Python developer for two years.
        Opening: I spent a decade on the operations side of banking, watching technology decisions
        get made by people who had never processed a wire transfer or resolved a failed ACH batch.
        That frustration turned into curiosity, and two years of self-teaching Python later, I'm
        ready to be on the other side of those decisions. I'm applying to [Company] because your
        work on payment infrastructure is exactly where my domain expertise and new technical skills
        intersect.

        Now write an opening paragraph for this person:
        Role: {job_title}
        Background: {background}
        Opening:
        """

    messages = [{"role": "user", "content": prompt}]
    response_text = get_completion(messages)
    print(f"Generated cover letter opening:\n{response_text}\n")
    return response_text


job_title = "Junior Data Engineer"
background = (
    "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
)
print("Test cover letter:")
generate_cover_letter(job_title, background)

# ---------------------------------------------------------------------------- #
# Task 4: Moderation Check


def is_safe(text: str) -> bool:
    result = client.moderations.create(model="omni-moderation-latest", input=text)
    flagged = result.results[0].flagged
    if flagged:
        print("Your message was flagged by our content filter. Please rephrase!\n")
        return False
    return True


mod_tests = [
    ("Can you help me rewrite my resume for a programming intern role?", True),
    (
        "I want to defenestrate my interviewer for asking about my greatest weakness.",
        False,
    ),
]
print("Test moderation check:")
for text, expected in mod_tests:
    result = is_safe(text)
    print(f"Test passed: {result == expected}\n")

# ---------------------------------------------------------------------------- #
# Task 5: The Chatbot Loop


def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message and handled it

        # 5. Check if the user wants to rewrite bullets
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print(
                "\nJob Application Helper: Paste your bullet points below, one per line."
            )
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            if raw_bullets:  # add to conversation history
                improved_bullets = rewrite_bullets(raw_bullets)
                messages.append({"role": "user", "content": "\n".join(raw_bullets)})
                messages.append({"role": "assistant", "content": str(improved_bullets)})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input(
                "Job Application Helper: Briefly describe your background: "
            ).strip()

            if job_title and background:  # add to conversation history
                opening_paragraph = generate_cover_letter(job_title, background)
                messages.append(
                    {
                        "role": "user",
                        "content": f"Job Title: {job_title}\nBackground: {background}",
                    }
                )
                messages.append({"role": "assistant", "content": opening_paragraph})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            try:  # add to conversation history
                reply = get_completion(messages)
                print(f"Job Application Helper: {reply}\n")
                messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                print(f"Error getting completion: {e}")


if __name__ == "__main__":
    run_chatbot()

# ---------------------------------------------------------------------------- #
# Task 6: Ethics Reflection

# The bot was trained on text written by and about certain kinds of people. This could produce biased advice that favors certain communication styles, industries, or cultural backgrounds. For example, the model might favor specific action verbs, phrases, or terms that are more common in the tech industry, as prompt examples were tech-based roles.

# If a job-seeker submits the bot's output directly without reviewing it, they might include inaccurate information, miss important details about their actual experience, or be unable to fully explain it. The bot's suggestions should be treated as starting points that require human review and customization.

# One guardrail I would add is a reminder for users to review and edit the produced content before submitting it anywhere. Another would be to implement a disclaimer that the advice is not guaranteed to be accurate or appropriate for all situations, and that users should use their own judgment when applying suggestions.
