import json
import matplotlib
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from scipy.stats import stats
from smolagents import ToolCallingAgent, CodeAgent, OpenAIServerModel, tool

warnings.filterwarnings("ignore", category=DeprecationWarning)

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")

client = OpenAI()
print("OpenAI client created.")

# ---------------------------------------------------------------------------- #
# Lesson 02: Tool Definitions and the ReAct Loop
print("\n====== Q1")  # Q1


def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"


tools = [
    {
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a Celsius temperature to Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {"celsius": {"type": "number"}},
                "required": ["celsius"],
            },
        },
    }
]
for temp in [0, 100, -40]:
    print(celsius_to_fahrenheit(temp))

# ---------------------------------------------------------------------------- #
print("\n====== Q2")  # Q2

tools_q2 = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_agent(user_prompt: str) -> str:
    """Run a minimal ReAct-style agent for a single user prompt."""

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""

    # Step 1: start the conversation with system and user messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools_q2,
        tool_choice="auto",  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Error: unknown tool {function_name}."

            # Print for debugging so we can see what happened
            print("Tool called:", function_name)
            print("Tool result:", tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ""
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ""


# Q2 Predictions:
# I think run_agent("Convert 100 degrees Celsius to Fahrenheit") will not trigger a tool call. The prompt does not mention "time", so the model should answer directly.
# I think only one API call will be made. The first (and only) call returns content without tool_calls.
result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
print(result)
# Prediction was correct: no tool call triggered, and only one API call made.


# ---------------------------------------------------------------------------- #
print("\n====== Q3")  # Q3


# extend agent
def run_agent(user_prompt: str) -> str:
    """
    Same fixed 2-call structure as run_agent from the lesson, but with two tools.
    Dispatches get_current_time and celsius_to_fahrenheit by name.
    """

    SYSTEM_PROMPT = """
    You are a helpful assistant that can tell the current time
    and convert temperatures from Celsius to Fahrenheit."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    first_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    first_message = first_response.choices[0].message
    messages.append(
        {
            "role": "assistant",
            "content": first_message.content,
            "tool_calls": first_message.tool_calls,
        }
    )

    if first_message.tool_calls:
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if function_name == "get_current_time":
                tool_result = get_current_time()
            elif function_name == "celsius_to_fahrenheit":
                tool_result = celsius_to_fahrenheit(**args)
            else:
                tool_result = f"Error: unknown tool {function_name}."

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": tool_result,
                }
            )

        second_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
        )
        return second_response.choices[0].message.content or ""
    else:
        return first_message.content or ""


tools = [
    {
        "type": "function",
        "function": {
            "name": "celsius_to_fahrenheit",
            "description": "Convert a Celsius temperature to Fahrenheit.",
            "parameters": {
                "type": "object",
                "properties": {"celsius": {"type": "number"}},
                "required": ["celsius"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current local time as a string.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)
# Yes, a tool was called (celsius_to_fahrenheit) because the prompt asks about a temperature conversion.

response_b = run_agent("What is the boiling point of water in plain English?")
print("\nResponse B:", response_b)
# No tool was called because the question does not require time or temperature conversion. The model provides the boiling point of water (100C / 212F) directly from its knowledge.

# ---------------------------------------------------------------------------- #
# Lesson 03: Multi-Tool Agent
print("\n====== Q4")  # Q4


class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------
    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------
    def list_csv_files(self) -> dict:
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str) -> dict:
        """
        Load a CSV file from resources/ and make it the active dataset.
        filename can be "bike_commute" or "bike_commute.csv".
        """
        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        self.df = pd.read_csv(path)
        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self) -> list:
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list = None) -> dict:
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str) -> dict:
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {
                "error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"
            }

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str = None, plot_type: str = "line") -> str:
        """
        Plot from the active CSV.
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
        if x == y:
            x = None
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
        title_csv = self.csv_name or "current CSV"

        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."

        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"

        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()

        return f"Plotted {y} vs {x} as a {plot_type}."

    # The lesson hit a tool-round limit here because no tool existed for correlation.
    # Adding it as a method keeps all CSV operations in one place and follows the
    # same pattern the lesson uses for describe_column.

    def compute_correlation(self, col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """
        error = self._ensure_loaded()
        if error:
            return error
        if col1 not in self.df.columns:
            return {
                "error": f"Column '{col1}' not found. Options: {self.df.columns.tolist()}"
            }
        if col2 not in self.df.columns:
            return {
                "error": f"Column '{col2}' not found. Options: {self.df.columns.tolist()}"
            }

        try:
            cols = self.df[[col1, col2]].dropna()
            r, p = stats.pearsonr(cols[col1], cols[col2])
            return {
                "col1": col1,
                "col2": col2,
                "pearson_r": round(float(r), 4),
                "p_value": round(float(p), 4),
            }
        except Exception as e:
            return {"error": str(e)}


matplotlib.use("Agg")
RESOURCES_DIR = Path("./resources")
csv_manager = CsvManager(resources_dir=RESOURCES_DIR)

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources directory.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": (
                "Load a CSV file from resources/ and make it the active dataset. "
                "Pass just the filename, e.g. 'bike_commute' or 'bike_commute.csv'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name of the CSV file to load from resources/.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Return the column names of the currently loaded CSV.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Return descriptive statistics for a single column in the loaded CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "The column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": (
                "Return summary statistics for one or more columns. "
                "Pass a list of column names, or omit to summarize all columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of column names to summarize. Omit to summarize all.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": (
                "Plot data from the loaded CSV. Supports 'line' and 'scatter' plot types. "
                "x is optional; if omitted, plots y vs row index."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {
                        "type": "string",
                        "description": "The column to plot on the y-axis.",
                    },
                    "x": {
                        "type": "string",
                        "description": "The column to plot on the x-axis. Optional.",
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter"],
                        "description": "Type of plot. Either 'line' or 'scatter'.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": (
                "Compute the Pearson correlation coefficient and p-value between "
                "two numeric columns in the loaded CSV."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "Name of the first column.",
                    },
                    "col2": {
                        "type": "string",
                        "description": "Name of the second column.",
                    },
                },
                "required": ["col1", "col2"],
            },
        },
    },
]
node_tools = {
    "list_csv_files": csv_manager.list_csv_files,
    "load_csv": csv_manager.load_csv,
    "get_columns": csv_manager.get_columns,
    "describe_column": csv_manager.describe_column,
    "summarize_columns": csv_manager.summarize_columns,
    "plot_data": csv_manager.plot_data,
    "compute_correlation": csv_manager.compute_correlation,
}
SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short."
)


def run_agent():
    """
    Simple command-line chat loop so it feels like a chatbot.
    We keep a single 'messages' list for the whole session so the model
    sees the conversation history each turn.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("CSV data agent at your service. Here to help look at your CSV data!")
    print("Type a question. Type 'exit' to quit.\n")
    print("To start, try 'list csv files' or 'load bike_commute.csv'\n")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ["exit", "quit", "q"]:
            print("Bye.")
            break

        print(f"User query: {user_text}")
        assistant_text = run_agent_cycle(messages, user_text)
        print(f"\nAssistant: {assistant_text}\n")


def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt,
    and then user text will be appended.

    The loop has three main steps:

    REASON:
    - Call the model with the conversation so far.
    - The model either replies normally, or asks to call a tool from tool set.

    ACT:
    - If tools are requested, run the Python functions

    OBSERVE:
    - Append each requested tool result back into the LLMs conversation history.
    - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
    - If the model returns an assistant message with no tool calls, this is the
        final answer for this react cycle, this implies that reasoning alone without
        tool calls was enough.
    - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = (
            json.dumps(result, default=str) if not isinstance(result, str) else result
        )
        tool_message = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )
        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")
            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}

            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))

            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


# ---------------------------------------------------------------------------- #
print("\n====== Q5")  # Q5 > Recreate the scenario that hit tool-round limit

messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(
    messages,
    "Load bike_commute.csv and compute the correlation between "
    "avg_traffic_density and avg_speed_kmh.",
)
print(result)

# ---------------------------------------------------------------------------- #
print("\n====== Q6")  # Q6

# Q6 roles:
# "system" is the initial instruction that sets the agent's behavior and persona, establishing what the agent should do (the "R" setup in ReAct).
# "user" is the human user's question or task, which triggers the ReAct cycle.
# "assistant" is the model's response. If it contains tool_calls, this is the "Act" step (deciding which tool to use and with what arguments). If it contains plain text, this is the final "Respond" step.
# "tool" is the result returned by executing the tool. This is the "Observe" step. The model reads this result and decides what to do next.
print(json.dumps(messages, indent=2, default=str))

# ---------------------------------------------------------------------------- #
# Lesson 04: smolagents
print("\n====== Q7")  # Q7


@tool
def list_csv_files() -> dict:
    """List available CSV files in resources/.
    Returns:
        A dict with a "files" list, or a message if none are found.
    """
    return csv_manager.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from resources/ and make it the active dataset.
    Args:
        filename: CSV filename in resources/. You can pass "bike_commute" or "bike_commute.csv".
    Returns:
        A dict with a status message and column names, or an error dict.
    """
    return csv_manager.load_csv(filename)


@tool
def get_columns() -> dict:
    """Return column names for the currently loaded CSV.
    Returns:
        A list of column names, or an error dict if no CSV is loaded.
    """
    return csv_manager.get_columns()


@tool
def summarize_columns(columns: list = None) -> dict:
    """Return summary stats for selected columns (or all columns).
    This includes count, mean, std, min, max, and percentiles for numeric columns,
    or count, unique, top, freq for categorical columns.
    Args:
        columns: Column names to summarize. If None, summarizes all columns.
    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """
    return csv_manager.summarize_columns(columns)


@tool
def describe_column(column: str) -> dict:
    """Describe a single column (basic stats) for the requested column.
    This includes count, mean, std, min, max, and percentiles for numeric column,
    or count, unique, top, freq for categorical column.
    Args:
        column: The name of the column to describe.
    Returns:
        A dict of basic stats for the column, or an error dict.
    """
    return csv_manager.describe_column(column)


@tool
def plot_data(y: str, x: str = None, plot_type: str = "line") -> str:
    """Plot from the active CSV.
    Args:
        y: Column name to plot on the y-axis.
        x: Column name to plot on the x-axis. If None, use row index.
        plot_type: "line" or "scatter". Scatter requires x and y.
    Returns:
        A short success message string, or an error dict/string.
    """
    return csv_manager.plot_data(y=y, x=x, plot_type=plot_type)


@tool  # rewrap new smolagents tool using @tool decorator.
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation between two numeric columns in the loaded CSV.
    Args:
        col1: Name of the first column.
        col2: Name of the second column.
    Returns:
        A dict with keys col1, col2, pearson_r, and p_value, or an error key if
        the columns are not found or no data is loaded.
    """
    return csv_manager.compute_correlation(col1, col2)


print(compute_correlation.description)

# The JSON schema in Q4 required manually writing out a nested dict with "type", "function", "name", "description", and a "parameters" object containing each argument's type and description. In comparison, smolagents reads the function's type annotations and Google-style docstring to produce the description part automatically; the Args section maps directly to parameter descriptions, and the return type annotation sets the output type.
# For smolagents to produce a good description (and avoid misused/skipped tools), the developer needs to provide accurate type annotations (arguments and return values), a first-line docstring explaining what the tool does, and a clear Args section describing parameters in plain English.

# ---------------------------------------------------------------------------- #
print("\n====== Q8")  # Q8

model = OpenAIServerModel(
    api_key=os.environ["OPENAI_API_KEY"],
    model_id="gpt-4o",
)
SYSTEM_PROMPT = (
    "You are a small data assistant to help analyze files stored in resources/. "
    "Use the available tools to do any work requested (do not guess). "
    "Keep answers short and student-friendly."
)
CODE_INSTRUCTIONS = """
You are a helpful CSV analysis assistant.

You can do two kinds of actions:
1) Call the provided tools.
2) Write and execute Python code when tools are not enough.

Rules:
- Prefer tools for simple tasks.
- IMPORTANT: If the user requests plot styling (color, marker, title text, labels, grid, etc.)
    that the plot_data tool cannot control, DO NOT call plot_data.
    Instead, write matplotlib code directly so the plot matches the request.
    If code execution fails, do not fall back to plot_data when the user requested styling (like color).
    Explain what failed and what you would need to proceed.
- Be honest: only claim you did something if the code or tool actually did it.
- Assume the active dataset lives in csv_manager.df after a CSV is loaded.
"""
TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    summarize_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    instructions=SYSTEM_PROMPT,
)
code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=CODE_INSTRUCTIONS,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "numpy"],
    max_steps=8,
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."
response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})
print("ToolCallingAgent response:", response_tool)
print("CodeAgent response:", response_code)

# ToolCallingAgent first called load_csv, and then called plot_data with the correct x and y columns. The scatter plot was produced, but with the default blue dot color since plot_data has no color parameter. Next, it called final_answer and acknowledged it could not set the color. The ToolCallingAgent is limited to the tools it has, so the request to use green dots wasn't fulfilled because the tool doesn't support color styling.

# CodeAgent wrote matplotlib code with color='green', loaded the CSV, accessed csv_manager.df directly, and produced the scatter plot with green dots. However, the agent tried to respond in plain text instead of calling final_answer, which triggering repeated code-parsing errors ("regex pattern not found") for steps 2-8. The core task was already complete, but the agent wasted tokens because it was unable to cleanly wrap up.

# ToolCallingAgent is better when the task maps to existing tools, when and you want controlled, auditable behavior. That is, it fails predictably when a tool can't do something. On the other hand, CodeAgent is more suited for tasks that require custom styling or logic beyond the tools, but it can waste steps on errors after completing the core work, and it needs careful prompting to call final_answer cleanly when done. The tradeoff is capability versus reliability.

# ---------------------------------------------------------------------------- #
print("\n====== Q9")  # Q9

# A task where ToolCallingAgent is a better choice (over CodeAgent) is calling an API to retrieve weather and news data, and then presenting them to the user. The key property making it a good fit is interaction with external systems via defined interfaces; APIs provide a well-defined function signature and returns predictable JSON or structured output. ToolCallingAgent excels at sequentially invoking tools, ensuring correct parameter formatting, and handling errors per tool.

# One meaningful risk of CodeAgent (that does not apply to ToolCallingAgent) is its over-reliance on correct parsing and formatting. It must generate and execute its own code to interact with APIs, meaning it risks introducing subtle bugs (such as incorrect URl encoding or missing headers), broken code if the API schema changes, manual debugging when requests fail, and incorrect or unintended code being be run (rather than merely being returned as text).
