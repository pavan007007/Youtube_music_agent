import os
import json
import sys

from openai import OpenAI
from dotenv import load_dotenv

from youtube_tool import search_youtube, play_youtube


# Fix Windows terminal Unicode problems
sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


# --------------------------------------------------
# NVIDIA LLM
# --------------------------------------------------

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY is missing from .env")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)


# --------------------------------------------------
# TOOLS
# --------------------------------------------------

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_youtube",
            "description": (
                "Search YouTube for a song, artist, music video, "
                "or other music requested by the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The exact song, artist, or music search query."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "play_youtube",
            "description": (
                "Open a selected YouTube video in the user's browser. "
                "Use this after searching YouTube and selecting the "
                "most relevant result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The YouTube video URL."
                    }
                },
                "required": ["url"]
            }
        }
    }
]


# --------------------------------------------------
# TOOL EXECUTOR
# --------------------------------------------------

def execute_tool(tool_name, arguments):

    if tool_name == "search_youtube":

        return search_youtube(
            arguments["query"]
        )

    elif tool_name == "play_youtube":

        return play_youtube(
            arguments["url"]
        )

    return {
        "success": False,
        "error": f"Unknown tool: {tool_name}"
    }


# --------------------------------------------------
# AGENT
# --------------------------------------------------

def run_agent(user_request):

    messages = [
        {
            "role": "system",
            "content": """
You are a YouTube music assistant.

Your job is to understand what song the user wants to play.

Follow this workflow:

1. Understand the user's music request.
2. Call search_youtube exactly once to find the song.
3. Examine the returned search results.
4. Select the single most relevant result.
5. Call play_youtube with that video's URL.
6. After play_youtube succeeds, STOP.
7. Give the user a short confirmation.

IMPORTANT RULES:

- Do not repeatedly search for the same song.
- Do not call search_youtube more than once for a request.
- Do not call play_youtube more than once.
- Once play_youtube succeeds, the task is complete.
- Do not continue calling tools after the song has been opened.
"""
        },
        {
            "role": "user",
            "content": user_request
        }
    ]

    # Hard safety limit
    MAX_ITERATIONS = 5

    # Track previous tool calls
    tool_history = set()

    # Prevent multiple YouTube searches
    youtube_search_used = False

    for iteration in range(MAX_ITERATIONS):

        print(
            f"\n[Agent step {iteration + 1}/{MAX_ITERATIONS}]"
        )

        try:

            response = client.chat.completions.create(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0
            )

        except Exception as e:

            print("\nLLM error:")
            print(e)
            return

        message = response.choices[0].message

        # --------------------------------------------------
        # NO TOOL CALL = FINAL ANSWER
        # --------------------------------------------------

        if not message.tool_calls:

            print("\nAgent:", message.content)
            return

        # Add assistant message to conversation
        messages.append(message)

        # --------------------------------------------------
        # PROCESS TOOL CALLS
        # --------------------------------------------------

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            try:
                arguments = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError:

                print("\nInvalid tool arguments.")
                print("Stopping agent.")
                return

            # --------------------------------------------------
            # REPEATED TOOL CALL PROTECTION
            # --------------------------------------------------

            call_signature = (
                tool_name,
                json.dumps(
                    arguments,
                    sort_keys=True
                )
            )

            if call_signature in tool_history:

                print("\nRepeated tool call detected.")
                print("Stopping agent to prevent an infinite loop.")
                return

            tool_history.add(call_signature)

            # --------------------------------------------------
            # SEARCH LIMIT
            # --------------------------------------------------

            if tool_name == "search_youtube":

                if youtube_search_used:

                    print(
                        "\nYouTube search already used."
                    )

                    print(
                        "Stopping agent."
                    )

                    return

                youtube_search_used = True

            # --------------------------------------------------
            # DISPLAY TOOL CALL
            # --------------------------------------------------

            print(
                f"\nTool: {tool_name}"
            )

            print(
                f"Arguments: {arguments}"
            )

            # --------------------------------------------------
            # EXECUTE TOOL
            # --------------------------------------------------

            result = execute_tool(
                tool_name,
                arguments
            )

            # --------------------------------------------------
            # PLAY SUCCESS = STOP IMMEDIATELY
            # --------------------------------------------------

            if tool_name == "play_youtube":

                if result.get("success"):

                    print(
                        "\nSong opened successfully."
                    )

                    print(
                        "Agent finished."
                    )

                    return

                else:

                    print(
                        "\nCould not open YouTube."
                    )

                    print(
                        result
                    )

                    return

            # --------------------------------------------------
            # SEND TOOL RESULT BACK TO LLM
            # --------------------------------------------------

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    # --------------------------------------------------
    # MAX ITERATIONS REACHED
    # --------------------------------------------------

    print(
        "\nMaximum iterations reached."
    )

    print(
        "Agent stopped to prevent an infinite loop."
    )


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    print("YouTube AI Music Agent")
    print("Type 'exit' to quit.")

    while True:

        try:

            user_input = input("\nYou: ").strip()

        except KeyboardInterrupt:

            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in [
            "exit",
            "quit"
        ]:

            print("Goodbye!")
            break

        run_agent(user_input)