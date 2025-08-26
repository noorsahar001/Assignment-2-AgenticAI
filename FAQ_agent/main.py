import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv

set_tracing_disabled(True)

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

#
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file!")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

Config = RunConfig(
    model=model,
    tracing_disabled=True,
)

def main():
    print("Welcome to the FAQ Agent!")
    print("This agent is designed to answer questions about FAQ. \n")
    print("You can ask:\n- What is your name?\n- What can you do?\n- Who made you?\n- How can I use you?\n- What technologies are you using?\n(Type 'exit' to quit)\n")
    
    # Create agent
    agent = Agent(
        name="FAQ Agent",
        instructions="""
You are a helpful FAQ bot.
1. What is your name?
2. What can you do?
3. Who made you?
4. How can I use you?
5. What technologies are you using?

If user asks anything else, say: 'Sorry, I can only answer predefined questions.'
""",
        model=model,
    )

    # User input loop
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit"]:
            print("Bye bye!")
            break

        # Run agent sync
        result = Runner.run_sync(agent, user_question, run_config=Config)

        # Output response
        print("FAQ Agent:", result.final_output)

if __name__ == "__main__":
    main()
