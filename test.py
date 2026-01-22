import time
from app import get_tax_assistant
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage
)


assistant = get_tax_assistant()

question = "Are taxes imposed on farmers"
user_id = "debug-90"

print("\n" + "=" * 80)
print("QUESTION:")
print(question)
print("=" * 80 + "\n")

start_time = time.time()

# Compile graph once (important for performance)
agent = assistant.builder.compile(checkpointer=assistant.checkpointer)

# Invoke with full state returned
result = agent.invoke(
    {"messages": [HumanMessage(content=question)]},
    {"configurable": {"thread_id": user_id}}
)

elapsed = time.time() - start_time

messages = result["messages"]

tool_calls_count = 0
retrieved_chars = 0

for idx, message in enumerate(messages, start=1):
    print(f"\n--- STEP {idx} ({type(message).__name__}) ---")

    if isinstance(message, HumanMessage):
        print("User:")
        print(message.content)

    elif isinstance(message, AIMessage):
        if message.tool_calls:
            tool_calls_count += len(message.tool_calls)
            for tc in message.tool_calls:
                print(f"🔧 Tool Call → {tc['name']}")
                print(f"Arguments: {tc['args']}")
        else:
            print("Assistant:")
            print(message.content)

    elif isinstance(message, ToolMessage):
        content = message.content or ""
        retrieved_chars += len(content)
        print(f"📄 Tool Result ({len(content)} chars):")
        print(content[:500] + ("..." if len(content) > 500 else ""))

print("\n" + "=" * 80)
print("📊 PERFORMANCE SUMMARY")
print("=" * 80)
print(f"Tool calls made      : {tool_calls_count}")
print(f"Total retrieved text : {retrieved_chars} chars")
print(f"Total latency        : {elapsed:.2f}s")

print("\n" + "=" * 80)
print("FINAL ANSWER")
print("=" * 80)
print(messages[-1].content)



