from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from operator import add

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add]


def chat(state: State):

    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("chat_node", chat)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


checkpoint = MemorySaver()
workspace = graph.compile(checkpointer=checkpoint)


while True:
    input_query = input("user: ")
    if input_query == "exit":
        break
    else:
        config = {
            'configurable':{
                'thread_id':"firstThread"
        }}
        response = workspace.invoke({"messages": [HumanMessage(content=input_query)]}, config=config)
        print(f"AI: {response['messages'][-1].content}")