from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class DhruvState(TypedDict):
    # add_messages acts as a reducer, appending new messages to the existing list
    messages: Annotated[list, add_messages]
    # Holds the latest description from the remote multimodal model
    current_scene: str