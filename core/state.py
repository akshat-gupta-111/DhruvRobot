from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class DhruvState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    raw_image_b64: Optional[str]   # Passed from Jetson; only processed if needed
    current_scene: str             # Populated only if intent requires perception
    intent: str                    # "movement", "perception", or "hybrid"