import time
from collections import deque
from typing import List, Tuple

class VisionHistory:
    def __init__(self, max_seconds=5):
        """
        Maintains a rolling buffer of visual observations over a specified time window.
        """
        self.max_seconds = max_seconds
        # Stores tuples of (timestamp, scene_description)
        self.buffer: deque[Tuple[float, str]] = deque()
        
    def add(self, scene_description: str):
        now = time.time()
        self.buffer.append((now, scene_description))
        self._cleanup()
        
    def get_recent_history(self) -> str:
        """Returns all scene descriptions stored within the time window as a combined string."""
        self._cleanup()
        if not self.buffer:
            return "No recent visual history available."
        
        history_lines = []
        for i, (_, desc) in enumerate(self.buffer):
            history_lines.append(f"[Observation {i+1}]: {desc}")
            
        return "\n".join(history_lines)
        
    def _cleanup(self):
        now = time.time()
        # Remove items older than max_seconds
        while self.buffer and now - self.buffer[0][0] > self.max_seconds:
            self.buffer.popleft()
