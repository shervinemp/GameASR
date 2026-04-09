import sys
from unittest.mock import MagicMock
sys.modules['pynput'] = MagicMock()
sys.modules['pynput.keyboard'] = MagicMock()

from voice_control.pipeline import Pipeline
from voice_control.rag.model import SPathRAG

try:
    print("Testing pipeline initialization...")
    pipe = Pipeline()
    print("Done pipeline initialization.")
except Exception as e:
    import traceback
    traceback.print_exc()
