# Quick test script — save as scripts/test_nvidia_judge.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / "src" / ".env")

from llm_clients import get_client

judge = get_client("nemotron-3-super")
response = judge.generate(
    "Output ONLY a JSON object with a 'test' field set to 'ok'. No other text."
)
print(response)