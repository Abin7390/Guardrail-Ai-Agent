"""Numbered guardrail steps (01_prompt_guard, 02_masking, ...), in pipeline order.

Module names starting with a digit cannot appear in import statements, so each
numbered step file is loaded here and aliased to a valid dotted path, e.g.
``from guard.steps.prompt_guard import classify``.
"""

import importlib
import sys

file_intake = importlib.import_module("guard.steps.00_file_intake")
sys.modules["guard.steps.file_intake"] = file_intake
prompt_guard = importlib.import_module("guard.steps.01_prompt_guard")
sys.modules["guard.steps.prompt_guard"] = prompt_guard
masking = importlib.import_module("guard.steps.02_masking")
sys.modules["guard.steps.masking"] = masking
embedding = importlib.import_module("guard.steps.03_embedding")
sys.modules["guard.steps.embedding"] = embedding
llm_flagging = importlib.import_module("guard.steps.04_llm_flagging")
sys.modules["guard.steps.llm_flagging"] = llm_flagging
