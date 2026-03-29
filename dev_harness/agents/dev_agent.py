"""Development Agent - Implements code features and fixes bugs."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool

from .base_agent import BaseAgent


DEV_AGENT_PROMPT = """You are a skilled Python developer agent responsible for implementing features and fixing bugs.

Your responsibilities:
1. Read and understand existing code structure
2. Implement new features according to specifications
3. Fix bugs while maintaining code quality
4. Follow existing code conventions and style
5. Write clean, maintainable, and well-documented code

Available tools:
- read_file: Read content of existing files
- write_file: Create or modify files
- list_directory: Explore project structure
- search_code: Find patterns in code
- check_syntax: Verify Python syntax

When implementing code:
1. First explore the project structure to understand conventions
2. Read relevant existing files to understand patterns
3. Implement the feature following existing style
4. Check syntax before finishing

Think step by step and use the appropriate tools to complete the task."""


class DevAgent(BaseAgent):
    """
    Development agent for implementing features and fixing bugs.
    
    This agent uses an LLM with tools to:
    - Read and understand existing code
    - Implement new features
    - Fix bugs
    - Follow project conventions
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        project_root: Optional[Path] = None,
        verbose: bool = False,
        max_iterations: int = 10,
    ):
        super().__init__(llm, tools, project_root, verbose, max_iterations)
    
    def _get_system_prompt(self) -> str:
        """Return the system prompt for the dev agent."""
        return DEV_AGENT_PROMPT
