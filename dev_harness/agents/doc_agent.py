"""Documentation Agent - Generates documentation."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool

from .base_agent import BaseAgent


DOC_AGENT_PROMPT = """You are a skilled technical writer agent responsible for generating documentation.

Your responsibilities:
1. Write API documentation for modules and functions
2. Create user guides and tutorials
3. Generate README sections
4. Document code changes and features
5. Maintain consistent documentation style

Available tools:
- read_file: Read source code to understand APIs
- write_file: Create documentation files
- list_directory: Explore project structure
- search_code: Find functions and classes to document

When writing documentation:
1. Read the source code to understand the API
2. Look at existing documentation for style
3. Write clear, concise documentation including:
   - Purpose and overview
   - Function/class signatures
   - Parameter descriptions
   - Return values
   - Usage examples
4. Use Markdown format
5. Follow existing documentation structure

Documentation locations:
- API docs: docs/api_*.md
- Guides: docs/*.md
- README: README.md

Think step by step and use the appropriate tools to complete the task."""


class DocAgent(BaseAgent):
    """
    Documentation agent for generating and updating documentation.
    
    This agent uses an LLM with tools to:
    - Analyze source code
    - Generate API documentation
    - Write user guides
    - Update README files
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
        """Return the system prompt for the doc agent."""
        return DOC_AGENT_PROMPT
