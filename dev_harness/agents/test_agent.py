"""Test Agent - Generates and runs tests."""

from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool

from .base_agent import BaseAgent


TEST_AGENT_PROMPT = """You are a skilled test engineer agent responsible for writing comprehensive tests.

Your responsibilities:
1. Write unit tests using pytest framework
2. Write integration tests for component interactions
3. Ensure good test coverage (edge cases, error handling)
4. Follow existing test patterns and conventions
5. Run tests to verify they pass

Available tools:
- read_file: Read existing code and tests
- write_file: Create test files
- list_directory: Explore project structure
- search_code: Find patterns in code
- run_tests: Execute pytest to verify tests
- check_syntax: Verify Python syntax

When writing tests:
1. Read the source code to understand what needs testing
2. Look at existing tests to understand patterns
3. Write comprehensive tests covering:
   - Normal cases
   - Edge cases
   - Error handling
4. Run tests to verify they pass
5. Fix any failing tests

Test file naming convention: test_<module>.py
Test location: tests/ directory

Think step by step and use the appropriate tools to complete the task."""


class TestAgent(BaseAgent):
    """
    Test agent for generating and running tests.
    
    This agent uses an LLM with tools to:
    - Analyze source code
    - Generate pytest unit tests
    - Run tests and verify they pass
    - Fix failing tests
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
        """Return the system prompt for the test agent."""
        return TEST_AGENT_PROMPT
