"""Base Agent class for the development harness."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage


class BaseAgent(ABC):
    """
    Abstract base class for all development agents.
    
    Each agent is responsible for a specific aspect of the development workflow:
    - DevAgent: Code implementation
    - TestAgent: Test generation
    - DocAgent: Documentation generation
    - ReviewAgent: Code review
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: List[BaseTool],
        project_root: Optional[Path] = None,
        verbose: bool = False,
        max_iterations: int = 10,
    ):
        """
        Initialize the base agent.
        
        Args:
            llm: Language model to use for reasoning
            tools: List of tools available to the agent
            project_root: Root directory of the project
            verbose: Whether to print verbose output
            max_iterations: Maximum iterations for tool use loop
        """
        self.llm = llm
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.project_root = project_root or Path.cwd()
        self.verbose = verbose
        self.max_iterations = max_iterations
        
        # Bind tools to LLM
        self.llm_with_tools = llm.bind_tools(tools)
    
    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return its output."""
        if tool_name not in self.tools_by_name:
            return f"Error: Unknown tool '{tool_name}'"
        
        tool = self.tools_by_name[tool_name]
        try:
            # Parse tool input as JSON if possible
            import json
            try:
                kwargs = json.loads(tool_input)
                if isinstance(kwargs, dict):
                    return tool.invoke(**kwargs)
                else:
                    return tool.invoke(tool_input)
            except json.JSONDecodeError:
                return tool.invoke(tool_input)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"
    
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the agent's task using a tool-calling loop.
        
        Args:
            task: Description of the task to perform
            context: Optional context information
            
        Returns:
            Dictionary containing the result and any metadata
        """
        # Build the prompt
        system_prompt = self._get_system_prompt()
        user_message = self._build_user_message(task, context)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        
        # Tool-calling loop
        intermediate_steps = []
        final_output = ""
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[Agent] Iteration {iteration + 1}/{self.max_iterations}")
            
            # Get LLM response
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # Check if there are tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_input = tool_call.get('arguments', {})
                    
                    if self.verbose:
                        print(f"  Calling tool: {tool_name}")
                    
                    # Execute tool
                    tool_output = self._run_tool(tool_name, str(tool_input))
                    
                    messages.append(ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call.get('id', ''),
                    ))
                    
                    intermediate_steps.append({
                        'tool': tool_name,
                        'input': tool_input,
                        'output': tool_output,
                    })
            else:
                # No more tool calls, return final response
                final_output = response.content
                if self.verbose:
                    print(f"  Final response received")
                break
        else:
            final_output = "Max iterations reached without completing the task."
        
        return {
            "success": True,
            "agent": self.__class__.__name__,
            "task": task,
            "output": final_output,
            "intermediate_steps": intermediate_steps,
            "iterations": len(intermediate_steps),
        }
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    def _build_user_message(self, task: str, context: Optional[Dict[str, Any]]) -> str:
        """Build the user message from task and context."""
        context_str = str(context) if context else "No additional context provided."
        
        return f"""Project root: {self.project_root}

Task: {task}

Context:
{context_str}

Please complete this task step by step using the available tools."""
    
    def _read_file(self, file_path: Path) -> str:
        """Read a file's content."""
        full_path = self.project_root / file_path
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return full_path.read_text(encoding="utf-8")
    
    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to a file."""
        full_path = self.project_root / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
    
    def _list_directory(self, dir_path: Path) -> List[str]:
        """List contents of a directory."""
        full_path = self.project_root / dir_path
        if not full_path.exists():
            raise FileNotFoundError(f"Directory not found: {full_path}")
        return [str(p) for p in full_path.iterdir()]
    
    def get_agent_info(self) -> Dict[str, str]:
        """Return information about this agent."""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__.split("\n")[1].strip() if self.__doc__ else "",
            "project_root": str(self.project_root),
            "tools": [tool.name for tool in self.tools],
        }
