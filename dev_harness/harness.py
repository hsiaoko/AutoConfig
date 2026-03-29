"""
AutoConfig Development Harness - Main Orchestrator

This module coordinates multiple agents to automate development tasks.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel

from .agents import BaseAgent, DevAgent, TestAgent, DocAgent, ReviewAgent
from .tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirTool,
    SearchCodeTool,
    RunTestsTool,
    CheckSyntaxTool,
)


logger = logging.getLogger(__name__)


class DevelopmentHarness:
    """
    Central orchestrator for the development agent system.
    
    The harness manages:
    - Agent initialization and lifecycle
    - Task distribution to appropriate agents
    - Multi-agent pipeline execution
    - Result aggregation and reporting
    """
    
    AGENT_CLASSES: Dict[str, Type[BaseAgent]] = {
        "dev": DevAgent,
        "test": TestAgent,
        "doc": DocAgent,
        "review": ReviewAgent,
    }
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        config_path: Optional[Path] = None,
        llm: Optional[BaseLanguageModel] = None,
        verbose: bool = False,
    ):
        """
        Initialize the development harness.
        
        Args:
            project_root: Root directory of the project
            config_path: Path to configuration file
            llm: Language model instance (optional, will create from config if None)
            verbose: Enable verbose logging
        """
        self.project_root = project_root or Path.cwd()
        self.config_path = config_path or (Path(__file__).parent / "config.yaml")
        self.config = self._load_config()
        self.verbose = verbose or self.config.get("agents", {}).get("verbose", False)
        
        # Set up logging
        self._setup_logging()
        
        # Initialize LLM
        self.llm = llm or self._create_llm()
        
        # Initialize tools
        self.tools = self._create_tools()
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {}
        self._init_agents()
        
        logger.info(f"Development Harness initialized for {self.project_root}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            return {}
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.config.get("output", {}).get("log_dir", ".harness-logs")
        log_path = self.project_root / log_dir
        
        if not log_path.exists():
            log_path.mkdir(parents=True, exist_ok=True)
        
        log_file = log_path / f"harness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.DEBUG if self.verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
    
    def _create_llm(self) -> BaseLanguageModel:
        """Create language model from configuration."""
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "openai")
        model = llm_config.get("model", "gpt-4o-mini")
        temperature = llm_config.get("temperature", 0.2)
        max_tokens = llm_config.get("max_tokens", 4096)
        
        # Check for API key in config or environment
        api_keys = self.config.get("api_keys", {})
        
        if provider == "openai":
            from langchain_openai import ChatOpenAI
            
            api_key = api_keys.get("openai") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        
        elif provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            
            api_key = api_keys.get("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")
            
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _create_tools(self) -> List:
        """Create the set of tools available to all agents."""
        return [
            ReadFileTool(project_root=self.project_root),
            WriteFileTool(project_root=self.project_root),
            ListDirTool(project_root=self.project_root),
            SearchCodeTool(project_root=self.project_root),
            RunTestsTool(project_root=self.project_root),
            CheckSyntaxTool(project_root=self.project_root),
        ]
    
    def _init_agents(self):
        """Initialize all enabled agents."""
        agent_config = self.config.get("agents", {})
        
        for agent_name, agent_class in self.AGENT_CLASSES.items():
            enabled = agent_config.get(agent_name, True)
            if not enabled:
                logger.info(f"Agent {agent_name} is disabled")
                continue
            
            agent = agent_class(
                llm=self.llm,
                tools=self.tools,
                project_root=self.project_root,
                verbose=self.verbose,
            )
            self.agents[agent_name] = agent
            logger.info(f"Initialized agent: {agent_name}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(name)
    
    def list_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.agents.keys())
    
    def run_task(
        self,
        agent_name: str,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single task on a specific agent.
        
        Args:
            agent_name: Name of the agent to use
            task: Task description
            context: Optional context for the task
            
        Returns:
            Task result dictionary
        """
        if agent_name not in self.agents:
            return {
                "success": False,
                "error": f"Unknown agent: {agent_name}",
            }
        
        agent = self.agents[agent_name]
        logger.info(f"Running task on {agent_name}: {task[:100]}...")
        
        result = agent.execute(task, context)
        
        logger.info(f"Task completed: success={result.get('success', False)}")
        
        return result
    
    def run_pipeline(
        self,
        task: str,
        workflow: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a multi-agent pipeline.
        
        Args:
            task: Overall task description
            workflow: List of agent names to run in sequence
            context: Context passed between agents
            
        Returns:
            Pipeline result with all agent outputs
        """
        if workflow is None:
            workflow = self.config.get("pipeline", {}).get(
                "default_workflow",
                ["dev", "test", "review", "doc"]
            )
        
        results = {
            "task": task,
            "workflow": workflow,
            "agent_results": {},
            "success": True,
        }
        
        current_context = context or {}
        
        for agent_name in workflow:
            if agent_name not in self.agents:
                logger.warning(f"Skipping unknown agent: {agent_name}")
                continue
            
            # Build task description with context from previous agents
            agent_task = f"{task}\n\nContext from previous steps:\n{current_context}"
            
            logger.info(f"Running agent: {agent_name}")
            result = self.agents[agent_name].execute(agent_task, current_context)
            
            results["agent_results"][agent_name] = result
            
            if not result.get("success", False):
                results["success"] = False
                logger.error(f"Agent {agent_name} failed: {result.get('error', 'Unknown error')}")
                # Continue pipeline even if one agent fails
            
            # Update context with this agent's output
            current_context[f"{agent_name}_output"] = result.get("output", "")
            current_context[f"{agent_name}_success"] = result.get("success", False)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the harness."""
        return {
            "project_root": str(self.project_root),
            "agents": list(self.agents.keys()),
            "llm_model": self.config.get("llm", {}).get("model", "unknown"),
            "verbose": self.verbose,
        }
