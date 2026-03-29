#!/usr/bin/env python3
"""
AutoConfig Development Harness CLI

Command-line interface for the development agent system.

Usage:
    python -m dev_harness run --task "Add feature X"
    python -m dev_harness run --agent test --target file.py
    python -m dev_harness pipeline --task "Implement feature Y"
    python -m dev_harness status
"""

import argparse
import sys
from pathlib import Path

from .harness import DevelopmentHarness


def cmd_run(args):
    """Run a single agent task."""
    harness = DevelopmentHarness(
        project_root=Path(args.project),
        verbose=args.verbose,
    )
    
    if args.agent:
        # Run specific agent
        result = harness.run_task(
            agent_name=args.agent,
            task=args.task,
            context={"target": args.target} if args.target else None,
        )
    else:
        # Run default agent (dev)
        result = harness.run_task(
            agent_name="dev",
            task=args.task,
            context={"target": args.target} if args.target else None,
        )
    
    _print_result(result)
    return 0 if result.get("success") else 1


def cmd_pipeline(args):
    """Run a multi-agent pipeline."""
    harness = DevelopmentHarness(
        project_root=Path(args.project),
        verbose=args.verbose,
    )
    
    workflow = None
    if args.workflow:
        workflow = args.workflow.split(",")
    
    result = harness.run_pipeline(
        task=args.task,
        workflow=workflow,
    )
    
    _print_result(result)
    return 0 if result.get("success") else 1


def cmd_status(args):
    """Show harness status."""
    harness = DevelopmentHarness(
        project_root=Path(args.project),
        verbose=args.verbose,
    )
    
    status = harness.get_status()
    
    print("\n=== Development Harness Status ===")
    print(f"Project Root: {status['project_root']}")
    print(f"LLM Model: {status['llm_model']}")
    print(f"Available Agents: {', '.join(status['agents'])}")
    print(f"Verbose: {status['verbose']}")
    print()
    
    return 0


def cmd_list_agents(args):
    """List available agents."""
    harness = DevelopmentHarness(
        project_root=Path(args.project),
        verbose=args.verbose,
    )
    
    print("\nAvailable Agents:")
    for agent_name in harness.list_agents():
        print(f"  - {agent_name}")
    print()
    
    return 0


def _print_result(result: dict):
    """Print task result in a formatted way."""
    print("\n" + "=" * 60)
    
    if result.get("success"):
        print("✓ Task Completed Successfully")
    else:
        print("✗ Task Failed")
    
    print(f"Agent: {result.get('agent', 'Unknown')}")
    print(f"Task: {result.get('task', 'N/A')}")
    
    if result.get("error"):
        print(f"\nError: {result['error']}")
    
    if result.get("output"):
        print("\n--- Output ---")
        print(result["output"])
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AutoConfig Development Harness - Multi-Agent Development System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a development task
  python -m dev_harness run --task "Add a new function to calculate graph diameter"
  
  # Run test agent on a specific file
  python -m dev_harness run --agent test --target autoconfig/utils/config_generator.py
  
  # Run full pipeline
  python -m dev_harness pipeline --task "Implement GPU resource support"
  
  # Run custom workflow
  python -m dev_harness pipeline --task "Fix bug in query extractor" --workflow dev,test,review
  
  # Check status
  python -m dev_harness status
        """
    )
    
    parser.add_argument(
        "--project", "-p",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a single agent task")
    run_parser.add_argument(
        "--agent", "-a",
        choices=["dev", "test", "doc", "review"],
        help="Agent to use (default: dev)",
    )
    run_parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task description",
    )
    run_parser.add_argument(
        "--target",
        help="Target file or directory",
    )
    run_parser.set_defaults(func=cmd_run)
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run a multi-agent pipeline",
    )
    pipeline_parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task description",
    )
    pipeline_parser.add_argument(
        "--workflow", "-w",
        help="Comma-separated list of agents (default: dev,test,review,doc)",
    )
    pipeline_parser.set_defaults(func=cmd_pipeline)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show harness status")
    status_parser.set_defaults(func=cmd_status)
    
    # List agents command
    list_parser = subparsers.add_parser("list-agents", help="List available agents")
    list_parser.set_defaults(func=cmd_list_agents)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
