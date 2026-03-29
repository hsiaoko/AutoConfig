"""Code operation tools for development agents."""

import subprocess
from pathlib import Path
from typing import Optional
from langchain.tools import BaseTool


class RunTestsTool(BaseTool):
    """Run tests using pytest."""
    
    name: str = "run_tests"
    description: str = "Run pytest tests. Input should be the test file or directory path relative to project root."
    project_root: Optional[Path] = None
    
    def _run(self, test_path: str = "tests", verbose: bool = False) -> str:
        """Run pytest on the specified path."""
        root = self.project_root or Path.cwd()
        full_path = root / test_path
        
        if not full_path.exists():
            return f"Error: Test path not found: {full_path}"
        
        cmd = ["python", "-m", "pytest", str(full_path), "-v"]
        if not verbose:
            cmd.append("--tb=short")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            output = []
            output.append(f"Command: {' '.join(cmd)}")
            output.append(f"Exit code: {result.returncode}")
            output.append("\n--- STDOUT ---")
            output.append(result.stdout)
            
            if result.stderr:
                output.append("\n--- STDERR ---")
                output.append(result.stderr)
            
            return "\n".join(output)
        except subprocess.TimeoutExpired:
            return "Error: Test execution timed out (120s)"
        except Exception as e:
            return f"Error running tests: {e}"
    
    async def _arun(self, test_path: str = "tests", verbose: bool = False) -> str:
        """Async run tests - not implemented, falls back to sync."""
        return self._run(test_path, verbose)


class CheckSyntaxTool(BaseTool):
    """Check Python syntax of a file."""
    
    name: str = "check_syntax"
    description: str = "Check Python syntax of a file. Input should be the file path relative to project root."
    project_root: Optional[Path] = None
    
    def _run(self, file_path: str) -> str:
        """Check Python syntax using py_compile."""
        import py_compile
        
        root = self.project_root or Path.cwd()
        full_path = root / file_path
        
        if not full_path.exists():
            return f"Error: File not found: {full_path}"
        
        if not file_path.endswith('.py'):
            return "Error: Only Python files can be syntax checked"
        
        try:
            py_compile.compile(str(full_path), doraise=True)
            return f"Syntax OK: {file_path}"
        except py_compile.PyCompileError as e:
            return f"Syntax Error in {file_path}:\n{e}"
        except Exception as e:
            return f"Error checking syntax: {e}"
    
    async def _arun(self, file_path: str) -> str:
        """Async syntax check - not implemented, falls back to sync."""
        return self._run(file_path)
