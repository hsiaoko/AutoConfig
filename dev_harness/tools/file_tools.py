"""File operation tools for development agents."""

import os
from pathlib import Path
from typing import Optional, List
from langchain.tools import BaseTool


class ReadFileTool(BaseTool):
    """Read content from a file."""
    
    name: str = "read_file"
    description: str = "Read the content of a file. Input should be the file path relative to project root."
    project_root: Optional[Path] = None
    
    def _run(self, file_path: str) -> str:
        """Read file content."""
        root = self.project_root or Path.cwd()
        full_path = root / file_path
        
        if not full_path.exists():
            return f"Error: File not found: {full_path}"
        
        try:
            content = full_path.read_text(encoding="utf-8")
            return f"Content of {file_path}:\n{content}"
        except Exception as e:
            return f"Error reading file: {e}"
    
    async def _arun(self, file_path: str) -> str:
        """Async read - not implemented, falls back to sync."""
        return self._run(file_path)


class WriteFileTool(BaseTool):
    """Write content to a file."""
    
    name: str = "write_file"
    description: str = "Write content to a file. Input should be JSON with 'file_path' and 'content' keys."
    project_root: Optional[Path] = None
    
    def _run(self, file_path: str, content: str) -> str:
        """Write content to file."""
        root = self.project_root or Path.cwd()
        full_path = root / file_path
        
        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            return f"Error writing file: {e}"
    
    async def _arun(self, file_path: str, content: str) -> str:
        """Async write - not implemented, falls back to sync."""
        return self._run(file_path, content)


class ListDirTool(BaseTool):
    """List contents of a directory."""
    
    name: str = "list_directory"
    description: str = "List contents of a directory. Input should be the directory path relative to project root."
    project_root: Optional[Path] = None
    
    def _run(self, dir_path: str = ".") -> str:
        """List directory contents."""
        root = self.project_root or Path.cwd()
        full_path = root / dir_path
        
        if not full_path.exists():
            return f"Error: Directory not found: {full_path}"
        
        if not full_path.is_dir():
            return f"Error: Not a directory: {full_path}"
        
        items = []
        for item in full_path.iterdir():
            item_type = "[DIR]" if item.is_dir() else "[FILE]"
            items.append(f"{item_type} {item.name}")
        
        return f"Contents of {dir_path}:\n" + "\n".join(sorted(items))
    
    async def _arun(self, dir_path: str = ".") -> str:
        """Async list - not implemented, falls back to sync."""
        return self._run(dir_path)


class SearchCodeTool(BaseTool):
    """Search for patterns in code files."""
    
    name: str = "search_code"
    description: str = "Search for a pattern in code files. Input should be the search pattern."
    project_root: Optional[Path] = None
    max_results: int = 20
    
    def _run(self, pattern: str, file_glob: str = "*.py") -> str:
        """Search for pattern in code files."""
        import glob
        import re
        
        root = self.project_root or Path.cwd()
        results = []
        
        # Find all matching files
        pattern_re = re.compile(pattern, re.IGNORECASE)
        
        for file_path in root.rglob(file_glob):
            # Skip common non-code directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if any(part in ['venv', '__pycache__', 'node_modules', '.git'] for part in file_path.parts):
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8")
                matches = []
                for i, line in enumerate(content.split('\n'), 1):
                    if pattern_re.search(line):
                        matches.append(f"  Line {i}: {line.strip()}")
                
                if matches:
                    rel_path = file_path.relative_to(root)
                    results.append(f"\n{rel_path}:")
                    results.extend(matches[:self.max_results])
            except Exception:
                continue
        
        if not results:
            return f"No matches found for pattern: {pattern}"
        
        return f"Found matches for '{pattern}':" + "\n".join(results)
    
    async def _arun(self, pattern: str, file_glob: str = "*.py") -> str:
        """Async search - not implemented, falls back to sync."""
        return self._run(pattern, file_glob)
