#!/usr/bin/env python3
"""
Corbot Tools - Tool definitions and executor for Qwen agentic loop.
Gives Qwen autonomous access to filesystem, shell, and 99 Klaus skills.
"""

import os
import json
import subprocess
import importlib.util
import glob as glob_module
import re
from pathlib import Path
from typing import Any

SKILLS_DIR = Path.home() / ".axe" / "skills"
ALLOWED_DIRS = [str(Path.home())]  # Sandbox to home directory
DEFAULT_PROJECT_DIR = str(Path.home() / "Desktop" / "klaus-projects")

# ============================================================
# TOOL DEFINITIONS (Ollama native format for Qwen)
# ============================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the full text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to write to"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a specific string with new content. The old_text must match exactly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file"},
                    "old_text": {"type": "string", "description": "Exact text to find and replace"},
                    "new_text": {"type": "string", "description": "Replacement text"}
                },
                "required": ["path", "old_text", "new_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at the given path with sizes and types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list. Defaults to working directory."}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Execute a shell command and return stdout/stderr. 30 second timeout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_dir": {"type": "string", "description": "Working directory for the command. Optional."}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a glob pattern. Returns list of matching file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py', '*.tsx')"},
                    "path": {"type": "string", "description": "Directory to search in. Defaults to working directory."}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_content",
            "description": "Search file contents for a regex pattern (like grep). Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "regex": {"type": "string", "description": "Regular expression to search for"},
                    "path": {"type": "string", "description": "Directory or file to search in"},
                    "file_pattern": {"type": "string", "description": "Optional glob to filter files (e.g. '*.py')"}
                },
                "required": ["regex"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_skill",
            "description": "Run one of Klaus's 99 skills. Each skill is a Python module with a main() function. Pass skill_id (e.g. 'skill_32_web_search') and params as a JSON object. Check the skill catalog in the system prompt for available skills and their parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_id": {"type": "string", "description": "Skill identifier, e.g. 'skill_32_web_search'"},
                    "params": {"type": "object", "description": "Parameters to pass to the skill's main() function"}
                },
                "required": ["skill_id"]
            }
        }
    },
    # ── IDE-specific tools ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Scaffold a new React + Tailwind project from template. Creates package.json, vite config, tailwind config, index.html, App.tsx, and main.tsx. Use this when the user asks to build a new app or project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name (lowercase, no spaces). Will be created in ~/Desktop/klaus-projects/"},
                    "description": {"type": "string", "description": "Brief description of what the project does"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "install_packages",
            "description": "Install npm packages in a project directory. Runs npm install with the given packages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {"type": "string", "description": "Space-separated package names (e.g. 'axios lucide-react date-fns')"},
                    "project_dir": {"type": "string", "description": "Project directory path. Defaults to working directory."}
                },
                "required": ["packages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_dev_server",
            "description": "Start the Vite dev server for a project. Returns the port number. Use this after creating/modifying a project so the user can see it in the preview panel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_dir": {"type": "string", "description": "Project directory path. Defaults to working directory."}
                },
                "required": []
            }
        }
    },
]

# ============================================================
# TOOL EXECUTORS
# ============================================================

def _sanitize_path(path: str) -> str:
    """Resolve path and check it's within allowed directories."""
    resolved = str(Path(path).expanduser().resolve())
    if not any(resolved.startswith(d) for d in ALLOWED_DIRS):
        raise PermissionError(f"Access denied: {path} is outside allowed directories")
    return resolved


def exec_read_file(path: str, **kwargs) -> str:
    path = _sanitize_path(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, 'r', errors='replace') as f:
            content = f.read()
        # Truncate very large files
        if len(content) > 100000:
            return content[:100000] + f"\n\n[Truncated - file is {len(content)} chars total]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def exec_write_file(path: str, content: str, **kwargs) -> str:
    path = _sanitize_path(path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def exec_edit_file(path: str, old_text: str, new_text: str, **kwargs) -> str:
    path = _sanitize_path(path)
    if not os.path.isfile(path):
        return f"Error: File not found: {path}"
    try:
        with open(path, 'r') as f:
            content = f.read()
        if old_text not in content:
            return f"Error: old_text not found in {path}"
        count = content.count(old_text)
        content = content.replace(old_text, new_text, 1)
        with open(path, 'w') as f:
            f.write(content)
        return f"Replaced 1 occurrence in {path}" + (f" ({count-1} more occurrences remain)" if count > 1 else "")
    except Exception as e:
        return f"Error editing file: {e}"


def exec_list_directory(path: str = None, **kwargs) -> str:
    if not path or path == ".":
        path = kwargs.get("_working_dir", ".")
    path = _sanitize_path(path)
    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"
    try:
        entries = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                entries.append(f"  [DIR]  {name}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    sz = f"{size}B"
                elif size < 1048576:
                    sz = f"{size/1024:.1f}KB"
                else:
                    sz = f"{size/1048576:.1f}MB"
                entries.append(f"  {sz:>8}  {name}")
        return f"{path}/\n" + "\n".join(entries) if entries else f"{path}/ (empty)"
    except Exception as e:
        return f"Error listing directory: {e}"


def exec_run_command(command: str, working_dir: str = None, **kwargs) -> str:
    if not working_dir:
        working_dir = kwargs.get("_working_dir")
    if working_dir:
        working_dir = _sanitize_path(working_dir)
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=working_dir
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        if not output:
            output = "(no output)"
        # Truncate long output
        if len(output) > 50000:
            output = output[:50000] + "\n[Truncated]"
        return f"Exit code: {result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out (30s limit)"
    except Exception as e:
        return f"Error running command: {e}"


def exec_search_files(pattern: str, path: str = None, **kwargs) -> str:
    if not path or path == ".":
        path = kwargs.get("_working_dir", ".")
    path = _sanitize_path(path)
    try:
        matches = sorted(glob_module.glob(os.path.join(path, pattern), recursive=True))
        if not matches:
            return f"No files matching '{pattern}' in {path}"
        # Limit results
        total = len(matches)
        matches = matches[:100]
        result = "\n".join(matches)
        if total > 100:
            result += f"\n\n[{total - 100} more results not shown]"
        return result
    except Exception as e:
        return f"Error searching files: {e}"


def exec_search_content(regex: str, path: str = None, file_pattern: str = None, **kwargs) -> str:
    if not path or path == ".":
        path = kwargs.get("_working_dir", ".")
    path = _sanitize_path(path)
    try:
        # Use ripgrep if available, else fall back to Python
        cmd = ["rg", "--no-heading", "--line-number", "-m", "50"]
        if file_pattern:
            cmd.extend(["-g", file_pattern])
        cmd.extend([regex, path])
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            output = result.stdout
            if len(output) > 50000:
                output = output[:50000] + "\n[Truncated]"
            return output or "No matches found"
        elif result.returncode == 1:
            return "No matches found"
        else:
            return f"Search error: {result.stderr}"
    except FileNotFoundError:
        # ripgrep not installed, use grep
        try:
            cmd = ["grep", "-rn", "--include", file_pattern or "*", regex, path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            return result.stdout[:50000] if result.stdout else "No matches found"
        except Exception as e:
            return f"Error: {e}"
    except Exception as e:
        return f"Error searching content: {e}"


def exec_run_skill(skill_id: str, params: dict = None, **kwargs) -> str:
    if params is None:
        params = {}
    skill_path = SKILLS_DIR / f"{skill_id}.py"
    if not skill_path.exists():
        # Try partial match
        candidates = list(SKILLS_DIR.glob(f"*{skill_id}*.py"))
        if candidates:
            skill_path = candidates[0]
            skill_id = skill_path.stem
        else:
            available = sorted([f.stem for f in SKILLS_DIR.glob("skill_*.py")])
            return f"Error: Skill '{skill_id}' not found. Available: {', '.join(available[:20])}..."

    try:
        spec = importlib.util.spec_from_file_location(skill_id, str(skill_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Try main() first, then find the best entry point
        if hasattr(module, 'main'):
            result = module.main(**params)
        else:
            # Known skill entry points mapping
            SKILL_ENTRY_POINTS = {
                'skill_32_web_search': 'search',
                'skill_33_web_reader': 'read_url',
                'skill_31_web_research': 'research',
                'skill_01_code_review': 'review_code',
                'skill_38_monitoring': 'get_full_status',
                'skill_06_file_operations': 'read_file',
                'skill_08_git_operations': 'git_status',
                'skill_22_shell_commands': 'run',
                'skill_17_csv_processing': 'read_csv',
                'skill_07_json_handling': 'read_json',
                'skill_25_web_scraping': 'scrape',
                'skill_43_pdf_tools': 'extract_text',
                'skill_35_database': 'query',
                'skill_44_screenshot': 'take_screenshot',
                'skill_90_send_email': 'send',
                'skill_89_sentiment_analysis': 'analyze',
                'skill_88_trend_analysis': 'analyze',
                'skill_87_survey_analysis': 'analyze',
                'skill_92_visualization': 'create',
                'skill_93_report_generator': 'generate',
                'skill_57_security_audit': 'audit',
                'skill_86_web_testing': 'test_url',
            }
            entry = SKILL_ENTRY_POINTS.get(skill_id)
            if entry and hasattr(module, entry):
                result = getattr(module, entry)(**params)
            else:
                # Auto-discover: find first callable that isn't private/test
                import inspect
                candidates = [
                    (name, obj) for name, obj in inspect.getmembers(module, inspect.isfunction)
                    if not name.startswith('_') and name not in ('test_skill', 'test', 'format_results')
                    and obj.__module__ == module.__name__
                ]
                if candidates:
                    func_name, func = candidates[0]
                    result = func(**params)
                else:
                    return f"Error: No callable entry point found in {skill_id}"

        if isinstance(result, dict):
            return json.dumps(result, indent=2, default=str)
        return str(result)
    except Exception as e:
        return f"Error running skill {skill_id}: {e}"


# ============================================================
# IDE TOOL EXECUTORS
# ============================================================

REACT_TAILWIND_TEMPLATE = {
    "package.json": lambda name, desc: json.dumps({
        "name": name,
        "private": True,
        "version": "0.0.1",
        "description": desc or f"{name} — built with Klaus",
        "type": "module",
        "scripts": {
            "dev": "vite --host 0.0.0.0",
            "build": "tsc && vite build",
            "preview": "vite preview"
        },
        "dependencies": {
            "react": "^19.0.0",
            "react-dom": "^19.0.0"
        },
        "devDependencies": {
            "@types/react": "^19.0.0",
            "@types/react-dom": "^19.0.0",
            "@vitejs/plugin-react": "^4.3.0",
            "autoprefixer": "^10.4.20",
            "postcss": "^8.4.49",
            "tailwindcss": "^3.4.17",
            "typescript": "^5.7.0",
            "vite": "^6.0.0"
        }
    }, indent=2),
    "vite.config.ts": lambda name, desc: """import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: { host: '0.0.0.0' }
})
""",
    "tsconfig.json": lambda name, desc: json.dumps({
        "compilerOptions": {
            "target": "ES2020",
            "useDefineForClassFields": True,
            "lib": ["ES2020", "DOM", "DOM.Iterable"],
            "module": "ESNext",
            "skipLibCheck": True,
            "moduleResolution": "bundler",
            "allowImportingTsExtensions": True,
            "isolatedModules": True,
            "moduleDetection": "force",
            "noEmit": True,
            "jsx": "react-jsx",
            "strict": True,
            "noUnusedLocals": False,
            "noUnusedParameters": False,
        },
        "include": ["src"]
    }, indent=2),
    "postcss.config.js": lambda name, desc: """export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
""",
    "tailwind.config.js": lambda name, desc: """/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
}
""",
    "index.html": lambda name, desc: f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
""",
    "src/main.tsx": lambda name, desc: """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
""",
    "src/index.css": lambda name, desc: """@tailwind base;
@tailwind components;
@tailwind utilities;
""",
    "src/App.tsx": lambda name, desc: f"""export default function App() {{
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900">{name}</h1>
        <p className="mt-2 text-gray-600">{desc or 'Built with Klaus'}</p>
      </div>
    </div>
  )
}}
""",
    "src/vite-env.d.ts": lambda name, desc: '/// <reference types="vite/client" />\n',
}


def exec_create_project(name: str, description: str = "", **kwargs) -> str:
    """Scaffold a React+Tailwind project."""
    global _LAST_PROJECT_DIR
    name = re.sub(r'[^a-z0-9_-]', '-', name.lower().strip())
    if not name:
        return "Error: Invalid project name"
    project_dir = os.path.join(DEFAULT_PROJECT_DIR, name)
    if os.path.exists(project_dir):
        _LAST_PROJECT_DIR = project_dir
        return f"Project already exists at {project_dir}. Use write_file/edit_file to modify it."
    try:
        os.makedirs(os.path.join(project_dir, "src"), exist_ok=True)
        for rel_path, gen_fn in REACT_TAILWIND_TEMPLATE.items():
            full_path = os.path.join(project_dir, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(gen_fn(name, description))
        _LAST_PROJECT_DIR = project_dir
        return f"Created project at {project_dir}\nFiles: {', '.join(REACT_TAILWIND_TEMPLATE.keys())}\nNext: install_packages(packages=\"\") to install deps, then start_dev_server(project_dir=\"{project_dir}\")"
    except Exception as e:
        return f"Error creating project: {e}"


_LAST_PROJECT_DIR: str = ""  # Track last created/used project


def exec_install_packages(packages: str = "", project_dir: str = None, **kwargs) -> str:
    """Install npm packages."""
    global _LAST_PROJECT_DIR
    wd = project_dir or _LAST_PROJECT_DIR or kwargs.get("_working_dir", DEFAULT_PROJECT_DIR)
    wd = _sanitize_path(wd)
    if not os.path.isfile(os.path.join(wd, "package.json")):
        # Try to find a package.json in a subdirectory
        for d in sorted(Path(wd).iterdir()):
            if d.is_dir() and (d / "package.json").exists():
                wd = str(d)
                break
        else:
            return f"Error: No package.json found in {wd}"
    _LAST_PROJECT_DIR = wd
    try:
        # Always install base deps first if node_modules doesn't exist
        if not os.path.isdir(os.path.join(wd, "node_modules")):
            base = subprocess.run(
                "npm install", shell=True, capture_output=True, text=True,
                timeout=120, cwd=wd
            )
            if base.returncode != 0:
                return f"Error installing base deps:\n{base.stderr[:2000]}"

        if packages.strip():
            result = subprocess.run(
                f"npm install {packages}", shell=True, capture_output=True, text=True,
                timeout=120, cwd=wd
            )
            if result.returncode != 0:
                return f"Error installing {packages}:\n{result.stderr[:2000]}"
            return f"Installed {packages} in {wd}"
        return f"Base dependencies installed in {wd}"
    except subprocess.TimeoutExpired:
        return "Error: npm install timed out (120s)"
    except Exception as e:
        return f"Error: {e}"


# Track running dev servers
_DEV_SERVERS: dict = {}  # port -> subprocess.Popen


def exec_start_dev_server(project_dir: str = None, **kwargs) -> str:
    """Start Vite dev server."""
    wd = project_dir or _LAST_PROJECT_DIR or kwargs.get("_working_dir", DEFAULT_PROJECT_DIR)
    wd = _sanitize_path(wd)
    if not os.path.isfile(os.path.join(wd, "package.json")):
        return f"Error: No package.json in {wd}"
    # Install deps if needed
    if not os.path.isdir(os.path.join(wd, "node_modules")):
        install_result = exec_install_packages("", project_dir=wd)
        if install_result.startswith("Error"):
            return install_result
    try:
        # Kill any existing server for this dir
        for port, info in list(_DEV_SERVERS.items()):
            if info.get("dir") == wd:
                try:
                    info["proc"].terminate()
                except:
                    pass
                del _DEV_SERVERS[port]

        proc = subprocess.Popen(
            "npx vite --host 0.0.0.0 --port 5173",
            shell=True, cwd=wd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True
        )
        import time
        time.sleep(3)  # Give it time to start
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr else ""
            return f"Error: Dev server exited immediately.\n{stderr[:1000]}"
        _DEV_SERVERS[5173] = {"proc": proc, "dir": wd}
        return f"Dev server started on port 5173 for {wd}\nPreview should auto-detect it."
    except Exception as e:
        return f"Error starting dev server: {e}"


# Executor dispatch
TOOL_EXECUTORS = {
    "read_file": exec_read_file,
    "write_file": exec_write_file,
    "edit_file": exec_edit_file,
    "list_directory": exec_list_directory,
    "run_command": exec_run_command,
    "search_files": exec_search_files,
    "search_content": exec_search_content,
    "run_skill": exec_run_skill,
    "create_project": exec_create_project,
    "install_packages": exec_install_packages,
    "start_dev_server": exec_start_dev_server,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with given arguments. Returns result string."""
    executor = TOOL_EXECUTORS.get(name)
    if not executor:
        return f"Error: Unknown tool '{name}'"
    try:
        return executor(**arguments)
    except Exception as e:
        return f"Error executing {name}: {e}"


# ============================================================
# SKILL CATALOG (for system prompt)
# ============================================================

def get_skill_catalog() -> str:
    """Generate a compact skill catalog string for the system prompt."""
    index_path = SKILLS_DIR / "SKILLS_INDEX.md"
    if index_path.exists():
        try:
            content = index_path.read_text()
            # Extract just skill lines (compact format)
            lines = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('- `skill_'):
                    # Extract: skill_01_code_review - Code quality analysis
                    match = re.match(r'- `(skill_\w+)`\s*[-–]\s*(.*)', line)
                    if match:
                        lines.append(f"  {match.group(1)}: {match.group(2)}")
            if lines:
                return "Available skills (use run_skill tool):\n" + "\n".join(lines)
        except:
            pass

    # Fallback: scan directory
    skills = sorted(SKILLS_DIR.glob("skill_*.py"))
    lines = []
    for s in skills:
        name = s.stem
        # Try to extract description from docstring
        try:
            first_lines = s.read_text()[:500]
            desc_match = re.search(r'""".*?Purpose:\s*(.*?)(?:\n|""")', first_lines, re.DOTALL)
            desc = desc_match.group(1).strip() if desc_match else name.replace('skill_', '').replace('_', ' ')
        except:
            desc = name.replace('skill_', '').replace('_', ' ')
        lines.append(f"  {name}: {desc}")

    return "Available skills (use run_skill tool):\n" + "\n".join(lines) if lines else "No skills found."
