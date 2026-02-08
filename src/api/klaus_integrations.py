"""
Klaus Integration Executor
Executes real actions for Klaus (iMessage, Calendar, Web Search, etc.)
"""

import subprocess
import json
import os
from pathlib import Path
import urllib.parse
import urllib.request

# AppleScript tool paths
APPLESCRIPT_DIR = Path.home() / ".axe/applescript"

class KlausIntegrations:
    """Execute Klaus's integration requests"""

    def __init__(self):
        self.tools = {
            "imessage": self.send_imessage,
            "calendar": self.check_calendar,
            "notes": self.create_note,
            "reminders": self.create_reminder,
            "web_search": self.web_search,
            "filesystem": self.filesystem_op,
        }

    def execute(self, action: str, params: dict) -> dict:
        """Execute an integration action"""
        if action in self.tools:
            try:
                result = self.tools[action](**params)
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return {"success": False, "error": f"Unknown action: {action}"}

    # ============ iMessage ============
    def send_imessage(self, to: str, message: str) -> str:
        """Send iMessage using AppleScript"""
        script = APPLESCRIPT_DIR / "imessage.sh"
        if not script.exists():
            return "iMessage script not found"

        result = subprocess.run(
            [str(script), "send", to, message],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout or result.stderr or "Message sent"

    # ============ Calendar ============
    def check_calendar(self, days: int = 1) -> str:
        """Check calendar events"""
        script = APPLESCRIPT_DIR / "calendar.sh"
        if not script.exists():
            return "Calendar script not found"

        result = subprocess.run(
            [str(script), "today"],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout or "No events found"

    # ============ Notes ============
    def create_note(self, title: str, body: str) -> str:
        """Create Apple Note"""
        script = APPLESCRIPT_DIR / "notes.sh"
        if not script.exists():
            return "Notes script not found"

        result = subprocess.run(
            [str(script), "create", title, body],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout or "Note created"

    # ============ Reminders ============
    def create_reminder(self, title: str, due: str = None) -> str:
        """Create reminder"""
        script = APPLESCRIPT_DIR / "reminders.sh"
        if not script.exists():
            return "Reminders script not found"

        args = [str(script), "add", title]
        if due:
            args.append(due)

        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return result.stdout or "Reminder created"

    # ============ Web Search ============
    def web_search(self, query: str, num_results: int = 5) -> str:
        """Perform DuckDuckGo search"""
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1"

            req = urllib.request.Request(url, headers={'User-Agent': 'Klaus/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read())

            results = []

            # Abstract (instant answer)
            if data.get("Abstract"):
                results.append(f"**Summary:** {data['Abstract']}")

            # Related topics
            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(f"- {topic['Text']}")

            return "\n".join(results) if results else "No results found"

        except Exception as e:
            return f"Search error: {e}"

    # ============ Filesystem ============
    def filesystem_op(self, operation: str, path: str, content: str = None) -> str:
        """Filesystem operations (read/write/list)"""
        path = Path(path).expanduser()

        if operation == "read":
            if path.exists():
                return path.read_text()[:5000]  # Limit size
            return f"File not found: {path}"

        elif operation == "write":
            if content:
                path.write_text(content)
                return f"Written to {path}"
            return "No content provided"

        elif operation == "list":
            if path.is_dir():
                files = [f.name for f in path.iterdir()][:50]
                return "\n".join(files)
            return f"Not a directory: {path}"

        return f"Unknown operation: {operation}"


# Singleton instance
integrations = KlausIntegrations()


def parse_and_execute(klaus_response: str) -> list:
    """
    Parse Klaus's response for action intents and execute them.

    Klaus should format actions as:
    [ACTION:imessage TO:+1234567890 MSG:Hello there]
    [ACTION:calendar DAYS:1]
    [ACTION:web_search QUERY:python tutorials]
    """
    import re

    results = []

    # Find all action blocks
    action_pattern = r'\[ACTION:(\w+)([^\]]*)\]'
    matches = re.findall(action_pattern, klaus_response, re.IGNORECASE)

    for action, params_str in matches:
        # Parse parameters
        params = {}
        param_pattern = r'(\w+):([^\s\]]+(?:\s+[^\s\]:]+)*)'
        for key, value in re.findall(param_pattern, params_str):
            params[key.lower()] = value.strip()

        # Execute
        result = integrations.execute(action.lower(), params)
        results.append({
            "action": action,
            "params": params,
            "result": result
        })

    return results
