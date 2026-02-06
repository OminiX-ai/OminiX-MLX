#!/usr/bin/env python3
"""Real tool-use orchestration for GLM-4.7-Flash.

Runs the model, parses <tool_call> XML, executes real tools,
feeds observations back, and gets the final answer.
"""

import subprocess
import sys
import re
import json
import datetime
import math
import os

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/GLM-4.7-Flash-MLX-6.5bit"
BINARY = "target/release/examples/generate_glm4_flash"

TOOLS_JSON = [
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Execute Python code and return stdout output",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code to execute"}
        }, "required": ["code"]}
    }},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "Timezone like UTC, US/Pacific, Asia/Tokyo"}
        }, "required": []}
    }},
    {"type": "function", "function": {
        "name": "http_get",
        "description": "Fetch a URL and return the response body (text only, max 500 chars)",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "URL to fetch"}
        }, "required": ["url"]}
    }},
]


def build_tools_system():
    tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in TOOLS_JSON)
    return f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, output the function name and arguments within the following XML format:
<tool_call>{{function-name}}<arg_key>{{arg-key-1}}</arg_key><arg_value>{{arg-value-1}}</arg_value>...</tool_call>"""


def run_model(prompt, max_tokens=300):
    """Run the GLM-4.7-Flash binary and return generated text."""
    result = subprocess.run(
        [BINARY, MODEL_DIR, prompt, str(max_tokens)],
        capture_output=True, text=True, timeout=120,
    )
    # Output is on stdout, model logs on stderr
    stdout = result.stdout
    # Extract text between "---" markers
    parts = stdout.split("---")
    if len(parts) >= 2:
        return parts[1].strip()
    return stdout.strip()


def parse_tool_calls(text):
    """Parse <tool_call> XML from model output."""
    calls = []
    pattern = r'<tool_call>(.*?)</tool_call>'
    for match in re.finditer(pattern, text, re.DOTALL):
        content = match.group(1).strip()
        # Extract function name (text before first <arg_key>)
        name_match = re.match(r'([^<]+)', content)
        if not name_match:
            continue
        func_name = name_match.group(1).strip()
        # Extract args
        args = {}
        keys = re.findall(r'<arg_key>(.*?)</arg_key>', content)
        vals = re.findall(r'<arg_value>(.*?)</arg_value>', content, re.DOTALL)
        for k, v in zip(keys, vals):
            args[k.strip()] = v.strip()
        calls.append({"name": func_name, "arguments": args})
    return calls


def execute_tool(name, args):
    """Actually execute a tool and return real results."""
    if name == "execute_python":
        code = args.get("code", "")
        print(f"\n  [EXECUTING PYTHON]\n  {'-'*40}")
        for line in code.split('\n'):
            print(f"  >>> {line}")
        print(f"  {'-'*40}")
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
            if result.returncode != 0:
                output = f"Error: {result.stderr.strip()}"
            print(f"  [OUTPUT] {output}")
            return output
        except subprocess.TimeoutExpired:
            return "Error: execution timed out"

    elif name == "get_current_time":
        tz = args.get("timezone", "UTC")
        # Use python to get timezone-aware time
        try:
            result = subprocess.run(
                ["python3", "-c", f"""
import datetime, zoneinfo
try:
    tz = zoneinfo.ZoneInfo("{tz}")
except Exception:
    tz = zoneinfo.ZoneInfo("UTC")
now = datetime.datetime.now(tz)
print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
"""],
                capture_output=True, text=True, timeout=5,
            )
            output = result.stdout.strip()
        except Exception:
            output = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  [TIME] {output}")
        return output

    elif name == "http_get":
        url = args.get("url", "")
        print(f"  [FETCHING] {url}")
        try:
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "5", url],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout[:500]
            print(f"  [RESPONSE] {output[:100]}...")
            return output
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <user_query>")
        sys.exit(1)

    user_query = sys.argv[2]
    max_turns = 3

    print(f"=" * 60)
    print(f"Real Tool-Use Test: GLM-4.7-Flash")
    print(f"=" * 60)
    print(f"User: {user_query}\n")

    system = build_tools_system()
    conversation = f"<|system|>\n{system}\n<|user|>{user_query}\n<|assistant|></think>"

    for turn in range(max_turns):
        print(f"--- Turn {turn + 1}: Running model ---")
        response = run_model(conversation, max_tokens=400)
        print(f"\nModel: {response}\n")

        # Parse tool calls
        tool_calls = parse_tool_calls(response)
        if not tool_calls:
            print("[No tool calls detected - final answer reached]")
            break

        print(f"[Detected {len(tool_calls)} tool call(s)]")

        # Execute each tool
        observations = []
        for tc in tool_calls:
            print(f"\n  Tool: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
            result = execute_tool(tc['name'], tc['arguments'])
            observations.append(result)

        # Build observation block
        obs_block = "<|observation|>" + "".join(
            f"<tool_response>{obs}</tool_response>" for obs in observations
        )

        # Strip any text after last tool_call for clean continuation
        last_tc_end = response.rfind("</tool_call>")
        if last_tc_end >= 0:
            response_prefix = response[:last_tc_end + len("</tool_call>")]
        else:
            response_prefix = response

        # Append to conversation
        conversation = (
            f"<|system|>\n{system}\n<|user|>{user_query}\n"
            f"<|assistant|></think>\n{response_prefix}\n"
            f"{obs_block}\n<|assistant|></think>"
        )

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
