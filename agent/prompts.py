def planner_prompt(user_prompt: str) -> str:
    PLANNER_PROMPT = f"""

You are the PLANNER agent. Convert the user prompt into a COMPLETE engineering project plan

User request: {user_prompt}

"""
    
    return PLANNER_PROMPT


def architect_prompt(plan: str) -> str:

    ARCHITECT_PROMPT =f"""

You are the ARCHITECT agent. Given this project plan,break into explicit engineering tasks.

RULES:
- For each FILE in the plan, create one or more IMPLEMENTATION TASKS.
-In each task description:
    * Specify exactly what to implement.
    * Name the variables, functions, Classes, and components to be defined.
    * Mention how this task depends on or will be used by previous tasks.
    * Include integration details: imports, expected function signatures, data flow.
- Order tasks so that dependencies are implemented first.
- Each step must be SELF-CONTAINED but also carry FORWARD the relevant context from

Project Plan:
{plan}

"""
    
    return ARCHITECT_PROMPT



def coder_system_prompt() -> str:
    CODER_SYSTEM_PROMPT = """You are a coding agent that edits and creates project files.

When using the tool `write_file`, you MUST output valid structured JSON with TWO separate fields:

{
  "path": "index.html",
  "content": "<!DOCTYPE html>...full file content...</html>"
}

❌ Do NOT wrap JSON in a string.  
❌ Do NOT include both path and content inside one field.  
✅ Path must only contain the filename or relative path.  
✅ Content must contain the full file content.
"""

    return CODER_SYSTEM_PROMPT