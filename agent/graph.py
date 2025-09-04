from langchain_groq import ChatGroq
from langchain.globals import set_debug, set_verbose 
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.agents import create_react_agent
from dotenv import load_dotenv
from prompts import *
from states import *
from tools import *
import os
import pprint

_ = load_dotenv()

set_debug(True)
set_verbose(True)



from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_coder_prompt(system_prompt: str, user_prompt: str):
    # Load the standard ReAct prompt
    react_prompt = hub.pull("hwchase17/react")

    # Add our system + user messages on top
    custom_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt),
        MessagesPlaceholder("agent_scratchpad"),  # keep scratchpad
    ])

    # Merge variables from react prompt (tools, tool_names, input)
    custom_prompt.input_variables = list(

    set(react_prompt.input_variables + custom_prompt.input_variables)
    
    )


    return custom_prompt



llm = ChatOpenAI(

    api_key = os.getenv("PERPLEXITY_API_KEY"),
    model = "sonar-pro",
    base_url="https://api.perplexity.ai"
)

llm_groq = ChatGroq(

    model="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
    # base_url="https://api.groq.com/openai/v1"
)




def planner_agent(state: dict) -> dict:
    users_prompt = state["user_prompt"]
    response = llm.with_structured_output(Plan).invoke(user_prompt)
    return {"plan":response}


def architect_agent(state: dict) -> dict:
    plan: Plan = state["plan"]
    response = llm.with_structured_output(TaskPlan).invoke(architect_prompt(plan))
    
    if response is None:
        raise ValueError("Architect did not return a valid response.")
    
    response.plan = plan

    return {"task_plan": response}



from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate

def coder_agent(state: dict) -> dict:
    """LangGraph tool-using coder agent."""
    coder_state: CoderState = state.get("coder_state")
    if coder_state is None:
        coder_state = CoderState(task_plan=state["task_plan"], current_step_idx=0)

    steps = coder_state.task_plan.implementation_steps
    if coder_state.current_step_idx >= len(steps):
        return {"coder_state": coder_state, "status": "DONE"}

    current_task = steps[coder_state.current_step_idx]
    existing_content = read_file.run(current_task.filepath)

    # Build system + user prompt for the task
    system_prompt = coder_system_prompt()
    user_prompt = (
        f"Task: {current_task.task_description}\n"
        f"File: {current_task.filepath}\n"
        f"Existing content:\n{existing_content}\n"
        "Use write_file(path, content) to save your changes."
    )

    # ✅ Use the official ReAct prompt (includes tool_names, tools, agent_scratchpad)
    # react_prompt = hub.pull("hwchase17/react")

    react_prompt = build_coder_prompt(system_prompt, user_prompt)

    coder_tools = [read_file, write_file, list_files, get_current_directory]

    # Create the agent
    agent = create_react_agent(llm_groq, coder_tools, prompt=react_prompt)

    # Wrap agent into executor(no stop)
    agent_executor = AgentExecutor(agent=agent, tools=coder_tools, verbose=True, handle_parsing_errors=True)

    # ✅ Invoke using input string (not messages)
    agent_executor.invoke(
        {"input": f"{system_prompt}\n\n{user_prompt}"}
    )

    coder_state.current_step_idx += 1
    return {"coder_state": coder_state}







graph = StateGraph(dict)
graph.add_node("planner", planner_agent)
graph.add_node("architect", architect_agent)
graph.add_node("coder",coder_agent)

graph.add_edge("planner","architect")
graph.add_edge("architect","coder")
graph.set_entry_point("planner")


agent = graph.compile()




if __name__ == "__main__":

    user_prompt = "Generate a simple calculator web application"

    result = agent.invoke({"user_prompt": user_prompt})

    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(result)


