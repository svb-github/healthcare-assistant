"""
🏥 Healthcare AI Assistant — Multi-Agent System
=================================================
An AI-powered healthcare assistant built with LangGraph + Gradio.

Concepts used from hackathon missions:
- mission3: LangGraph workflow — conditional routing, nodes & edges
- mission4: ReAct tools — custom health calculator tools
- mission5: Multi-agent — supervisor routes to specialist agents
- mission7: Self-healing — error handling with graceful fallback
- small_business_bot: Gradio UI + Docker + uv deployment pattern

Deployment: GitHub → Docker → GCP Cloud Run (same as small_business_bot)
"""

import os
import sys
import json
from pathlib import Path
from typing import TypedDict, Annotated, List, Literal
from dotenv import load_dotenv

# Ensure we can find .env and tools.py relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Load environment variables from the .env file next to this script
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from tools import (
    calculate_bmi,
    calculate_daily_calories,
    calculate_heart_rate_zones,
    calculate_water_intake,
    assess_symptom_severity,
)

# ============================================================================
# AGENT STATE (mission3 concept: state management)
# ============================================================================

class AgentState(TypedDict):
    """State shared across all nodes in the graph."""
    messages: Annotated[List[BaseMessage], add_messages]
    route: str  # Which specialist agent to route to
    error_count: int  # For self-healing retry logic (mission7 concept)


# ============================================================================
# LLM SETUP
# ============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


# ============================================================================
# SYSTEM PROMPTS FOR EACH SPECIALIST AGENT
# ============================================================================

ROUTER_PROMPT = """You are a healthcare assistant router. Your ONLY job is to classify the user's message into one of these categories:

1. "symptom" — if the user is describing symptoms, feeling unwell, or asking about medical conditions
2. "wellness" — if the user asks about BMI, calories, diet, exercise, water intake, fitness, or lifestyle
3. "medicine" — if the user asks about medications, drugs, side effects, dosage, or drug interactions
4. "general" — for any other general health questions, greetings, or unclear queries

CRITICAL: You MUST respond with ONLY a single word — one of: symptom, wellness, medicine, general
Do NOT add any explanation. Just the category word."""

SYMPTOM_AGENT_PROMPT = """You are a compassionate and knowledgeable Symptom Analysis Agent. You help users understand their symptoms and provide guidance on next steps.

IMPORTANT RULES:
1. You are NOT a doctor. ALWAYS include a disclaimer that this is not a medical diagnosis.
2. Ask clarifying questions if the symptom description is vague.
3. When you have enough info, use the symptom severity assessment tool data provided.
4. Suggest when to see a doctor — be conservative (err on the side of caution).
5. Never prescribe medication. Only provide general health information.
6. For emergencies (chest pain, difficulty breathing, etc.), IMMEDIATELY advise calling emergency services.

When analyzing symptoms, consider:
- Duration of symptoms
- Severity
- Associated symptoms
- Age-related risk factors
- Whether fever is present

At the end of EVERY response, provide 3 follow-up questions under "💡 You might also want to ask:" heading."""

WELLNESS_AGENT_PROMPT = """You are a friendly Wellness & Fitness Advisor Agent. You help users with nutrition, exercise, and healthy lifestyle guidance.

You have access to these health calculators (data will be provided when relevant):
- BMI Calculator
- Daily Calorie Needs (Mifflin-St Jeor equation)
- Heart Rate Training Zones (Karvonen method)
- Daily Water Intake Calculator

IMPORTANT RULES:
1. Be encouraging and positive — never shame users about their weight or habits.
2. Provide evidence-based advice.
3. Suggest consulting a dietitian or fitness professional for personalized plans.
4. When users provide their metrics (weight, height, age), USE the calculator tools and share results.
5. Offer practical, actionable tips — not just theory.

Focus areas: BMI analysis, calorie planning, hydration, exercise guidance, sleep hygiene, stress management.

At the end of EVERY response, provide 3 follow-up questions under "💡 You might also want to ask:" heading."""

MEDICINE_AGENT_PROMPT = """You are a Medicine Information Agent. You provide general educational information about medications.

IMPORTANT RULES:
1. You are NOT a pharmacist or doctor. ALWAYS include a disclaimer.
2. NEVER prescribe medications or recommend specific drugs for conditions.
3. Only provide GENERAL educational information about medicines when asked.
4. Always advise consulting a doctor or pharmacist before starting/stopping any medication.
5. Mention common side effects and important drug interactions when relevant.
6. For dosage questions, ALWAYS say "Follow your doctor's prescription" — never suggest doses.

You can discuss:
- General information about drug classes (antibiotics, painkillers, etc.)
- Common side effects (educational purposes only)
- General drug interaction awareness
- When to seek medical attention regarding medications

At the end of EVERY response, provide 3 follow-up questions under "💡 You might also want to ask:" heading."""

GENERAL_AGENT_PROMPT = """You are a friendly Healthcare AI Assistant. You provide general health education and wellness information.

IMPORTANT RULES:
1. Be warm, empathetic, and approachable.
2. For medical emergencies, immediately advise calling emergency services.
3. Always recommend consulting healthcare professionals for specific medical concerns.
4. Provide evidence-based general health information.
5. Cover topics like preventive health, healthy habits, mental health awareness, and health literacy.

You are here to educate and guide — NOT to diagnose or treat.

At the end of EVERY response, provide 3 follow-up questions under "💡 You might also want to ask:" heading."""


# ============================================================================
# GRAPH NODES (mission3 concept: nodes and edges)
# ============================================================================

def router_node(state: AgentState) -> dict:
    """
    Supervisor/Router node — classifies user query and decides which specialist to invoke.
    Concept from: mission5 (supervisor pattern) + mission3 (conditional routing)
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    try:
        route_response = llm.invoke([
            SystemMessage(content=ROUTER_PROMPT),
            HumanMessage(content=last_message),
        ])
        route = route_response.content.strip().lower()

        # Validate the route
        valid_routes = ["symptom", "wellness", "medicine", "general"]
        if route not in valid_routes:
            route = "general"  # Fallback (self-healing concept from mission7)

        return {"route": route, "error_count": 0}

    except Exception as e:
        # Self-healing: if routing fails, default to general agent (mission7 concept)
        print(f"[Self-Healing] Router error: {e}. Falling back to 'general' agent.")
        return {"route": "general", "error_count": state.get("error_count", 0) + 1}


def symptom_agent_node(state: AgentState) -> dict:
    """Symptom Analysis Agent — analyzes symptoms and provides guidance."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Try to extract symptom info and run tool
    tool_context = ""
    try:
        # Ask the LLM to extract structured symptom data
        extraction_prompt = """Extract symptom information from this message. Return a JSON object with:
        - "symptoms": list of symptom strings
        - "duration_days": number of days (default 1 if not mentioned)
        - "has_fever": true/false
        - "age": patient age (default 30 if not mentioned)
        
        If the message is vague or just a greeting, return: {"symptoms": []}
        
        RESPOND WITH ONLY THE JSON, no other text."""

        extraction = llm.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=last_message),
        ])

        symptom_data = json.loads(extraction.content.strip())
        if symptom_data.get("symptoms"):
            assessment = assess_symptom_severity(
                symptoms=symptom_data["symptoms"],
                duration_days=symptom_data.get("duration_days", 1),
                has_fever=symptom_data.get("has_fever", False),
                age=symptom_data.get("age", 30),
            )
            tool_context = f"\n\n[TOOL RESULT - Symptom Assessment]:\n{json.dumps(assessment, indent=2)}"
    except Exception as e:
        # Self-healing: continue without tool data if extraction fails (mission7 concept)
        print(f"[Self-Healing] Symptom tool error: {e}. Continuing without tool data.")
        tool_context = ""

    # Build messages for the specialist agent
    agent_messages = [SystemMessage(content=SYMPTOM_AGENT_PROMPT)]
    # Add conversation history (skip the last message, we'll re-add it with tool context)
    for msg in messages[:-1]:
        agent_messages.append(msg)
    # Add the current message with tool context
    agent_messages.append(HumanMessage(content=last_message + tool_context))

    try:
        response = llm.invoke(agent_messages)
        return {"messages": [response]}
    except Exception as e:
        # Self-healing fallback (mission7 concept)
        error_msg = AIMessage(content=f"I'm having trouble analyzing your symptoms right now. Please try again, or if you're experiencing an emergency, please call emergency services immediately.\n\n⚕️ Error details: {str(e)}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}


def wellness_agent_node(state: AgentState) -> dict:
    """Wellness & Fitness Advisor Agent — BMI, calories, hydration, exercise."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Try to extract health metrics and run tools
    tool_context = ""
    try:
        extraction_prompt = """Extract health metrics from this message. Return a JSON object with any of these fields that are mentioned:
        - "weight_kg": weight in kg (convert from lbs if needed: lbs * 0.453592)
        - "height_cm": height in cm (convert from feet/inches if needed)
        - "age": age in years
        - "gender": "male" or "female"
        - "activity_level": "sedentary", "light", "moderate", "active", or "very_active"
        - "resting_hr": resting heart rate in BPM
        - "query_type": one of "bmi", "calories", "heart_rate", "water", "general_wellness"
        
        If no specific metrics are mentioned, return: {"query_type": "general_wellness"}
        
        RESPOND WITH ONLY THE JSON, no other text."""

        extraction = llm.invoke([
            SystemMessage(content=extraction_prompt),
            HumanMessage(content=last_message),
        ])

        metrics = json.loads(extraction.content.strip())
        tool_results = []

        # Run relevant tools based on available data
        if metrics.get("weight_kg") and metrics.get("height_cm"):
            bmi_result = calculate_bmi(metrics["weight_kg"], metrics["height_cm"])
            tool_results.append(f"BMI Calculator: {json.dumps(bmi_result, indent=2)}")

            if metrics.get("age") and metrics.get("gender"):
                calorie_result = calculate_daily_calories(
                    weight_kg=metrics["weight_kg"],
                    height_cm=metrics["height_cm"],
                    age=metrics["age"],
                    gender=metrics["gender"],
                    activity_level=metrics.get("activity_level", "moderate"),
                )
                tool_results.append(f"Calorie Calculator: {json.dumps(calorie_result, indent=2)}")

            water_result = calculate_water_intake(
                weight_kg=metrics["weight_kg"],
                activity_level=metrics.get("activity_level", "moderate"),
            )
            tool_results.append(f"Water Intake: {json.dumps(water_result, indent=2)}")

        if metrics.get("age") and (metrics.get("query_type") == "heart_rate" or metrics.get("resting_hr")):
            hr_result = calculate_heart_rate_zones(
                age=metrics["age"],
                resting_hr=metrics.get("resting_hr", 70),
            )
            tool_results.append(f"Heart Rate Zones: {json.dumps(hr_result, indent=2)}")

        if tool_results:
            tool_context = "\n\n[TOOL RESULTS]:\n" + "\n\n".join(tool_results)

    except Exception as e:
        print(f"[Self-Healing] Wellness tool error: {e}. Continuing without tool data.")
        tool_context = ""

    agent_messages = [SystemMessage(content=WELLNESS_AGENT_PROMPT)]
    for msg in messages[:-1]:
        agent_messages.append(msg)
    agent_messages.append(HumanMessage(content=last_message + tool_context))

    try:
        response = llm.invoke(agent_messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"I'm having trouble processing your wellness query right now. Please try again.\n\nError: {str(e)}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}


def medicine_agent_node(state: AgentState) -> dict:
    """Medicine Information Agent — general drug/medication education."""
    messages = state["messages"]

    agent_messages = [SystemMessage(content=MEDICINE_AGENT_PROMPT)]
    for msg in messages:
        agent_messages.append(msg)

    try:
        response = llm.invoke(agent_messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"I'm having trouble retrieving medicine information right now. Please consult your pharmacist or doctor directly.\n\nError: {str(e)}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}


def general_agent_node(state: AgentState) -> dict:
    """General Health Agent — handles greetings, general health questions, and fallback."""
    messages = state["messages"]

    agent_messages = [SystemMessage(content=GENERAL_AGENT_PROMPT)]
    for msg in messages:
        agent_messages.append(msg)

    try:
        response = llm.invoke(agent_messages)
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"I'm experiencing a temporary issue. Please try again in a moment.\n\nError: {str(e)}")
        return {"messages": [error_msg], "error_count": state.get("error_count", 0) + 1}


# ============================================================================
# CONDITIONAL ROUTING (mission3 concept: conditional edges)
# ============================================================================

def route_to_specialist(state: AgentState) -> Literal["symptom_agent", "wellness_agent", "medicine_agent", "general_agent"]:
    """
    Conditional edge function — routes to the appropriate specialist agent.
    Concept from: mission3 (03_conditional_routing.py) + mission5 (supervisor pattern)
    """
    route = state.get("route", "general")

    route_map = {
        "symptom": "symptom_agent",
        "wellness": "wellness_agent",
        "medicine": "medicine_agent",
        "general": "general_agent",
    }

    return route_map.get(route, "general_agent")


# ============================================================================
# BUILD THE LANGGRAPH (mission3 + mission5 concepts)
# ============================================================================

def create_healthcare_agent():
    """
    Build and compile the multi-agent LangGraph workflow.
    
    Graph structure:
        START → router → (conditional) → symptom_agent / wellness_agent / medicine_agent / general_agent → END
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("symptom_agent", symptom_agent_node)
    workflow.add_node("wellness_agent", wellness_agent_node)
    workflow.add_node("medicine_agent", medicine_agent_node)
    workflow.add_node("general_agent", general_agent_node)

    # Add edges
    workflow.add_edge(START, "router")

    # Conditional routing from router to specialist agents
    workflow.add_conditional_edges(
        "router",
        route_to_specialist,
        {
            "symptom_agent": "symptom_agent",
            "wellness_agent": "wellness_agent",
            "medicine_agent": "medicine_agent",
            "general_agent": "general_agent",
        },
    )

    # All specialist agents go to END
    workflow.add_edge("symptom_agent", END)
    workflow.add_edge("wellness_agent", END)
    workflow.add_edge("medicine_agent", END)
    workflow.add_edge("general_agent", END)

    return workflow.compile()


# Initialize the agent
agent = create_healthcare_agent()


# ============================================================================
# GRADIO CHAT INTERFACE (same pattern as small_business_bot)
# ============================================================================

def chat_interface(history: List[dict]) -> str:
    """
    Gradio chat interface function.
    Converts Gradio message history to LangChain messages and invokes the agent.
    """
    # Convert Gradio history to LangChain messages
    messages = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    # Run the multi-agent graph
    try:
        response = agent.invoke({
            "messages": messages,
            "route": "",
            "error_count": 0,
        })
        return response["messages"][-1].content
    except Exception as e:
        return f"⚠️ Something went wrong. Please try again.\n\nError: {str(e)}\n\nIf this persists, please ensure your OPENAI_API_KEY is configured correctly."


# ============================================================================
# GRADIO UI (same pattern as small_business_bot, enhanced for healthcare)
# ============================================================================

EXAMPLE_QUERIES = [
    "I've been having headaches and dizziness for 3 days",
    "Calculate my BMI — I'm 75kg and 170cm tall",
    "What are the side effects of ibuprofen?",
    "I'm a 28 year old male, 70kg, 175cm, moderate activity. What should my daily calories be?",
    "How much water should I drink daily if I weigh 65kg?",
    "What are the heart rate zones for a 35 year old?",
    "Tips for better sleep hygiene",
    "When should I see a doctor for a persistent cough?",
]

CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; }
.disclaimer-box { 
    background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    border-left: 4px solid #ffc107;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 14px;
}
.header-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 24px;
    border-radius: 12px;
    color: white;
    margin-bottom: 16px;
    text-align: center;
}
.header-box h1 { color: white !important; margin: 0 0 8px 0; font-size: 28px; }
.header-box p { color: rgba(255,255,255,0.9) !important; margin: 0; font-size: 15px; }
.agent-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 4px;
}
"""

with gr.Blocks(title="Healthcare AI Assistant") as demo:

    # Header
    gr.HTML("""
    <div class="header-box">
        <h1>🏥 Healthcare AI Assistant</h1>
        <p>Your intelligent health companion — powered by multi-agent AI</p>
        <div style="margin-top: 12px;">
            <span class="agent-badge" style="background: rgba(255,255,255,0.2);">🩺 Symptom Analyzer</span>
            <span class="agent-badge" style="background: rgba(255,255,255,0.2);">💪 Wellness Advisor</span>
            <span class="agent-badge" style="background: rgba(255,255,255,0.2);">💊 Medicine Info</span>
            <span class="agent-badge" style="background: rgba(255,255,255,0.2);">❤️ General Health</span>
        </div>
    </div>
    """)

    # Disclaimer
    gr.HTML("""
    <div class="disclaimer-box">
        ⚕️ <strong>Medical Disclaimer:</strong> This AI assistant provides <em>general health information only</em>. 
        It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        Always consult a qualified healthcare provider for medical concerns. 
        In case of emergency, call your local emergency number immediately.
    </div>
    """)

    # Chatbot
    chatbot = gr.Chatbot(
        height=480,
        placeholder="👋 Hello! I'm your Healthcare AI Assistant. Ask me about symptoms, wellness, medications, or general health topics.",
    )

    # Input row
    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Describe your symptoms, ask about medications, or get wellness advice...",
            container=False,
            scale=8,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1)
        clear_btn = gr.ClearButton([msg, chatbot], value="Clear", scale=1)

    # Example queries
    gr.Examples(
        examples=[[q] for q in EXAMPLE_QUERIES],
        inputs=[msg],
        label="💡 Try these example questions:",
    )

    # Chat logic
    def respond(user_message: str, chat_history: list):
        if not user_message.strip():
            return "", chat_history
        chat_history.append({"role": "user", "content": user_message})
        bot_message = chat_interface(chat_history)
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    # Cloud Run provides the PORT environment variable. Default to 8080 for local testing.
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port, css=CUSTOM_CSS)
