"""Scenario generation and response evaluation via LangChain + OpenAI."""

import os

from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

parser = JsonOutputParser()
llm = ChatOpenAI(model="gpt-4.1-nano", api_key=os.getenv("OPENAI_API_KEY"), temperature=1.0)

scenario_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a voice-based emotional reaction trainer."
            "Generate short realistic one-liner scenarios that provoke an emotional or thoughtful spoken reaction from the player."
            "Avoid specifying any emotion or reaction explicitly."
            "Your response must be valid JSON with fields: `scenario`.",
        ),
        ("user", "Give me one such scenario."),
    ]
)
scenario_chain = scenario_prompt | llm | parser


def generate_scenario():
    return scenario_chain.invoke({})


evaluation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an evaluator in an emotional reaction game."
            "You will receive a scenario and the player's spoken response."
            "Determine whether the response is reasonable and what emotion it conveys."
            'Respond in this JSON format: {"reasonable": true|false, "emotion": "<label>", "explanation": "<brief>"}',
        ),
        ("user", "Scenario: {scenario}\nUser response: {user_response}"),
    ]
)
evaluation_chain = evaluation_prompt | llm | parser


def evaluate_response(scenario: str, user_response: str):
    return evaluation_chain.invoke({"scenario": scenario, "user_response": user_response})
