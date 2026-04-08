import os
import re
import json
from typing import Dict, List, Tuple, Optional
from src.agents import LLM_Agent, Reasoning_Agent, get_best_agent, MultiAPI_Agent
from src.lean_runner import execute_lean_code

type LeanCode = Dict[str, str]

MAX_ITERATIONS = 3
RAG_K = 3


def load_rag_context(query: str) -> str:
    try:
        from src.embedding_db import VectorDB
        from src.embedding_models import MiniEmbeddingModel

        npy_file = "embeddings.npy"
        if os.path.exists(npy_file):
            model = MiniEmbeddingModel()
            chunks, scores = VectorDB.get_top_k(npy_file, model, query, k=RAG_K)
            return "\n\n---\n\n".join(chunks)
    except Exception as e:
        print(f"RAG not available: {e}")
    return ""


def extract_lean_blocks(text: str) -> Tuple[Optional[str], Optional[str]]:
    code_match = re.search(
        r"-- << CODE START >>\s*(.*?)\s*-- << CODE END >>", text, re.DOTALL
    )
    proof_match = re.search(
        r"-- << PROOF START >>\s*(.*?)\s*-- << PROOF END >>", text, re.DOTALL
    )

    code = code_match.group(1).strip() if code_match else None
    proof = proof_match.group(1).strip() if proof_match else None
    return code, proof


def clean_response(response: str) -> Tuple[str, str]:
    code, proof = extract_lean_blocks(response)
    if code is None:
        code = response.strip()
    if proof is None:
        proof = "sorry"
    return code, proof


def create_planning_prompt(
    problem_description: str, task_lean_code: str, rag_context: str = ""
) -> str:
    return f"""You are a planning agent for Lean 4 code generation. Analyze the problem and create a plan.

## Problem Description
{problem_description}

## Lean Code Template
{task_lean_code}

## Relevant Context (from RAG)
{rag_context}

Your task is to create a detailed plan for implementing the code and proof. Consider:
1. What is the function supposed to do?
2. What is the specification/property to prove?
3. What Lean 4 tactics will be needed for the proof?
4. What is the algorithmic approach for the implementation?

Respond with a clear, concise plan."""


def create_generation_prompt(
    problem_description: str,
    task_lean_code: str,
    plan: str,
    rag_context: str = "",
    use_thinking: bool = False,
) -> List[Dict[str, str]]:
    system_prompt = """You are an expert Lean 4 code generation agent. Your task is to:
1. Generate the implementation code (replacing {{code}} placeholder)
2. Generate the proof (replacing {{proof}} placeholder)

Follow the plan created by the planning agent. Use proper Lean 4 syntax and tactics.
The proof should use tactics like: rfl, simp, rw, exact, apply, cases, induction, etc.
DO NOT use 'sorry' in the proof - generate a real proof.

Important: Your response must use these exact markers:
-- << CODE START >>
<your code here>
-- << CODE END >>

-- << PROOF START >>
<your proof here>
-- << PROOF END >>"""

    user_prompt = f"""## Problem Description
{problem_description}

## Lean Code Template
{task_lean_code}

## Planning
{plan}

## Relevant Context (from RAG)
{rag_context}

Generate the code and proof following the template markers."""

    messages = [{"role": "system", "content": system_prompt}]
    if use_thinking:
        messages.append(
            {
                "role": "user",
                "content": f"<thinking>\n{plan}\n</thinking>\n\n{user_prompt}",
            }
        )
    else:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def create_verification_prompt(
    code: str, proof: str, task_lean_code: str, lean_output: str, error_type: str
) -> str:
    return f"""You are a verification agent. The previous attempt had an error.

## Error Type
{error_type}

## Lean Output
{lean_output}

## Current Code Implementation
{code}

## Current Proof
{proof}

## Original Template
{task_lean_code}

Analyze the error and provide a corrected code and proof. Focus on fixing the specific issue.

Respond with:
-- << CODE START >>
<corrected code>
-- << CODE END >>

-- << PROOF START >>
<corrected proof>
-- << PROOF END >>"""


def planning_agent(
    problem_description: str, task_lean_code: str, agent: MultiAPI_Agent
) -> str:
    rag_context = load_rag_context(problem_description)
    prompt = create_planning_prompt(problem_description, task_lean_code, rag_context)
    messages = [
        {"role": "system", "content": "You are a planning agent."},
        {"role": "user", "content": prompt},
    ]
    return agent.get_response(messages)


def generation_agent(
    problem_description: str,
    task_lean_code: str,
    plan: str,
    agent: MultiAPI_Agent,
    use_thinking: bool = False,
) -> Tuple[str, str]:
    rag_context = load_rag_context(problem_description)
    messages = create_generation_prompt(
        problem_description, task_lean_code, plan, rag_context, use_thinking
    )
    response = agent.get_response(messages)
    return clean_response(response)


def verification_agent(
    code: str,
    proof: str,
    task_lean_code: str,
    lean_output: str,
    error_type: str,
    agent: MultiAPI_Agent,
) -> Tuple[str, str]:
    prompt = create_verification_prompt(
        code, proof, task_lean_code, lean_output, error_type
    )
    messages = [
        {"role": "system", "content": "You are a verification/correction agent."},
        {"role": "user", "content": prompt},
    ]
    response = agent.get_response(messages)
    return clean_response(response)


def verify_code(task_lean_code: str, code: str, proof: str) -> Tuple[bool, str, str]:
    full_code = task_lean_code
    full_code = full_code.replace("{{code}}", code)
    if "{{proof}}" in full_code:
        full_code = full_code.replace("{{proof}}", proof)
    else:
        full_code = full_code.replace("{{proof}}", "sorry")

    output = execute_lean_code(full_code)
    success = "Lean code executed successfully" in output
    return success, output, full_code


def main_workflow(problem_description: str, task_lean_code: str = "") -> LeanCode:
    """Main workflow for the coding agent."""
    agent = get_best_agent()

    plan = planning_agent(problem_description, task_lean_code, agent)

    generated_function_implementation = "sorry"
    generated_proof = "sorry"

    for iteration in range(MAX_ITERATIONS):
        print(f"Iteration {iteration + 1}/{MAX_ITERATIONS}")

        code, proof = generation_agent(
            problem_description,
            task_lean_code,
            plan,
            agent,
            use_thinking=(iteration == 0),
        )

        generated_function_implementation = code
        generated_proof = proof

        success, output, full_code = verify_code(task_lean_code, code, proof)

        if success:
            print(f"Success on iteration {iteration + 1}")
            break

        error_type = "compilation_error" if "error" in output.lower() else "proof_error"

        if iteration < MAX_ITERATIONS - 1:
            code, proof = verification_agent(
                code, proof, task_lean_code, output, error_type, agent
            )
            generated_function_implementation = code
            generated_proof = proof

    return {"code": generated_function_implementation, "proof": generated_proof}


def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    problem_description = ""
    lean_code_template = ""

    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()

    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()

    return problem_description, lean_code_template


def get_unit_tests_from_taskpath(task_path: str) -> List[str]:
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    return unit_tests


def get_task_lean_template_from_taskpath(task_path: str) -> str:
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    return task_lean_template
