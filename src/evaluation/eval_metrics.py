# src/evaluation/eval_metrics.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.agent.graph import agente_app
from langchain_core.messages import HumanMessage

# Banco de Casos de Prueba (Golden Dataset para Evaluación)
TEST_CASES = [
    {
        "id": "TC-01",
        "input": "I need to buy 2 standard laptops for 3,600 USD total. Vendors are SystemCo and TechSys. Purpose: equipment renewal.",
        "expected_procedure": "Direct Award",
        "expected_grounded_term": "Article 3"
    },
    {
        "id": "TC-02",
        "input": "Purchase of 100 laptops for 600,000 USD. Vendor: TechSolutions SA.",
        "expected_procedure": "Open Public Tender",
        "expected_grounded_term": "Open Public Tender"
    }
]

def run_evaluation():
    print("🚀 Running Automated Evaluation Suite for Capstone...\n")
    correct_classifications = 0
    grounded_responses = 0
    total_tests = len(TEST_CASES)

    config = {"configurable": {"thread_id": "eval_session"}}

    for test in TEST_CASES:
        print(f"Running Test Case: {test['id']}...")
        result = agente_app.invoke({"messages": [HumanMessage(content=test["input"])]}, config)
        
        final_output = ""
        for m in result["messages"]:
            if m.content:
                final_output += str(m.content)

        # 1. Evaluación de Clasificación (Accuracy)
        if test["expected_procedure"].lower() in final_output.lower():
            correct_classifications += 1
            print(f"  -> Classification: PASSED")
        else:
            print(f"  -> Classification: FAILED")

        # 2. Evaluación de Fundamentación (Groundedness Rate)
        if test["expected_grounded_term"].lower() in final_output.lower():
            grounded_responses += 1
            print(f"  -> Groundedness: PASSED (Found citation)")
        else:
            print(f"  -> Groundedness: FAILED")
        print("-" * 40)

    # Cálculo de métricas finales
    accuracy = (correct_classifications / total_tests) * 100
    groundedness = (grounded_responses / total_tests) * 100

    print("\n" + "="*40)
    print("📊 FINAL EVALUATION RESULTS:")
    print(f"Classification Accuracy: {accuracy:.1f}%")
    print(f"RAG Groundedness Rate: {groundedness:.1f}%")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()