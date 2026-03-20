import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithReference, 
    Faithfulness, 
    FactualCorrectness,
    SemanticSimilarity,
    AnswerRelevancy
)

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig


from langchain_community.callbacks import get_openai_callback

def main():
    maritaca_api_key = os.environ.get("MARITACA_API_KEY", "your key here")


    # Fonte: https://docs.maritaca.ai/pt/modelos
    MARITACA_PRICES = {
        "sabia-3": {
            "input": 0.0010 / 1000,  # Custo por token de entrada (prompt)
            "output": 0.0020 / 1000  # Custo por token de saída (completion)
        }
    }
    
    # client openai maritaca
    model_name = "sabia-3"
    maritaca_llm = ChatOpenAI(
        api_key=maritaca_api_key,
        base_url="https://chat.maritaca.ai/api",
        model=model_name,
        temperature=0.0,
        n=1
    )
    
    print("Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    evaluator_llm = LangchainLLMWrapper(maritaca_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    ragas_config = RunConfig(max_workers=4, max_retries=5)
    answer_relevancy_metric = AnswerRelevancy(strictness=1)
    print("Loading generated dataset...")
    try:
        df = pd.read_json("pipeline_outputs.json", orient="records")
    except FileNotFoundError:
        print("Error: pipeline_outputs.json not found.")
        return

    eval_dataset = Dataset.from_pandas(df)
    print(f"Loaded {len(eval_dataset)} rows. Starting RAGAS Evaluation")

    with get_openai_callback() as cb:
        score = evaluate(
            eval_dataset,
            metrics=[
                FactualCorrectness(),
                LLMContextRecall(),
                LLMContextPrecisionWithReference(),
                Faithfulness(),
                SemanticSimilarity(),
                answer_relevancy_metric
            ],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=ragas_config
        )

    results_df = score.to_pandas()
    results_df.to_csv("ragas_evaluation_results.csv", index=False)
    
    print("\nEvaluation complete! Results saved to ragas_evaluation_results.csv")
    

    print("\n--- Maritaca AI API Usage Report ---")
    print(f"Total calls to API: {cb.successful_requests}")
    print(f"Total tokens used: {cb.total_tokens}")
    print(f"  - Prompt tokens: {cb.prompt_tokens}")
    print(f"  - Completion tokens: {cb.completion_tokens}")
    

    price_info = MARITACA_PRICES.get(model_name)
    if price_info:
        cost = (cb.prompt_tokens * price_info["input"]) + (cb.completion_tokens * price_info["output"])
        print(f"\nEstimated cost for '{model_name}': R$ {cost:.6f}")
    else:
        print(f"\nCould not calculate cost for model '{model_name}'. Price not defined.")


if __name__ == "__main__":
    main()