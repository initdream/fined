import pandas as pd
import json
from chat_pipeline import run_pipeline_for_evaluation, memory_store

def main():
    df = pd.read_csv("questions.csv")

    results = {
        "user_input": [], 
        "reference": [], 
        "response":[],  
        "retrieved_contexts":[] 
    }

    print(f"Starting pipeline generation for {len(df)} questions...")

    for index, row in df.iterrows():
        question = row['question']
        ground_truth = row['ground_truth']
        
        # clear chat memory
        if hasattr(memory_store, "messages"):
            memory_store.messages =[]

        answer, contexts = run_pipeline_for_evaluation(question)
        
        results["user_input"].append(question)
        results["reference"].append(ground_truth)
        results["response"].append(answer)
        results["retrieved_contexts"].append(contexts)

        print(f"\n{'='*50}")
        print(f"ITERATION {index + 1}")
        print(f"{'='*50}")
        print(f"USER INPUT (Question): \n{question}\n")
        print(f"REFERENCE (Ground Truth): \n{ground_truth}\n")
        print(f"RESPONSE (Ollama Answer): \n{answer}\n")
        
        print(f"RETRIEVED CONTEXTS ({len(contexts)} chunks from Ranker):")
        for i, ctx in enumerate(contexts):
            snippet = ctx.replace('\n', ' ')[:150] 
            print(f"   [Chunk {i+1}] {snippet}...")
        print(f"{'='*50}\n")
        
        print(f"Processed {index + 1}/{len(df)}: {question[:40]}...")

    output_df = pd.DataFrame(results)
    output_df.to_json("pipeline_outputs.json", orient="records", indent=4)
    print("\nDataset generation complete! Saved to pipeline_outputs.json")

if __name__ == "__main__":
    main()