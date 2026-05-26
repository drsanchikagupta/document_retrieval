import os
import json
import logging
import pandas as pd
from datasets import Dataset

# Import Ragas metrics and evaluation tools
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics import LLMContextPrecisionWithoutReference

# -----Logging Configuration-----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

RAGAS_EVAL_FILE = "ragas_eval_data.jsonl"

def load_logs_to_dataset(file_path):
    """Reads the append-only JSONL log file and formats it for Ragas."""
    logger.info(f"Reading log data from {file_path}...")
    if not os.path.exists(file_path):
        logger.error(f"Log file not found: {file_path}. Run some queries first via the API!")
        return None

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if not data:
        logger.warning("Log file is empty.")
        return None

    # Convert the JSON rows into a structured Pandas DataFrame
    df = pd.DataFrame(data)
    
    # Ragas strictly expects these 3 columns as input text or text lists
    # Note: Ragas typically requires a 'ground_truth' column for some metrics,
    # but Faithfulness and Answer Relevancy can run without it.
    ragas_dict = {
        "question": df["question"].tolist(),
        "answer": df["answer"].tolist(),
        "contexts": df["contexts"].tolist()
    }
    
    # Convert to a Hugging Face Dataset object required by Ragas
    return Dataset.from_dict(ragas_dict)

def main():
    logger.info("Starting offline Ragas evaluation...")

    # 1. Load your collected logs
    dataset = load_logs_to_dataset(RAGAS_EVAL_FILE)
    if dataset is None:
        return

    logger.info(f"Loaded {len(dataset)} records for evaluation.")

    # 2. Initialize local Ollama models to act as the "Judge"
    # This keeps the evaluation 100% free instead of using OpenAI
    logger.info("Initializing local Ollama judge models...")
    eval_llm = ChatOllama(
        model="llama3",
        model_kwargs={"format": "json"}  # Forces Ollama to strictly speak in valid JSON
    )
    eval_embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # 3. Define which free metrics you want to evaluate
    # - Faithfulness: Checks if the answer stays true to the document (Hallucination check)
    # - Answer Relevancy: Checks if the AI actually addressed the user's prompt
    # - Context Precision: Checks if the FAISS retriever fetched correct information
    context_precision_no_ref = LLMContextPrecisionWithoutReference(llm=eval_llm)

    metrics = [faithfulness, answer_relevancy, context_precision_no_ref]

    logger.info("Running evaluations through local Ollama (this may take a few minutes)...")
    
    # 4. Execute the evaluation suite
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=True  # Stops silent NaN suppression
    )

    # 5. Output Results
    print("\n" + "="*40)
    print("         RAGAS EVALUATION RESULTS       ")
    print("="*40)
    print(result)
    print("="*40)

    # Optional: Save evaluation results out to a CSV for historical tracking
    output_csv = "ragas_summary_scores.csv"
    result_df = result.to_pandas()
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Detailed scores saved successfully to {output_csv}")

if __name__ == "__main__":
    main()
