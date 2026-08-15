import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from groq import Groq  # Fixed typo: Groqs -> Groq
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class SalesLLMEngine:
    """
    Car Sales LLM Engine - Cloud API Strategy
    Primary: Hugging Face Inference API
    Fallback: Groq API
    """

    def __init__(self):
        # Initialize clients
        self.hf_client = InferenceClient(api_key=HF_TOKEN) if HF_TOKEN else None
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None  # Fixed typo

    def generate_completion(self, user_prompt: str, system_prompt: str) -> str:
        """Attempts generation with HuggingFace; falls back to Groq if HF fails."""

        # --- Step 1: HuggingFace Inference API ---
        if self.hf_client:
            try:
                logger.info("🤗 Using Hugging Face API...")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                response = self.hf_client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.3
                )

                logger.info("✅ Hugging Face API call successful.")
                return response.choices[0].message.content

            except Exception as e:
                logger.error(f"❌ Hugging Face API call failed: {e}")
                logger.info("Attempting fallback to Groq API...")

        else:
            logger.warning("⚠️ Hugging Face API key not found. Skipping to Groq API...")

        # --- Step 2: Groq API Fallback ---
        if self.groq_client:
            try:
                logger.info("🚀 Using Groq API...")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]

                completion = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_completion_tokens=512,
                    temperature=0.3
                )

                logger.info("✅ Groq API call successful.")
                return completion.choices[0].message.content

            except Exception as e:
                logger.error(f"❌ Groq API call failed: {e}")
                raise RuntimeError("❌ Both Hugging Face and Groq API calls failed. Please check your API keys and network connection.") from e

        else:
            raise ValueError("⚠️ Groq API key not found. Please set the GROQ_API_KEY in your .env file.")


class SalesPromptFormatter:
    """Load car sales database to build contextual prompts for the LLM engine."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def get_sample_context(self, limit: int = 5) -> str:
        """Extracts sample sales data to provide context for few-shot learning prompts."""
        if not self.dataset_path.exists():
            return "No inventory database found"

        df = pd.read_parquet(self.dataset_path)
        sample = df.head(limit)

        context_lines = []
        for _, row in sample.iterrows():
            context_lines.append(
                f"- {row.get('company', 'Brand')} {row.get('model', 'Model')} | Price: Rp.{row.get('price', 0):,.0f}"
            )

        return "\n".join(context_lines)


if __name__ == "__main__":
    # 1. Load context from database
    db_file = BASE_DIR / "database" / "car_sales_prediction_sales.parquet"
    formatter = SalesPromptFormatter(dataset_path=db_file)
    inventory_context = formatter.get_sample_context(limit=5)

    # 2. Define system and user prompts
    system_prompt = (
        "You are an expert car sales consultant. "
        f"Here is a sample of available inventory:\n{inventory_context}\n"
        "Recommend vehicles accurately and professionally based on customer budget."
    )
    user_prompt = "I have a budget of Rp. 200,000,000. What cars can you recommend from your inventory?"

    # 3. Instantiate Engine and Run
    engine = SalesLLMEngine()
    response = engine.generate_completion(user_prompt=user_prompt, system_prompt=system_prompt)

    logger.info("\n--- LLM Response ---\n")
    logger.info(response)