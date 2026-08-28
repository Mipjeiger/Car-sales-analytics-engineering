from datetime import datetime
import json
import re
import logging
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
from huggingface_hub import InferenceClient
from groq import Groq

logger = logging.getLogger(__name__)

# Load datadir
def find_project_root() -> Path:
        """Find project root by looking for 'database' directory up the path tree"""
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "database").exists():
                return parent
        return Path(__file__).resolve().parents[3]

BASE_DIR = find_project_root()
PARQUET_PATH = BASE_DIR / 'database' / 'car_sales_prediction_sales.parquet'
ENV_DIR = BASE_DIR / '.env'

load_dotenv(dotenv_path=ENV_DIR)

class CarChatbot:
    """Define car chatbot to get LLM answering the question based on real data"""
    def __init__(
        self,
        parquet_path: Optional[Path] = PARQUET_PATH,
        hf_token: Optional[str] = os.getenv("HUGGINGFACE_API_KEY"),
        groq_api_key: Optional[str] = os.getenv("GROQ_API_KEY"),
        hf_model: str = 'meta-llama/Llama-3.1-8B-Instruct',
        groq_model: str = 'llama-3.3-70b-versatile',
    ):
        self.context = {}
        self.hf_model = hf_model
        self.groq_model = groq_model

        # 1. Intialize AI clients
        self.hf_client = InferenceClient(token=hf_token) if (InferenceClient and hf_token) else None
        self.groq_client = Groq(api_key=groq_api_key) if (Groq and groq_api_key) else None

        # 2. Load Real Parquet Data
        try:
            self.df = pd.read_parquet(parquet_path)

            # Normalize strings for case-insensitive lookup
            self.df['company_clean'] = self.df['company'].astype(str).str.lower()
            self.df['model_clean'] = self.df['model'].astype(str).str.lower()
            
        except Exception as e:
            print(f'Warning: Could not load Parquet file ({e}). Initializing empty DataFrame.')
            self.df = pd.DataFrame()

        # 3.  Extract Unique Companies & Models for Dynamic Entity Matching
        self.companies = self.df['company_clean'].unique().tolist() if not self.df.empty else []
        self.models = self.df['model_clean'].unique().tolist() if not self.df.empty else []

        # 4. Regex Patterns for Intent Detection
        self.intent_patterns = {
            'price': re.compile(r'\b(price|cost|budget|expensive|cheap|afford)\b', re.I),
            'recommend': re.compile(r'\b(recommend|suggest|which car|best|suitable|recommendation)\b', re.I,),
            'sales': re.compile(r'\b(sales|sell|popular|best seller|top|upsell)\b', re.I),
            'specs': re.compile(r'\b(specs|specification|engine|transmission|horsepower|torque|feature)\b', re.I,),
            'compare': re.compile(r'\b(compare|versus|vs|difference)\b', re.I),
            'greeting': re.compile(r'\b(hi|hello|hey|good morning|good afternoon)\b', re.I),
        }

    def budget_tier(self, budget: int) -> str:
        """Categorize budget into tiers based on quantiles of the dataset"""
        if self.df.empty or "price" not in self.df.columns:
            return "unknown"

        prices = pd.to_numeric(self.df['price'], errors='coerce').dropna()
        if prices.empty:
            return "unknown"

        q1, q2, q3 = prices.quantile([0.25, 0.5, 0.75])

        if budget <= q1:
            return "low"
        elif budget <= q2:
            return "medium"
        elif budget <= q3:
            return "high"
        else:
            return "high tier"

    def use_case_tier(self, use_case: str) -> str:
        """Categorize use case into tiers based on common patterns"""
        use_case_lower = use_case.lower()

        if re.search(r'\b(family|family trip|mudik|liburan|vacation|holiday)\b', use_case_lower):
            return "family"
        elif re.search(r'\b(fashion|stylish|style|luxury)\b', use_case_lower):
            return "fashion"
        elif re.search(r'\b(sport|racing|performance)\b', use_case_lower):
            return "sport"
        else:
            return "general"

    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        text = str(message).strip()

        priority = [
            "greeting",
            "compare",
            "recommend",
            "specs",
            "sales",
            "price",
        ]

        for intent in priority:
            pattern = self.intent_patterns.get(intent)
            if pattern.search(text):
                return intent

        return "default"

    def _parse_budget(self, message: str) -> Optional[int]:
        """Parse budgets properly from user message, considering currency and scale units"""
        text = message.lower().replace(" ", "")

        match = re.search(
            r"(?:budget|under|below|max|harga|rp|idr)?[:=]?"
            r"(\d+(?:[.,]\d+)?)"
            r"(juta|million|miliar|milliard|billion|m|k|rb|ribu|thousand)\b",
            text,
        )

        if not match:
            return None

        try:
            # Decimal separator is treated as a decimal for unit-based scaling values
            value = float(match.group(1).replace(",", "."))
            unit = match.group(2)

            multiplier = {
                "juta": 1_000_000,
                "million": 1_000_000,
                "m": 1_000_000,
                "miliar": 1_000_000_000,
                "milliard": 1_000_000_000,
                "billion": 1_000_000_000,
                "k": 1_000,
                "rb": 1_000,
                "ribu": 1_000,
                "thousand": 1_000,
            }[unit]

            return int(value * multiplier)

        except (ValueError, KeyError):
            return None

    def extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities like budget, company, and model dynamically from real dataset"""
        entities: Dict[str, Any] = {}
        text = str(message).lower()

        # Budget tier
        if re.search(r"\b(low|cheap|affordable|murah|entry[- ]level)\b", text):
            entities["budget_tier"] = "low"
        elif re.search(r"\b(medium|mid|moderate|menengah)\b", text):
            entities["budget_tier"] = "medium"
        elif re.search(r"\b(high|expensive|luxury|premium|tinggi)\b", text):
            entities["budget_tier"] = "high"

        # Numeric budget
        budget = self._parse_budget(text)
        if budget is not None:
            entities['budget'] = budget
            entities['budget_tier'] = self.budget_tier(budget)

        # Use case tier extraction
        if re.search(r"\b(family|trip|mudik|liburan|vacation|holiday)\b", text):
            entities["use_case_tier"] = "family"
        elif re.search(r"\b(fashion|stylish|style|luxury|nongkrong|chill)\b", text):
            entities["use_case_tier"] = "fashion"
        elif re.search(r"\b(sport|racing|performance|race|competition)\b", text):
            entities["use_case_tier"] = "sport"
        else:
            entities["use_case_tier"] = "general"

        # Match longer name first to avoid partial matches
        for company in sorted(self.companies, key=len, reverse=True):
            if re.search(rf"\b{re.escape(company)}\b", text):
                entities['company'] = company
                break

        for model in sorted(self.models, key=len, reverse=True):
            if re.search(rf"\b{re.escape(model)}\b", text):
                entities['model'] = model
                break

        return entities

    def _retrieve_db_context(
        self, intent: str, entities: Dict[str, Any]
    ) -> str:
        """Fetch relative database rows to inject into LLM system"""
        if self.df.empty:
            return "No historical sales database context available."

        filtered = self.df.copy()

        # Ensure numeric price
        if "price" in filtered.columns:
            filtered['price'] = pd.to_numeric(filtered['price'], errors='coerce')
            filtered = filtered.dropna(subset=['price'])

        # Exact filters
        if entities.get('company') and 'company_clean' in filtered.columns:
            filtered = filtered[filtered['company_clean'] == entities['company']]

        if entities.get('model') and 'model_clean' in filtered.columns:
            filtered = filtered[filtered['model_clean'] == entities['model']]

        # Numeric filters
        if entities.get('budget') and 'price' in filtered.columns:
            filtered = filtered[filtered['price'] <= entities['budget']]

        # Quantile budget filter
        if entities.get('budget_tier') in {'low', 'medium', 'high'} and 'price' in filtered.columns:
            prices = pd.to_numeric(self.df['price'], errors='coerce').dropna()
            q1, q2, q3 = prices.quantile([0.25, 0.5, 0.75])

            tier = entities['budget_tier']

            if tier == 'low':
                filtered = filtered[filtered['price'] <= q1]
            elif tier == 'medium':
                filtered = filtered[(filtered['price'] > q1) & (filtered['price'] <= q2)]
            elif tier == 'high':
                filtered = filtered[filtered['price'] > q2]

        if filtered.empty:
            return "No matching records found in the database for the given criteria."

        # Rank by relevant sales information when exist
        if 'sales' in filtered.columns:
            filtered['sales'] = pd.to_numeric(filtered['sales'], errors='coerce').fillna(0)
            filtered = filtered.sort_values(by='sales', ascending=False)

        columns = [
            col for col in [
                "company",
                "model",
                "price",
                "sales",
                "quantity",
                "engine",
                "transmission",
                "body_style",
            ]
            if col in filtered.columns
        ]

        result = filtered[columns].drop_duplicates(
            subset=[col for col in ['company', 'model'] if col in columns]
        ).head(7).copy()

        if 'price' in result.columns:
            result['price'] = result['price'].map(lambda value: f"IDR {int(value):,}")

        return (
            "These are the only matching records from the database:\n"
            f"{result.to_string(index=False)}"
        )

    def _call_llm(self, prompt: str, system_prompt: str) -> Optional[str]:
        """Call HuggingFace first: if rate-limited or failed, fallback to Groq API"""
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ]

        # 1. Primary HuggingFace Inference API
        if self.hf_client:
            try:
                response = self.hf_client.chat_completion(
                    messages=messages,
                    model=self.hf_model,
                    max_tokens=256,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"❌ HuggingFace API call failed: {e}. Attempting Groq fallback.")

        # 2. Fallback Groq API
        if self.groq_client:
            try:
                response = self.groq_client.chat.completions.create(
                    messages=messages,
                    model=self.groq_model,
                    max_tokens=256,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"❌ Groq API call failed: {e}. No further fallback available.")

    def generate_response(
        self, message: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate dynamic response based on intent and parquet queries"""
        intent = self.detect_intent(message)
        entities = self.extract_entities(message)

        # 1. Fetch relevant ground-truth facts from Parquet DB
        db_context = self._retrieve_db_context(intent, entities)

        # 2. Construct system prompt for LLM
        system_prompt = (
            'You are a car sales assistant.'
            'Answer only using the database context below. '
            'Do not invent cars, prices, specifications, suitability, or sales facts.'
            'If no matching record exists, say so clearly and politely.'
            "Do not claim a car is suitable for racing unless the database supports it."
            'Return concise Markdown.\n\n'
            f'Intent: {intent}\n'
            f'User requirements: {json.dumps(entities)}\n'
            f'Database context:\n{db_context}'
        )

        # 3. Attempt LLM Generation (HF -> Groq)
        llm_response = self._call_llm(message, system_prompt)

        # 4. Fallback to rule-based response if LLMs fail
        if not llm_response:
            llm_response = f"📊 According to our database records:\n{db_context}\n\nHow else can I assist you with these models?"

        return {
            'response': llm_response,
            'intent': intent,
            'entities': entities,
            'timestamp': datetime.now().isoformat(),
        }

    def chat(
        self, message: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main chat handler with session state memory"""
        result = self.generate_response(message, self.context.get(session_id))

        if session_id:
            if session_id not in self.context:
                self.context[session_id] = {}
            self.context[session_id]['last_intent'] = result['intent']
            self.context[session_id]['last_entities'] = result['entities']

        return result

    def get_session_history(self) -> Dict[str, Any]:
        """Return stored context for all active chat sessions"""
        return {
            str(session_id): dict(session_data) for session_id, session_data in self.context.items()
        }

    def reset_context(self, session_id: str):
        if session_id in self.context:
            del self.context[session_id]