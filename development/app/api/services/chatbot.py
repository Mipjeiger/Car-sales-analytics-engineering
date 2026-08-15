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

    def detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        for intent, pattern in self.intent_patterns.items():
            if pattern.search(message):
                return intent
        return 'default'

    def extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities like budget, company, and model dynamically from real dataset"""
        entities = {}
        message_lower = message.lower()

        # Qualitative Budget Mapping (e.g., "low budget", "cheap")
        if re.search(
            r'\b(low budget|cheap|affordable|murah|entry level)\b', message_lower
        ):
            default_low_budget = int(self.df['price'].quantile(0.25)) if not self.df.empty else 175_000_000
            entities['budget'] = default_low_budget
            entities['budget_tier'] = 'low'

        # Use-Case Extraction ("family trip")
        if re.search(
            r'\b(family|family trip|mudik|liburan|vacation)\b', message_lower
        ):
            entities['use_case'] = 'family'
            entities['preferred_body_style'] = ['SUV', 'Passenger']

        # 2. Extract numeric budget with mandatory currency context or scale unit
        budget_match = re.search(
            r'(?:budget|under|below|max|harga|rp|idr)?\s*[\:\=]?\s*'
            r'(\d+(?:[\.,]\d+)?)\s*'
            r'(juta|milliard|miliar|million|billion|m|k|rb|ribu|b)?\b',
            message_lower
        )

        if budget_match and not 'family' in message_lower and budget_match.group(2):
            try:
                raw_str = budget_match.group(1).replace(',', '').replace('.', '')
                val = float(raw_str)
                unit = budget_match.group(2)

                if unit in ['juta', 'million', 'm']:
                    val *= 1_000_000
                elif unit in ['k', 'rb', 'ribu', 'thousand']:
                    val *= 1_000
                entities['budget'] = int(val)

            except ValueError:
                pass

        # 3. Dynamic Company Matching
        for company in self.companies:
            if re.search(rf'\b{re.escape(company)}\b', message_lower):
                entities['company'] = company
                break

        # 4. Dynamic Model Matching
        for model in self.models:
            if re.search(rf'\b{re.escape(model)}\b', message_lower):
                entities['model'] = model
                break

        return entities

    def _retrieve_db_context(
        self, intent: str, entities: Dict[str, Any]
    ) -> str:
        """Fetch relative database rows to inject into LLM system"""
        if self.df.empty:
            return "No historical sales database context available."

        model = entities.get('model')
        company = entities.get('company')
        budget = entities.get('budget')

        filtered = self.df.copy()

        if intent == 'sales':
            top_sales = (
                self.df.groupby(['company', 'model'])
                .size()
                .reset_index(name='sales_count')
                .sort_values(by='sales_count', ascending=False)
                .head(7)
            )
            return f'Top 7 best-selling models:\n{top_sales.to_string(index=False)}'

        # Handle specs, price, recommend, and general query intents
        if intent in ['specs', 'price', 'recommend', 'default']:
            if model:
                filtered = filtered[filtered['model_clean'] == model]
            elif company:
                filtered = filtered[filtered['company_clean'] == company]

            if budget:
                filtered = filtered[filtered['price'] <= budget]

            if not filtered.empty:
                summary = (
                    filtered[
                        ['company', 'model', 'price', 'engine', 'transmission', 'body_style']
                    ]
                    .drop_duplicates(subset=['company', 'model'])
                    .head(7)
                    .copy()
                )
                # Convert raw floats to readable IDR format (e.g. IDR 150,000,000)
                summary['price'] = summary['price'].apply(lambda x: f"IDR {int(x):,}")
                return f'Matching Car Catalog Context:\n{summary.to_string(index=False)}'

            if budget:
                min_price = int(self.df['price'].min())
                return f'No models found within the budget of IDR {budget:,}. The minimum price in our database is IDR {min_price:,}.'

        return f'General Dataset Summary: {len(self.df)} sales records indexed across {len(self.companies)} brands.'

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
            'You are an expert AI Car Sales Consultant.'
            'Answer the user question concisely using ONLY the provided database context. '
            'Do not invent features or prices. Format output using clean Markdown.\n\n'
            f'### DATABASE CONTEXT:\n{db_context}'
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

    def reset_context(self, session_id: str):
        if session_id in self.context:
            del self.context[session_id]