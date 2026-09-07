import os
import re
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parent):
        if (parent / "database").exists():
            return parent
    return Path(__file__).resolve().parents[3]

BASE_DIR = find_project_root()
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class ProductionChatbot:
    def __init__(self):
        self.database_path = BASE_DIR / "database" / "car_sales_prediction_sales.parquet"
        
        # Load data
        if self.database_path.exists():
            self.df = pd.read_parquet(self.database_path)
        else:
            self.df = pd.DataFrame()
            logger.warning("⚠️ Database not found, using empty dataset")
        
        # Load car data
        self.car_data = self._load_car_data()
        
        # Initialize LLM (but don't fail if not available)
        self.use_llm = False
        try:
            from training.llm_training import SalesPromptFormatter, SalesLLMEngine
            self.prompt_formatter = SalesPromptFormatter(dataset_path=self.database_path)
            self.llm_engine = SalesLLMEngine()
            self.use_llm = bool(self.llm_engine and 
                               (hasattr(self.llm_engine, 'hf_client') and self.llm_engine.hf_client is not None or
                                hasattr(self.llm_engine, 'groq_client') and self.llm_engine.groq_client is not None))
            logger.info(f"✅ LLM available: {self.use_llm}")
        except Exception as e:
            logger.warning(f"⚠️ LLM initialization failed: {e}. Using rule-based only.")
        
        # Rule-based patterns
        self.intent_patterns = {
            'price': re.compile(r'\b(price|cost|budget|expensive|cheap|afford|harga|sekitar)\b', re.I),
            'recommend': re.compile(r'\b(recommend|suggest|which car|best|suitable|rekomen|recommended)\b', re.I),
            'sales': re.compile(r'\b(sales|sell|popular|best seller|top)\b', re.I),
            'specs': re.compile(r'\b(specs|specification|engine|horsepower|torque|feature)\b', re.I),
            'compare': re.compile(r'\b(compare|versus|vs|difference)\b', re.I),
            'greeting': re.compile(r'\b(hi|hello|hey|good morning|good afternoon|hai|halo)\b', re.I),
            'help': re.compile(r'\b(help|support|assist|can you help|bantuan)\b', re.I),
        }
        
        self.responses = {
            "price": "💰 I can help with pricing! What's your budget range? (e.g., 200M - 500M IDR)",
            "recommend": "🚗 To recommend the right car, tell me your budget, family size, and preferred body type.",
            "sales": "📊 I can show top-selling cars. Which brand or model interests you?",
            "specs": "🔧 I can provide detailed specs. Which car model do you want to know about?",
            "compare": "⚖️ I can compare up to 3 cars. List the models you want to compare.",
            "greeting": "👋 Hello! How can I help with your car search today?",
            "help": "💡 I can help with: pricing, recommendations, sales data, specs, and comparisons.",
            "default": "🤔 I'm not sure I understand. Try asking about: pricing, recommendations, sales, or specs.",
        }
        
        self.context: Dict[str, Dict[str, Any]] = {}
        logger.info("✅ ProductionChatbot initialized (fallback mode)")
    
    def _load_car_data(self) -> Dict[str, Dict[str, Any]]:
        """Load car data from dataset"""
        if self.df.empty:
            return {}
        
        car_data = {}
        for _, row in self.df.iterrows():
            model = str(row.get("model", "")).strip()
            if model:
                car_data[model.lower()] = {
                    "company": row.get("company", "Unknown"),
                    "model": model,
                    "body_style": row.get("body_style", "Unknown"),
                    "price": row.get("price", 0),
                    "engine": row.get("engine", "Unknown"),
                    "dealer_name": row.get("dealer_name", "Unknown"),
                    "sales": row.get("sales", 0),
                    "quantity": row.get("quantity", 0)
                }
        return car_data
    
    def detect_intent(self, message: str) -> str:
        """Detect user intent"""
        message = message.lower()
        for intent, pattern in self.intent_patterns.items():
            if pattern.search(message):
                return intent
        return 'default'
    
    def extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from message"""
        entities = {}
        message_lower = message.lower()
        
        # Extract budget
        budget_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:m|million|jt|juta|idr|rp)", message_lower, re.I)
        if budget_match:
            entities["budget"] = budget_match.group(1)
        
        # Extract car model
        for model_key in self.car_data:
            if model_key in message_lower:
                entities["car_model"] = self.car_data[model_key]["model"]
                entities["company"] = self.car_data[model_key]["company"]
                break
        
        return entities
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Main chat interface"""
        try:
            intent = self.detect_intent(message)
            entities = self.extract_entities(message)
            response_text = self.responses.get(intent, self.responses['default'])
            
            # Specific responses based on entities
            if entities.get('car_model'):
                model = entities['car_model']
                data = self.car_data.get(model.lower())
                if data:
                    response_text = (
                        f"🚗 {data['company']} {data['model']}\n"
                        f"💰 Price: Rp.{float(data['price']):,.0f}\n"
                        f"🏷️ Body: {data['body_style']}\n"
                        f"🔧 Engine: {data['engine']}\n"
                        f"🏪 Dealer: {data['dealer_name']}"
                    )
            elif intent == 'price' and entities.get('budget'):
                response_text = f"💰 With budget Rp.{entities['budget']}, I can recommend options. What body type do you prefer?"
            
            # Store context
            if session_id:
                if session_id not in self.context:
                    self.context[session_id] = {}
                self.context[session_id]['last_intent'] = intent
                self.context[session_id]['last_entities'] = entities
                self.context[session_id]['last_message'] = message
            
            return {
                'response': response_text,
                'intent': intent,
                'entities': entities,
                'source': 'rule-based' if not self.use_llm else 'llm',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            return {
                'response': "I'm having trouble processing your request. Please try again.",
                'intent': 'error',
                'entities': {},
                'source': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    def get_intents(self) -> Dict:
        """Get available intents"""
        return {
            'intents': list(self.intent_patterns.keys()),
            'description': 'Supported intents for car chat assistant'
        }

# Singleton
_chatbot = None

def get_chatbot() -> ProductionChatbot:
    global _chatbot
    if _chatbot is None:
        _chatbot = ProductionChatbot()
    return _chatbot