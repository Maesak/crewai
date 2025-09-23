import os
import warnings
from typing import List
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

try:
    from crewai import Agent, Task, Crew
    from crewai.llm import LLM
    from transformers import pipeline
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.schema import Document
    import torch
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install crewai transformers langchain-community langchain-huggingface faiss-cpu torch")
    exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# class Config:
#     """Configuration settings"""
    
#     HUGGINGFACE_TOKEN = None  # DO NOT HARDCODE

#     LLM_MODEL = "distilgpt2"
#     EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
#     MAX_LENGTH = 100
#     MAX_NEW_TOKENS = 30
#     TEMPERATURE = 0.3
#     TOP_P = 0.8
#     RETRIEVAL_K = 2
    
    @classmethod
    def setup_environment(cls):
        """Setup environment variables"""
        token = os.getenv('HUGGINGFACE_TOKEN')
        if not token:
            raise ValueError("Please set HUGGINGFACE_TOKEN as an environment variable")
        os.environ['HUGGINGFACE_TOKEN'] = token
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['OPENAI_API_KEY'] = 'dummy-key'

# =============================================================================
# CUSTOM LLM WRAPPER
# =============================================================================

class HuggingFaceLLM(LLM):
    def __init__(self, model_name: str = None, token: str = None, **kwargs):
        self.model_name = model_name or Config.LLM_MODEL
        self.token = token or os.getenv('HUGGINGFACE_TOKEN')
        super().__init__(model="custom-hf-model", **kwargs)

        print(f"Loading model: {self.model_name}")
        try:
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model {self.model_name}: {e}")
            self.generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2", device=-1)
            print("✅ Fallback GPT2 loaded")

    def _generate_response(self, prompt: str) -> str:
        try:
            if not prompt.strip().endswith(("Answer:", "Response:")):
                prompt += "\nAnswer:"
            outputs = self.generator(
                prompt,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                temperature=Config.TEMPERATURE,
                top_p=Config.TOP_P,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                return_full_text=False
            )
            generated_text = outputs[0]['generated_text'] if outputs else "No answer."
            return generated_text.strip()
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return "Error generating response."

# =============================================================================
# RAG KNOWLEDGE BASE
# =============================================================================

class RAGKnowledgeBase:
    def __init__(self, embedding_model: str = None):
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs={'device': 'cpu'})
        self.documents = self._create_sample_documents()
        self.vectorstore = FAISS.from_documents(self.documents, self.embeddings)

    def _create_sample_documents(self) -> List[Document]:
        docs_data = [
            "The Eiffel Tower is located in Paris, France. It was built in 1889 and is 330 meters tall.",
            "The Great Wall of China stretches over 13,000 miles across northern China.",
            "The Taj Mahal is in Agra, India. It was built between 1631 and 1648.",
            "Mount Everest is 8,848 meters tall and located between Nepal and Tibet.",
            "The Amazon Rainforest covers much of the Amazon Basin in South America.",
            "The Sahara Desert covers about 9 million square kilometers in North Africa.",
            "The Pacific Ocean covers about 46% of Earth's water surface.",
            "Tokyo is the capital of Japan with over 37 million people.",
            "The Roman Colosseum was built between 70-80 AD in Rome, Italy.",
            "Aurora Borealis appears in polar regions due to solar wind."
        ]
        return [Document(page_content=content) for content in docs_data]

    def retrieve(self, query: str, k: int = None) -> str:
        k = k or Config.RETRIEVAL_K
        results = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(f"Context {i+1}: {doc.page_content}" for i, doc in enumerate(results))

# =============================================================================
# QA SYSTEM
# =============================================================================

class SimpleQASystem:
    def __init__(self):
        Config.setup_environment()
        self.llm = HuggingFaceLLM()
        self.knowledge_base = RAGKnowledgeBase()

    def answer_question(self, query: str) -> str:
        context = self.knowledge_base.retrieve(query)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return self.llm._generate_response(prompt)

# =============================================================================
# MAIN
# =============================================================================

def main():
    Config.setup_environment()
    qa_system = SimpleQASystem()

    queries = [
        "Where is the Eiffel Tower?",
        "How long is the Great Wall of China?",
        "What is the Taj Mahal?"
    ]

    for query in queries:
        print(f"\nQuestion: {query}")
        print(f"Answer: {qa_system.answer_question(query)}")

if __name__ == "__main__":
    main()
