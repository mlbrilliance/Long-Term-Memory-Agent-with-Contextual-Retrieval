"""
Adaptive embedding system for the Long-Term Memory Agent.

This module provides a dynamic embedding system that can select and adapt
the most appropriate embedding model based on content type, domain, and task.
"""

import hashlib
import logging
import re
import time
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self, name: str, dimensions: int):
        """
        Initialize the embedding model.

        Args:
            name: Name of the embedding model
            dimensions: Dimensions of embeddings produced
        """
        self.name = name
        self.dimensions = dimensions
        self.call_count = 0
        self.total_tokens = 0
        self.total_time = 0.0

    def encode(self, text: str) -> list[float]:
        """
        Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        start_time = time.time()

        # Estimate token count (rough approximation)
        token_count = len(text.split())

        # Call implementation-specific encoding
        result = self._encode_impl(text)

        # Update stats
        elapsed = time.time() - start_time
        self.call_count += 1
        self.total_tokens += token_count
        self.total_time += elapsed

        return result

    def _encode_impl(self, text: str) -> list[float]:
        """
        Implementation-specific encoding logic.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        raise NotImplementedError("Subclasses must implement _encode_impl")

    def get_stats(self) -> dict[str, Any]:
        """
        Get usage statistics for this model.

        Returns:
            Dictionary with model statistics
        """
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0

        return {
            "name": self.name,
            "dimensions": self.dimensions,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "avg_time_per_call": avg_time,
            "tokens_per_second": self.total_tokens / self.total_time if self.total_time > 0 else 0,
        }


class DummyEmbeddingModel(EmbeddingModel):
    """Dummy embedding model for testing."""

    def __init__(self, name: str = "dummy", dimensions: int = 384):
        """
        Initialize the dummy embedding model.

        Args:
            name: Name of the embedding model
            dimensions: Dimensions of embeddings produced
        """
        super().__init__(name, dimensions)

    def _encode_impl(self, text: str) -> list[float]:
        """
        Generate deterministic random embeddings based on text hash.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        # Use text hash as seed for reproducibility
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(text_hash % 2**32)

        # Generate random embedding
        embedding = np.random.normal(0, 1, self.dimensions)

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()


class SentenceTransformerModel(EmbeddingModel):
    """Embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model.

        Args:
            model_name: Name of the sentence transformer model
        """
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            dimensions = self.model.get_sentence_embedding_dimension()

            super().__init__(model_name, dimensions)
            logger.info(f"Loaded SentenceTransformer model: {model_name} ({dimensions}d)")

        except ImportError:
            logger.warning(
                "SentenceTransformer not available. Install with: pip install sentence-transformers"
            )
            # Fall back to dummy model
            super().__init__(model_name, 384)
            self.model = None

    def _encode_impl(self, text: str) -> list[float]:
        """
        Encode text using sentence-transformers.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        if self.model is None:
            # Fall back to dummy encoding
            return DummyEmbeddingModel()._encode_impl(text)

        # Encode with sentence-transformers
        embedding = self.model.encode(text)
        return embedding.tolist()


class OpenAIEmbeddingModel(EmbeddingModel):
    """Embedding model using OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str | None = None):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key
        """
        # Model dimensions mapping
        dimensions_map = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        dimensions = dimensions_map.get(model_name, 1536)
        super().__init__(model_name, dimensions)

        self.api_key = api_key
        self.client = None

        # Try to initialize OpenAI client
        try:
            import openai

            if api_key:
                openai.api_key = api_key

            self.client = openai.OpenAI()
            logger.info(f"Initialized OpenAI embedding model: {model_name}")

        except ImportError:
            logger.warning("OpenAI not available. Install with: pip install openai")

    def _encode_impl(self, text: str) -> list[float]:
        """
        Encode text using OpenAI API.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        if self.client is None:
            # Fall back to dummy encoding
            return DummyEmbeddingModel()._encode_impl(text)

        try:
            # Call OpenAI API
            response = self.client.embeddings.create(model=self.name, input=text)

            # Extract embedding
            embedding = response.data[0].embedding
            return embedding

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Fall back to dummy encoding
            return DummyEmbeddingModel()._encode_impl(text)


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """Embedding model using HuggingFace Transformers."""

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        """
        Initialize the HuggingFace embedding model.

        Args:
            model_name: Name of the HuggingFace model
        """
        # Default dimensions based on common models
        dimensions_map = {
            "distilbert-base-uncased": 768,
            "bert-base-uncased": 768,
            "roberta-base": 768,
            "albert-base-v2": 768,
            "gpt2": 768,
        }

        dimensions = dimensions_map.get(model_name, 768)
        super().__init__(model_name, dimensions)

        self.model = None
        self.tokenizer = None

        # Try to initialize HuggingFace components
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

            logger.info(f"Loaded HuggingFace model: {model_name} on {self.device}")

        except ImportError:
            logger.warning(
                "HuggingFace Transformers not available. Install with: pip install transformers torch"
            )

    def _encode_impl(self, text: str) -> list[float]:
        """
        Encode text using HuggingFace model.

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        if self.model is None or self.tokenizer is None:
            # Fall back to dummy encoding
            return DummyEmbeddingModel()._encode_impl(text)

        try:
            import torch

            # Tokenize and prepare input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use mean of last hidden state as embedding
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.tolist()

        except Exception as e:
            logger.error(f"HuggingFace embedding error: {e}")
            # Fall back to dummy encoding
            return DummyEmbeddingModel()._encode_impl(text)


class DomainDetector:
    """Detector for content domains to guide embedding model selection."""

    def __init__(self):
        """Initialize the domain detector."""
        # Domain definitions with keywords and patterns
        self.domains = {
            "general": {"keywords": ["the", "and", "for", "with", "about"], "patterns": []},
            "code": {
                "keywords": [
                    "function",
                    "class",
                    "variable",
                    "method",
                    "import",
                    "def",
                    "return",
                    "if",
                    "else",
                    "for",
                    "while",
                    "try",
                    "except",
                    "public",
                    "private",
                    "static",
                    "void",
                    "int",
                    "string",
                    "bool",
                    "array",
                    "list",
                ],
                "patterns": [
                    r"```[\s\S]+?```",  # Code blocks
                    r"`[\w.()[\]{}]+`",  # Inline code
                    r"(?:def|class|function|public|private)\s+\w+\s*\(",  # Function/method definitions
                    r"(?:var|let|const|int|bool|string|float)\s+\w+\s*=",  # Variable declarations
                    r"import\s+[\w.{}]+\s+from",  # Import statements
                ],
            },
            "science": {
                "keywords": [
                    "experiment",
                    "data",
                    "analysis",
                    "study",
                    "research",
                    "hypothesis",
                    "theory",
                    "evidence",
                    "observation",
                    "conclusion",
                    "methodology",
                    "results",
                    "findings",
                    "statistical",
                    "scientific",
                ],
                "patterns": [
                    r"\b\d+(?:\.\d+)?\s*(?:kg|g|mg|mm|cm|m|km|s|ms|A|K|mol|J|W|N|Pa|Hz|V)\b",  # Units
                    r"p\s*(?:<|>|=)\s*0\.\d+",  # p-values
                    r"(?:±|\+/-)(?:\s*\d+(?:\.\d+)?)",  # Error margins
                ],
            },
            "medical": {
                "keywords": [
                    "patient",
                    "treatment",
                    "diagnosis",
                    "symptoms",
                    "disease",
                    "clinical",
                    "therapy",
                    "medication",
                    "doctor",
                    "hospital",
                    "surgery",
                    "prescription",
                    "dose",
                    "chronic",
                    "acute",
                    "condition",
                    "pathology",
                    "prognosis",
                ],
                "patterns": [
                    r"\b[A-Z][a-z]+ disease\b",
                    r"\b(?:MRI|CT|EKG|EEG|CBC|WBC|RBC|IV)\b",  # Medical abbreviations
                ],
            },
            "legal": {
                "keywords": [
                    "court",
                    "law",
                    "legal",
                    "legislation",
                    "regulation",
                    "compliance",
                    "contract",
                    "agreement",
                    "statute",
                    "jurisdiction",
                    "plaintiff",
                    "defendant",
                    "attorney",
                    "clause",
                    "provision",
                    "liability",
                ],
                "patterns": [
                    r"(?:Section|§)\s+\d+(?:\.\d+)*",  # Legal sections
                    r"v(?:s|\.)\s+",  # versus in case names
                    r"(?:plaintiff|defendant|appellant|respondent)\s",
                ],
            },
            "financial": {
                "keywords": [
                    "investment",
                    "profit",
                    "loss",
                    "market",
                    "stock",
                    "bond",
                    "asset",
                    "equity",
                    "dividend",
                    "portfolio",
                    "revenue",
                    "expense",
                    "balance",
                    "budget",
                    "fiscal",
                    "financial",
                    "capital",
                    "interest",
                    "tax",
                ],
                "patterns": [
                    r"\$\d+(?:,\d+)*(?:\.\d+)?",  # Currency amounts
                    r"\d+(?:\.\d+)?\s*%",  # Percentages
                    r"(?:Q[1-4]|FY\d{2,4})",  # Fiscal quarters/years
                ],
            },
        }

    def detect_domain(self, text: str) -> tuple[str, float]:
        """
        Detect the most likely domain for the content.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (domain_name, confidence_score)
        """
        scores = {}
        text_lower = text.lower()

        # Calculate scores for each domain
        for domain, features in self.domains.items():
            score = 0

            # Check for keywords
            for keyword in features["keywords"]:
                if keyword.lower() in text_lower:
                    score += 1

            # Check for patterns
            for pattern in features["patterns"]:
                matches = re.findall(pattern, text)
                score += len(matches) * 2  # Patterns are stronger signals

            # Normalize score based on text length
            word_count = len(text_lower.split())
            normalized_score = score / max(1, word_count / 20)

            scores[domain] = normalized_score

        # Find domain with highest score
        best_domain = max(scores.items(), key=lambda x: x[1])

        # If the best score is very low, default to general
        if best_domain[1] < 0.05 and best_domain[0] != "general":
            return "general", scores["general"]

        return best_domain


class AdaptiveEmbeddingManager:
    """
    Manager for adaptively selecting and using embedding models.

    This class provides:
    1. Dynamic selection of embedding models based on content
    2. Caching of embeddings for efficiency
    3. Fallback mechanisms for robustness
    4. Usage statistics and performance monitoring
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the adaptive embedding manager.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.domain_detector = DomainDetector()
        self.embedding_cache = {}
        self.call_count = 0
        self.cache_hits = 0

        # Initialize available models
        self.models = {}
        self._initialize_models()

        # Domain to model mapping
        self.domain_models = {
            "general": "all-MiniLM-L6-v2",
            "code": "flax-sentence-embeddings/st-codesearch-distilroberta-base",
            "science": "allenai/specter",
            "medical": "pritamdeka/S-PubMedBert-MS-MARCO",
            "legal": "nlpaueb/legal-bert-base-uncased",
            "financial": "yiyanghkust/finbert-tone",
        }

        # Use OpenAI as default if available and specified
        if self.config.get("use_openai", False) and "openai" in self.models:
            for domain in self.domain_models:
                self.domain_models[domain] = "text-embedding-ada-002"

    def _initialize_models(self):
        """Initialize available embedding models."""
        # Always add dummy model as fallback
        self.models["dummy"] = DummyEmbeddingModel()

        # Try to add SentenceTransformer models
        try:
            self.models["all-MiniLM-L6-v2"] = SentenceTransformerModel("all-MiniLM-L6-v2")

            # Domain-specific models if specified in config
            if self.config.get("load_domain_models", False):
                self.models["flax-sentence-embeddings/st-codesearch-distilroberta-base"] = (
                    SentenceTransformerModel(
                        "flax-sentence-embeddings/st-codesearch-distilroberta-base"
                    )
                )
                self.models["allenai/specter"] = SentenceTransformerModel("allenai/specter")
        except Exception as e:
            logger.warning(f"Error initializing SentenceTransformer models: {e}")

        # Try to add OpenAI model if API key is provided
        openai_api_key = self.config.get("openai_api_key")
        if openai_api_key:
            try:
                self.models["text-embedding-ada-002"] = OpenAIEmbeddingModel(api_key=openai_api_key)
            except Exception as e:
                logger.warning(f"Error initializing OpenAI model: {e}")

        # Try to add HuggingFace models if specified
        if self.config.get("load_huggingface_models", False):
            try:
                self.models["distilbert-base-uncased"] = HuggingFaceEmbeddingModel(
                    "distilbert-base-uncased"
                )
            except Exception as e:
                logger.warning(f"Error initializing HuggingFace models: {e}")

    def _select_model(self, text: str, domain: str | None = None) -> EmbeddingModel:
        """
        Select the most appropriate model for the content.

        Args:
            text: Text to embed
            domain: Optional explicit domain

        Returns:
            Selected embedding model
        """
        # Use specified domain or detect it
        if domain is None:
            domain, _ = self.domain_detector.detect_domain(text)

        # Get model name for this domain
        model_name = self.domain_models.get(domain, "all-MiniLM-L6-v2")

        # Get actual model or fall back to default
        if model_name in self.models:
            return self.models[model_name]
        elif "all-MiniLM-L6-v2" in self.models:
            return self.models["all-MiniLM-L6-v2"]
        else:
            # Last resort fallback
            return self.models["dummy"]

    def _cache_key(self, text: str, model_name: str) -> str:
        """
        Generate a cache key for an embedding.

        Args:
            text: Text to embed
            model_name: Name of the model

        Returns:
            Cache key
        """
        # Create a hash of the text and model
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{model_name}:{text_hash}"

    def embed(
        self,
        text: str,
        domain: str | None = None,
        force_model: str | None = None,
        use_cache: bool = True,
    ) -> list[float]:
        """
        Generate an embedding for the text.

        Args:
            text: Text to embed
            domain: Optional domain to guide model selection
            force_model: Optional model name to use
            use_cache: Whether to use the embedding cache

        Returns:
            Embedding vector
        """
        self.call_count += 1

        # Select model
        model = None
        if force_model and force_model in self.models:
            model = self.models[force_model]
        else:
            model = self._select_model(text, domain)

        # Check cache if enabled
        if use_cache:
            cache_key = self._cache_key(text, model.name)
            if cache_key in self.embedding_cache:
                self.cache_hits += 1
                return self.embedding_cache[cache_key]

        # Generate embedding
        try:
            embedding = model.encode(text)

            # Cache the result
            if use_cache:
                cache_key = self._cache_key(text, model.name)
                self.embedding_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding with {model.name}: {e}")

            # Fall back to dummy model
            fallback_model = self.models["dummy"]
            return fallback_model.encode(text)

    def get_stats(self) -> dict[str, Any]:
        """
        Get usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        model_stats = {model_name: model.get_stats() for model_name, model in self.models.items()}

        return {
            "call_count": self.call_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.call_count),
            "cache_size": len(self.embedding_cache),
            "models": model_stats,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()

    def get_available_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of model names
        """
        return list(self.models.keys())

    def get_model_for_domain(self, domain: str) -> str:
        """
        Get the model name for a specific domain.

        Args:
            domain: Domain name

        Returns:
            Model name
        """
        return self.domain_models.get(domain, "all-MiniLM-L6-v2")
