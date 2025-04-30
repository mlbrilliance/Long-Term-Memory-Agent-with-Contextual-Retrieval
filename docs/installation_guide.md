# Enhanced Memory System Installation & Integration Guide

This guide provides step-by-step instructions for installing, configuring, and integrating the Enhanced Long-Term Memory (LTM) Agent system into your projects.

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Basic Usage](#3-basic-usage)
4. [Advanced Integration](#4-advanced-integration)
5. [Monitoring and Maintenance](#5-monitoring-and-maintenance)
6. [Troubleshooting](#6-troubleshooting)

## 1. Installation

### 1.1 Prerequisites

Before installing the LTM Agent, ensure your system meets the following requirements:

- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- Access permissions to create and modify files

### 1.2 Install from Source

```bash
# Clone the repository
git clone https://github.com/your-org/langchain-contextual-retrieval.git
cd langchain-contextual-retrieval

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 1.3 Install using pip

```bash
# Install from PyPI
pip install ltm-agent

# Or install a specific version
pip install ltm-agent==1.2.0
```

## 2. Configuration

### 2.1 Basic Configuration

Create a copy of the default configuration file:

```bash
cp config/memory_system_config.yaml my_config.yaml
```

Adjust the settings in `my_config.yaml` according to your needs. The most important settings to consider are:

- `agent.knowledge_limit`: Number of knowledge units to retrieve (min 5 recommended)
- `vector_store.embedding_model`: Embedding model to use
- `correction.priority_boost`: Boost factor for correction units

### 2.2 Environment Variables

Some components require environment variables to be set. Create a `.env` file based on the provided example:

```bash
cp .env.example .env
```

Edit the `.env` file to set:

```
VECTOR_DB_URI=your_vector_db_connection_string
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI models
```

### 2.3 Vector Store Setup

The LTM Agent requires a vector database for knowledge storage. Choose one of the supported options:

#### Option A: In-Memory Vector Store (Development/Testing)

No additional setup required. Use the following configuration:

```yaml
vector_store:
  type: "InMemoryVectorStore"
  collection_name: "ltm_agent_memory"
```

#### Option B: Chroma (Local Persistent Storage)

```bash
# Install Chroma dependencies
pip install chromadb

# Configure in YAML
vector_store:
  type: "ChromaMemoryStore"
  collection_name: "ltm_agent_memory"
  persist_directory: "./chroma_db"
```

#### Option C: Cloud Vector Database

Follow the specific setup instructions for your chosen cloud vector database and update the configuration accordingly.

## 3. Basic Usage

### 3.1 Minimal Example

```python
import asyncio
from ltm_agent import LongTermMemoryAgent, MemoryManager
from ltm_agent.memory import InMemoryVectorStore, SimpleContextualizer

async def main():
    # Initialize components
    vector_store = InMemoryVectorStore(collection_name="my_memory")
    contextualizer = SimpleContextualizer()

    # Create memory manager
    memory_manager = MemoryManager(
        memory_store=vector_store,
        contextualizer=contextualizer
    )

    # Create LTM Agent
    agent = LongTermMemoryAgent(
        memory_manager=memory_manager,
        contextualizer=contextualizer,
        knowledge_limit=5,
        enable_consolidation=True
    )

    # Interact with the agent
    response = await agent.async_invoke("What is Python?")
    print(f"Response: {response}")

    # Provide feedback/correction
    await agent.process_feedback(
        "Python is a high-level, interpreted programming language known for its readability and versatility."
    )

    # Ask again to see the correction applied
    response2 = await agent.async_invoke("Tell me about Python")
    print(f"Updated response: {response2}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3.2 Loading from Config File

```python
import asyncio
from ltm_agent import LongTermMemoryAgent, load_from_config

async def main():
    # Load agent from config file
    agent = await load_from_config("path/to/my_config.yaml")

    # Use the agent
    response = await agent.async_invoke("What is Python?")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Advanced Integration

### 4.1 Integrating with External LLMs

By default, the system uses OpenAI GPT models, but you can use any LLM that follows the interface:

```python
from ltm_agent import LongTermMemoryAgent
from ltm_agent.memory import MemoryManager, SimpleContextualizer

# Custom LLM example (must have an "agenerate" method)
class CustomLLM:
    async def agenerate(self, prompt, **kwargs):
        # Generate response using your custom LLM
        response_text = "Your custom LLM response"

        # Return in the expected format
        class Response:
            def __init__(self, content):
                self.content = content

        return Response(response_text)

# Create agent with custom LLM
agent = LongTermMemoryAgent(
    memory_manager=memory_manager,
    contextualizer=contextualizer,
    llm=CustomLLM(),
    knowledge_limit=5
)
```

### 4.2 Custom Contextualizer

Implement a custom contextualizer to control how knowledge is contextualized:

```python
from ltm_agent.memory.interfaces import BaseContextualizer

class CustomContextualizer(BaseContextualizer):
    async def contextualize(self, content, context=None):
        # Add your custom contextualization logic
        enhanced_content = f"[ENHANCED] {content}"
        return enhanced_content

    async def decontextualize(self, contextualized_content):
        # Add your custom decontextualization logic
        original_content = contextualized_content.replace("[ENHANCED] ", "")
        return original_content

# Use custom contextualizer
agent = LongTermMemoryAgent(
    memory_manager=memory_manager,
    contextualizer=CustomContextualizer(),
    knowledge_limit=5
)
```

### 4.3 Advanced Correction Mechanisms

To maximize correction effectiveness, implement custom correction detection:

```python
from ltm_agent import LongTermMemoryAgent

class EnhancedCorrectionAgent(LongTermMemoryAgent):
    async def process_feedback(self, feedback):
        # Analyze feedback to detect if it's a correction
        is_correction = self._detect_correction(feedback)

        if is_correction:
            # Apply enhanced correction processing
            await self._process_as_correction(feedback)
        else:
            # Process as standard feedback
            await super().process_feedback(feedback)

    def _detect_correction(self, feedback):
        # Custom correction detection logic
        correction_markers = ["actually", "incorrect", "wrong", "instead"]
        return any(marker in feedback.lower() for marker in correction_markers)

    async def _process_as_correction(self, feedback):
        # Enhanced correction processing
        # Implement your custom logic here
        # ...

        # Then call the parent method
        await super().process_feedback(feedback)
```

## 5. Monitoring and Maintenance

### 5.1 Using the Monitoring Dashboard

We provide a simple dashboard to monitor correction system performance:

```bash
# Run the dashboard with default settings
python scripts/correction_monitor.py

# Use custom metrics file
python scripts/correction_monitor.py --metrics-file path/to/metrics.json

# Save visualization to file
python scripts/correction_monitor.py --output correction_performance.png

# Text-only mode (for terminals without graphics)
python scripts/correction_monitor.py --text-only
```

### 5.2 Memory Management

Periodically check and optimize your memory storage:

```python
async def perform_maintenance(agent):
    # Get current memory stats
    memory_count = await agent.memory_manager.get_memory_count()
    print(f"Current memory count: {memory_count}")

    # Perform consolidation
    if hasattr(agent, "consolidate_memory"):
        print("Consolidating memory...")
        await agent.consolidate_memory()

    # Check memory count after consolidation
    memory_count_after = await agent.memory_manager.get_memory_count()
    print(f"Memory count after consolidation: {memory_count_after}")
```

### 5.3 Backup and Migration

Regularly back up your vector store data:

```bash
# Chroma DB backup
cp -r ./chroma_db ./chroma_db_backup_$(date +%Y%m%d)

# Export knowledge to portable format (if implement export functionality)
python scripts/export_knowledge.py --output knowledge_export.json
```

## 6. Troubleshooting

### 6.1 Correction Not Applied

If corrections aren't being reflected in responses:

1. Check correction detection:
   ```python
   # Add debug logging to feedback processing
   feedback = "This is a correction..."
   is_correction = any(marker in feedback.lower() for marker in [
       "actually", "instead", "correction", "wrong"
   ])
   print(f"Detected as correction: {is_correction}")
   ```

2. Check retrieval context:
   ```python
   # Log what's being retrieved for similar queries
   retrieved_units = await memory_manager.get_related_knowledge(query)
   print(f"Retrieved {len(retrieved_units)} units for query: {query}")
   for unit, score in retrieved_units:
       print(f"- {unit.unique_id} (score: {score:.4f}): {unit.metadata.get('type', 'unknown')}")
   ```

3. Try increasing the correction priority boost:
   ```python
   # In config.yaml
   correction:
     priority_boost: 2.0  # Increase from default 1.5
   ```

### 6.2 Performance Issues

If experiencing slow performance:

1. Check vector store configuration:
   ```python
   # Use smaller embedding model
   vector_store = InMemoryVectorStore(
       collection_name="memory",
       embedding_model="paraphrase-MiniLM-L3-v2"  # Smaller, faster model
   )
   ```

2. Reduce knowledge limit:
   ```python
   agent = LongTermMemoryAgent(
       memory_manager=memory_manager,
       knowledge_limit=3  # Reduce from 5
   )
   ```

3. Disable consolidation for high-throughput scenarios:
   ```python
   agent = LongTermMemoryAgent(
       memory_manager=memory_manager,
       enable_consolidation=False  # Disable automatic consolidation
   )
   ```

### 6.3 Import Errors

If encountering import errors:

```bash
# Check installation
pip show ltm-agent

# Ensure you're in the right directory
cd langchain-contextual-retrieval

# Try reinstalling
pip install -e .
```

---

## Additional Resources

- [API Reference Documentation](./api_reference.md)
- [Memory System Improvements Documentation](./memory_system_improvements.md)
- [Configuration Reference](./configuration_reference.md)

For further assistance, please file an issue on the GitHub repository.

---

*Last updated: April 28, 2025*
