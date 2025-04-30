# Long-Term Memory Agent with Contextual Retrieval

A Python-based Langchain Agent utilizing Anthropic's Contextual Retrieval methodology for persistent, evolving long-term memory management within the WindSurf IDE environment.

## Project Overview

This project implements a Long-Term Memory Agent that can:
- Store and retrieve information from various knowledge sources (actions, feedback, corpus)
- Use contextual relevance to prioritize memory retrieval
- Provide persistent memory that evolves over time
- Integrate seamlessly with the WindSurf IDE environment

## Current Status

- Project setup and initialization
- Environment configuration
- Core data structures (KnowledgeUnit)
- Memory interface design
- Base memory implementation
- Vector store implementation
- Memory manager implementation
- Agent integration
- Knowledge graph representation for memory connections
- Enhanced contextualizer for improved relationship detection
- Enhanced memory manager with knowledge graph integration
- Multi-step reasoning evaluation
- Contradiction detection and resolution
- Persistent vector storage with Chroma and Pinecone adapters
- Memory pruning system for efficient knowledge management
- Batch processing system for vector operations
- Distributed memory manager with caching and sharding
- Multi-modal knowledge support
- Adaptive embedding system with domain-specific models
- Comprehensive monitoring and telemetry system
- Resilience and failure recovery mechanisms
- Background maintenance for automatic optimization

## New Components

### Memory Enhancements

- **Knowledge Graph**: Represents relationships between knowledge units
  - `knowledge_graph.py`: Graph-based memory representation
  - Support for various relationship types
  - Bidirectional traversal capabilities

- **Enhanced Contextualizer**: Improved context retrieval
  - `enhanced_contextualizer.py`: Better relationship detection
  - Contradiction detection and resolution
  - Context scoring based on multiple relevance factors

- **Enhanced Memory Manager**: Advanced memory management
  - `enhanced_manager.py`: Knowledge graph integration
  - Multi-hop reasoning capabilities
  - Progressive learning with context accumulation

### Scalability Features

- **Persistent Vector Stores**: Long-term storage
  - `persistent_store.py`: Adapters for Chroma and Pinecone
  - Consistent interface with in-memory stores
  - Support for metadata filtering

- **Memory Pruning**: Efficient knowledge management
  - `pruning.py`: Strategies for removing outdated or redundant knowledge
  - Age-based, relevance-based, and redundancy-based pruning
  - Archival capabilities for important but infrequently used knowledge

- **Batch Processing**: Performance optimization
  - `batch_processor.py`: Efficient vector operations
  - Parallel processing capabilities
  - Queue-based operation management

- **Distributed Memory Manager**: Scaling to large knowledge bases
  - `distributed_manager.py`: Distributed caching
  - Sharding strategies
  - Load balancing

- **Multi-Modal Knowledge**: Support for various data types
  - `multimodal_store.py`: Text, images, and more
  - Modal-specific embedding strategies
  - Cross-modal retrieval capabilities

### Robustness Features

- **Adaptive Embeddings**: Dynamic model selection
  - `adaptive_embeddings.py`: Domain-specific embedding models
  - Content-aware model selection
  - Fallback mechanisms
  - Embedding caching

- **Monitoring and Telemetry**: Performance tracking
  - `monitoring.py`: Comprehensive metrics collection
  - Performance bottleneck detection
  - Usage pattern analysis
  - Alerting capabilities

- **Resilience and Recovery**: Fault tolerance
  - `resilience.py`: Failure detection and diagnosis
  - Automatic recovery strategies
  - Circuit breaker pattern
  - Backup and restore mechanisms

- **Background Maintenance**: Automatic optimization
  - `maintenance.py`: Scheduled maintenance tasks
  - Memory consolidation
  - Database optimization
  - Integrity checking

### Evaluation Tools

- **Multi-Step Reasoning Tests**: Evaluate memory connections
  - `test_multi_step_reasoning.py`: Progressive learning evaluation
  - Relationship detection testing
  - Complex query scenarios

- **Contradiction Detection Tests**: Evaluate coherence
  - `test_contradiction_detection.py`: Contradiction handling
  - Knowledge correction evaluation
  - Conflicting information resolution

## Dependencies

This project uses:
- Python 3.10+
- LangChain and LangChain Core
- Anthropic Claude API
- ChromaDB for vector storage
- Pydantic for data validation
- pytest for testing

## Setup Instructions

1. **Clone the repository**

2. **Set up a virtual environment**
   ```bash
   uv venv
   # Activate the environment (Windows)
   .\.venv\Scripts\activate
   # Activate the environment (Linux/Mac)
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**
   Create a `.env` file based on the `.env.example` template:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

## Project Structure

```
.
├── src/
│   └── ltm_agent/      # Main package source code
│       ├── __init__.py
│       ├── agent/      # Agent-related code
│       │   └── ltm_agent.py         # Long-Term Memory Agent implementation
│       ├── core/       # Core data structures and utilities
│       │   ├── config.py            # Configuration management
│       │   └── models.py            # Data models (KnowledgeUnit)
│       ├── memory/     # Memory management implementation
│       │   ├── interfaces.py        # Memory interfaces
│       │   ├── base.py              # Base memory implementations
│       │   ├── in_memory_store.py   # In-memory vector store
│       │   ├── manager.py           # Basic memory manager
│       │   ├── contextualizer.py    # Basic contextualizer
│       │   ├── knowledge_graph.py   # Graph-based memory representation
│       │   ├── enhanced_contextualizer.py # Improved context retrieval
│       │   ├── enhanced_manager.py  # Advanced memory management
│       │   ├── persistent_store.py  # Persistent vector stores (Chroma, Pinecone)
│       │   ├── pruning.py           # Memory pruning system
│       │   ├── batch_processor.py   # Batch processing system
│       │   ├── distributed_manager.py # Distributed memory management
│       │   ├── multimodal_store.py  # Multi-modal knowledge support
│       │   ├── adaptive_embeddings.py # Dynamic embedding model selection
│       │   ├── monitoring.py        # Monitoring and telemetry system
│       │   ├── resilience.py        # Resilience and failure recovery
│       │   └── maintenance.py       # Background maintenance
│       └── main.py     # Application entry point
├── tests/              # Tests mirroring src structure
│   └── ltm_agent/
│       ├── agent/
│       │   └── test_ltm_agent.py
│       ├── core/
│       │   └── test_models.py
│       └── memory/
│           ├── test_manager.py
│           └── test_vector_store.py
├── scripts/            # Evaluation and benchmark scripts
│   ├── benchmark_runner.py         # Benchmark framework
│   ├── enhanced_evaluation.py      # Enhanced evaluation system
│   ├── test_memory_connections.py  # Basic memory connection tests
│   ├── test_multi_step_reasoning.py # Multi-step reasoning tests
│   ├── test_contradiction_detection.py # Contradiction handling tests
│   └── test_benchmark.py           # Benchmark tests
├── examples/           # Example applications
│   ├── simple_demo.py              # Basic agent demo
│   ├── scalable_memory_example.py  # Scalability features demo
│   ├── minimal_scalable_demo.py    # Minimal scalability demo
│   ├── simple_enhanced_retrieval.py # Enhanced retrieval demo
│   └── resilient_memory_system.py  # Full-featured resilient system demo
├── batch_demo.py       # Batch processing performance demo
├── .env.example        # Environment variables template
└── pyproject.toml      # Project metadata and tool configurations
```

## Example Applications

- **Scalable Memory Example**: `examples/scalable_memory_example.py`
  - Demonstrates all scalability features together
  - Performance metrics visualization
  - Sample knowledge generation

- **Minimal Scalable Demo**: `examples/minimal_scalable_demo.py`
  - Focused approach to demonstrate key enhancements
  - Simplified integration pattern
  - Step-by-step explanation

- **Batch Processing Demo**: `batch_demo.py`
  - Demonstrates batch processing performance
  - Comparative benchmarks
  - Throughput analysis

- **Resilient Memory System**: `examples/resilient_memory_system.py`
  - Demonstrates the full suite of enhancements
  - Interactive testing capabilities
  - Failure simulation and recovery

## Enhanced Memory System

The latest release of the Long-Term Memory Agent includes major improvements to core memory system functionality:

### Key Enhancements

- **Progressive Learning**: Improved ability to build knowledge across multiple interactions
- **Knowledge Correction**: Enhanced detection and processing of corrective feedback
- **Multi-hop Reasoning**: Better connections between knowledge units for complex reasoning
- **Cross-referencing**: More intelligent relationship detection between related concepts
- **Memory Consolidation**: Improved organization of related information for faster retrieval

### Interactive Demo

Try the enhanced memory system with your own inputs using the interactive demo:

```bash
python scripts/interactive_demo.py
```

#### Using the Interactive Demo

The demo lets you experience the key improvements we've made to the memory system:

1. **Ask questions**: Type any question to see how the system responds based on its current knowledge
2. **Provide corrections**: Type `correct: [your correction]` to correct the last response
3. **View knowledge**: Type `knowledge` to see all knowledge units stored in the system
4. **Toggle debug mode**: Type `debug` to show/hide detailed knowledge content
5. **Exit the demo**: Type `exit` or `quit` to end the session

#### Testing Progressive Learning

To test how the system builds knowledge across multiple interactions:
1. Ask a basic question about a topic (e.g., "What is Python?")
2. Follow up with more specific questions on the same topic
3. Type `knowledge` to see how information accumulates

#### Testing Knowledge Correction

To test how the system incorporates corrections:
1. Ask a question to get an initial response
2. Provide a correction by typing: `correct: [your better information]`
3. Ask the same or similar question again to see if the correction is applied
4. Type `knowledge` to see the correction units created

#### Testing Cross-Referencing

To test how the system connects related information:
1. Ask questions across related domains
2. See how later responses incorporate information from different questions
3. Type `knowledge` to see the relationships between knowledge units

### Documentation

For detailed information about the memory system improvements, check:
- `docs/memory_system_improvements.md` - Technical details and implementation notes
- `docs/installation_guide.md` - Integrating these improvements into your own projects

## License

MIT License - See the LICENSE file for details.
