# 🧠 Long-Term Memory Agent with Contextual Retrieval

<div align="center">

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green.svg?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/status-Active_Development-brightgreen.svg?style=for-the-badge)

</div>

<p align="center">
A sophisticated Python-based agent with advanced memory capabilities, leveraging Anthropic's Contextual Retrieval methodology for persistent, evolving long-term memory management.
</p>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Enhanced Memory System](#-enhanced-memory-system)
- [Components](#-components)
  - [Memory Enhancements](#memory-enhancements)
  - [Scalability Features](#scalability-features)
  - [Robustness Features](#robustness-features)
  - [Evaluation Tools](#evaluation-tools)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Example Applications](#-example-applications)
- [Interactive Demo](#-interactive-demo)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🌟 Overview

This project implements a Long-Term Memory Agent that can:

- 📚 Store and retrieve information from various knowledge sources
- 🎯 Use contextual relevance to prioritize memory retrieval
- 🔄 Provide persistent memory that evolves over time
- 🧩 Support multi-hop reasoning across knowledge units

---

## ✨ Features

### Core Capabilities

- **Progressive Learning**: Build knowledge across multiple interactions
- **Knowledge Correction**: Update understanding when given corrections
- **Multi-hop Reasoning**: Connect different knowledge pieces
- **Cross-referencing**: Establish relationships between related concepts
- **Memory Consolidation**: Organize and merge related information

### Advanced Functionality

- ✅ Knowledge graph representation for memory connections
- ✅ Enhanced contextualizer for improved relationship detection
- ✅ Vector storage with Chroma and Pinecone integrations
- ✅ Memory pruning for efficient knowledge management
- ✅ Batch processing for optimized vector operations
- ✅ Distributed memory management with caching
- ✅ Multi-modal knowledge support
- ✅ Adaptive embedding system with domain-specific models
- ✅ Comprehensive monitoring and telemetry
- ✅ Resilience and failure recovery mechanisms
- ✅ Background maintenance for automatic optimization

---

## 🧠 Enhanced Memory System

The latest release includes major improvements to core memory functionality:

<div align="center">
<table>
<tr>
<td align="center"><b>🔄 Progressive Learning</b></td>
<td align="center"><b>🔍 Knowledge Correction</b></td>
</tr>
<tr>
<td>Build knowledge across interactions</td>
<td>Update understanding with feedback</td>
</tr>
<tr>
<td align="center"><b>🔗 Multi-hop Reasoning</b></td>
<td align="center"><b>📊 Cross-referencing</b></td>
</tr>
<tr>
<td>Connect different knowledge pieces</td>
<td>Establish relationships between concepts</td>
</tr>
<tr>
<td align="center"><b>📚 Memory Consolidation</b></td>
<td align="center"><b>🛠️ Contradiction Resolution</b></td>
</tr>
<tr>
<td>Organize related information</td>
<td>Detect and resolve conflicting knowledge</td>
</tr>
</table>
</div>

---

## 🧩 Components

### Memory Enhancements

- **🕸️ Knowledge Graph**: Represents relationships between knowledge units
  - `knowledge_graph.py`: Graph-based memory representation
  - Support for various relationship types
  - Bidirectional traversal capabilities

- **🔍 Enhanced Contextualizer**: Improved context retrieval
  - `enhanced_contextualizer.py`: Better relationship detection
  - Contradiction detection and resolution
  - Context scoring based on multiple relevance factors

- **🧠 Enhanced Memory Manager**: Advanced memory management
  - `enhanced_manager.py`: Knowledge graph integration
  - Multi-hop reasoning capabilities
  - Progressive learning with context accumulation

### Scalability Features

- **💾 Persistent Vector Stores**: Long-term storage
  - `persistent_store.py`: Adapters for Chroma and Pinecone
  - Consistent interface with in-memory stores
  - Support for metadata filtering

- **✂️ Memory Pruning**: Efficient knowledge management
  - `pruning.py`: Strategies for removing outdated or redundant knowledge
  - Age-based, relevance-based, and redundancy-based pruning
  - Archival capabilities for important but infrequently used knowledge

- **⚡ Batch Processing**: Performance optimization
  - `batch_processor.py`: Efficient vector operations
  - Parallel processing capabilities
  - Queue-based operation management

- **🌐 Distributed Memory Manager**: Scaling to large knowledge bases
  - `distributed_manager.py`: Distributed caching
  - Sharding strategies
  - Load balancing

- **🔄 Multi-Modal Knowledge**: Support for various data types
  - `multimodal_store.py`: Text, images, and more
  - Modal-specific embedding strategies
  - Cross-modal retrieval capabilities

### Robustness Features

- **🧮 Adaptive Embeddings**: Dynamic model selection
  - `adaptive_embeddings.py`: Domain-specific embedding models
  - Content-aware model selection
  - Fallback mechanisms
  - Embedding caching

- **📊 Monitoring and Telemetry**: Performance tracking
  - `monitoring.py`: Comprehensive metrics collection
  - Performance bottleneck detection
  - Usage pattern analysis
  - Alerting capabilities

- **🔄 Resilience and Recovery**: Fault tolerance
  - `resilience.py`: Failure detection and diagnosis
  - Automatic recovery strategies
  - Circuit breaker pattern
  - Backup and restore mechanisms

- **🔧 Background Maintenance**: Automatic optimization
  - `maintenance.py`: Scheduled maintenance tasks
  - Memory consolidation
  - Database optimization
  - Integrity checking

### Evaluation Tools

- **🧪 Multi-Step Reasoning Tests**: Evaluate memory connections
  - `test_multi_step_reasoning.py`: Progressive learning evaluation
  - Relationship detection testing
  - Complex query scenarios

- **⚠️ Contradiction Detection Tests**: Evaluate coherence
  - `test_contradiction_detection.py`: Conflicting information handling
  - Resolution strategy testing
  - Preference configuration

- **📏 Benchmark Suite**: Performance evaluation
  - `test_benchmark.py`: Throughput and latency measurement
  - Scalability testing
  - Storage efficiency analysis

---

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-contextual-retrieval.git
   cd langchain-contextual-retrieval
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run tests to verify setup**
   ```bash
   pytest
   ```

---

## 📁 Project Structure

```
.
├── src/
│   └── ltm_agent/      # Main package source code
│       ├── __init__.py
│       ├── agent/      # Agent-related code
│       │   └── ltm_agent.py         # Long-Term Memory Agent implementation
│       ├── core/       # Core data structures and utilities
│       │   ├── config.py            # Configuration management
│       │   ├── models.py            # Data models
│       │   └── exceptions.py        # Custom exceptions
│       ├── memory/     # Memory systems
│       │   ├── manager.py           # Memory manager
│       │   ├── vector_store.py      # Vector store
│       │   ├── knowledge_graph.py   # Graph-based memory
│       │   ├── enhanced_contextualizer.py # Improved context retrieval
│       │   ├── enhanced_manager.py  # Advanced memory management
│       │   ├── persistent_store.py  # Long-term storage
│       │   ├── pruning.py           # Memory pruning system
│       │   ├── batch_processor.py   # Batch processing system
│       │   ├── distributed_manager.py # Distributed memory management
│       │   ├── multimodal_store.py  # Multi-modal knowledge support
│       │   ├── adaptive_embeddings.py # Dynamic embedding model selection
│       │   ├── monitoring.py        # Monitoring and telemetry system
│       │   ├── resilience.py        # Resilience and failure recovery
│       │   └── maintenance.py       # Background maintenance
│       └── utils/      # Utility functions
│           └── helpers.py           # Helper utilities
├── tests/              # Test suite
│   ├── __init__.py
│   ├── test_agent.py               # Agent tests
│   ├── test_memory.py              # Memory system tests
│   ├── test_vector_store.py        # Vector store tests
│   ├── test_multi_step_reasoning.py # Memory connections tests
│   ├── test_contradiction_detection.py # Coherence tests
│   └── test_benchmark.py           # Benchmark tests
├── examples/           # Example applications
│   ├── simple_demo.py              # Basic agent demo
│   ├── scalable_memory_example.py  # Scalability features demo
│   ├── minimal_scalable_demo.py    # Minimal scalability demo
│   ├── simple_enhanced_retrieval.py # Enhanced retrieval demo
│   └── resilient_memory_system.py  # Full-featured resilient system demo
├── scripts/            # Utility scripts
│   ├── interactive_demo.py         # Interactive demo application
│   └── enhanced_interactive_demo.py # Enhanced demo with all features
├── memory/             # Enhanced memory system implementation
│   ├── memory_unit.py              # Memory unit implementation
│   ├── knowledge_graph.py          # Knowledge graph implementation
│   ├── enhanced_memory_system.py   # Combined enhanced system
│   └── simple_enhanced_memory.py   # Simplified implementation
├── .env.example        # Environment variables template
├── pyproject.toml      # Project metadata and tool configurations
└── README.md           # This file
```

---

## 🚀 Example Applications

<div align="center">
<table>
<tr>
<th>Example</th>
<th>Description</th>
</tr>
<tr>
<td><b>📊 Scalable Memory</b><br><code>examples/scalable_memory_example.py</code></td>
<td>Demonstrates all scalability features with performance metrics visualization</td>
</tr>
<tr>
<td><b>🔍 Minimal Scalable Demo</b><br><code>examples/minimal_scalable_demo.py</code></td>
<td>Focused approach with simplified integration pattern</td>
</tr>
<tr>
<td><b>⚡ Batch Processing</b><br><code>batch_demo.py</code></td>
<td>Demonstrates batch processing performance with comparative benchmarks</td>
</tr>
<tr>
<td><b>🛡️ Resilient Memory System</b><br><code>examples/resilient_memory_system.py</code></td>
<td>Full suite of enhancements with failure simulation and recovery</td>
</tr>
</table>
</div>

---

## 🎮 Interactive Demo

Try the enhanced memory system with your own inputs:

```bash
python scripts/enhanced_interactive_demo.py
```

> 💡 **Tip**: The demo lets you experience all five key memory system improvements!

### Demo Commands

| Command | Description |
|---------|-------------|
| `question` | Ask any question to test the agent's knowledge |
| `correct: [text]` | Provide correction to the previous response |
| `knowledge` | View all knowledge units stored in the system |
| `debug` | Toggle debug mode to show/hide detailed knowledge content |
| `exit` or `quit` | End the session |

### Demo Experiments

<details>
<summary><b>🔄 Testing Progressive Learning</b></summary>

1. Ask a basic question about a topic (e.g., "What is Python?")
2. Follow up with more specific questions on the same topic
3. Type `knowledge` to see how information accumulates
</details>

<details>
<summary><b>🔍 Testing Knowledge Correction</b></summary>

1. Ask a question to get an initial response
2. Provide a correction by typing: `correct: [your better information]`
3. Ask the same or similar question again
4. Type `knowledge` to see the correction units created
</details>

<details>
<summary><b>📊 Testing Cross-Referencing</b></summary>

1. Ask questions across related domains
2. See how later responses incorporate information from different questions
3. Type `knowledge` to see the relationships between knowledge units
</details>

---

## 📚 Documentation

For detailed information about the memory system improvements, check:

- [`docs/memory_system_improvements.md`](docs/memory_system_improvements.md) - Technical details and implementation notes
- [`docs/installation_guide.md`](docs/installation_guide.md) - Integrating these improvements into your own projects
- [`tests/memory_system/README.md`](tests/memory_system/README.md) - Memory system test documentation

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
<p>
<img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?style=for-the-badge&logo=python&logoColor=white">
</p>
<p> 2025 Synapse Initiative Team</p>
</div>
