# Long-Term Memory Agent with Persistent Storage

This project implements a sophisticated agent with long-term memory capabilities, allowing it to:

1. Remember information across multiple sessions
2. Provide contextually relevant answers based on past interactions
3. Learn from user feedback
4. Consolidate related knowledge automatically

## 🚀 Quick Start

### Setup Environment
1. Ensure you have Python 3.8+ installed
2. Verify your `.env` file contains the required API keys:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   PERPLEXITY_API_KEY=your_api_key_here
   ```

### Run the Agent
From the project root directory:
```
python run_agent.py
```

## 💬 Interacting with the Agent

The agent offers two primary interaction modes:

### 1. Questions & Answers
Simply type your question at the prompt and press Enter:
```
Query: What is the capital of France?
```

### 2. Providing Feedback/Information
To teach the agent new information, use the feedback prefix:
```
Query: feedback: The distance from Earth to Moon is approximately 384,400 kilometers.
```

### Ending a Session
To exit the agent:
```
Query: exit
```
or
```
Query: quit
```

## 🗄️ Memory System

This agent uses a persistent storage system based on SQLite, which means:

- **Persistent Knowledge**: Information learned stays in memory between sessions
- **Automatic Consolidation**: The system periodically combines related information
- **Contextual Retrieval**: Answers are generated based on relevant stored knowledge

The persistent database is saved in `persistent_agent_memory.db` in the project root.

## 📊 Comparison with Demo Script

| Feature | run_agent.py | enhanced_interactive_demo.py |
|---------|-------------|------------------------------|
| Storage | Persistent (SQLite) | Temporary (in-memory) |
| Architecture | Full agent | Simplified implementation |
| Memory Features | Consolidation, cross-referencing | Basic retrieval |
| Best For | Production use, long-term learning | Quick testing, demos |

## 🔧 Advanced Configuration

You can modify these parameters in the code for advanced use cases:

- **Knowledge Limit**: Maximum number of memories to retrieve per query
- **Consolidation Interval**: How often to consolidate related memories (default: 24 hours)
- **Embedding Dimension**: Size of vector embeddings (default: 768)

## 📝 Example Interactions

```
Query: What do you know about artificial intelligence?
[Agent responds with known information about AI]

Query: feedback: AI research began formally in 1956 at the Dartmouth Workshop.
[Agent confirms it has learned this new information]

Query: When did AI research begin?
[Agent responds with the information you just provided]
```

## 🛠️ Troubleshooting

- If the agent doesn't seem to remember information, verify the database file exists
- For installation issues, ensure all dependencies are properly installed
- If you see error messages about API keys, check your `.env` file
