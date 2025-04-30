"""
Memory Visualizer - Tool for visualizing memory connections and knowledge graphs.

This script provides visualization tools to better understand the connections
between knowledge units and how they form a knowledge graph over time.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltm_agent.memory.manager import MemoryManager

# Configure directories
viz_dir = Path(__file__).parent.parent / "logs" / "visualizations"
viz_dir.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(viz_dir / "memory_visualizer.log"), logging.StreamHandler()],
)
logger = logging.getLogger("memory_visualizer")


class MemoryVisualizer:
    """Visualizes memory connections and knowledge graphs."""

    def __init__(self, output_dir: str | None = None):
        """
        Initialize the memory visualizer.

        Args:
            output_dir: Optional directory to save visualizations
        """
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = viz_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _extract_topics(self, content: str) -> list[str]:
        """
        Extract potential topics from content.

        This is a simple extraction that looks for capitalized words
        and known keywords.

        Args:
            content: The content to extract topics from

        Returns:
            List of potential topics
        """
        # List of common stop words to exclude
        stop_words = {
            "the",
            "of",
            "and",
            "a",
            "to",
            "in",
            "is",
            "you",
            "that",
            "it",
            "he",
            "was",
            "for",
            "on",
            "are",
            "as",
            "with",
            "his",
            "they",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "had",
            "by",
            "not",
            "but",
            "what",
            "all",
            "were",
            "we",
            "when",
            "your",
            "can",
            "said",
            "there",
            "use",
            "an",
            "each",
        }

        # Keywords that are likely topics
        known_keywords = {
            "python",
            "machine learning",
            "data science",
            "neural network",
            "ai",
            "deep learning",
            "reinforcement learning",
            "programming",
            "library",
            "algorithm",
            "data",
            "model",
            "training",
            "javascript",
            "analysis",
            "tensorflow",
            "pytorch",
        }

        words = content.split()
        potential_topics = set()

        # Look for capitalized words that aren't at the start of a sentence
        for i, word in enumerate(words):
            clean_word = word.strip(".,;:!?()[]{}-\"'").lower()

            # Skip stop words
            if clean_word in stop_words or len(clean_word) <= 2:
                continue

            # If it's a known keyword, add it
            if clean_word in known_keywords:
                potential_topics.add(clean_word)

            # If it's capitalized and not at the start of a sentence, it might be a topic
            if word[0].isupper() and i > 0 and words[i - 1][-1] not in ".!?":
                potential_topics.add(clean_word)

        # Look for known bigrams (two-word terms)
        content_lower = content.lower()
        for keyword in known_keywords:
            if " " in keyword and keyword in content_lower:
                potential_topics.add(keyword)

        return list(potential_topics)

    def _create_knowledge_graph(self, knowledge_units: list[dict[str, Any]]) -> nx.Graph:
        """
        Create a graph representation of the knowledge units and their connections.

        Args:
            knowledge_units: List of knowledge units

        Returns:
            NetworkX graph representing the knowledge
        """
        G = nx.Graph()

        # First pass: Add all knowledge units as nodes
        for unit in knowledge_units:
            unit_id = unit["id"]
            content = unit["content"]
            metadata = unit.get("metadata", {})

            # Extract topics from content
            topics = self._extract_topics(content)

            # Create a node for the knowledge unit
            G.add_node(
                unit_id,
                type="knowledge_unit",
                content=content[:50] + "..." if len(content) > 50 else content,
                topics=topics,
                metadata=metadata,
            )

            # Also add topic nodes
            for topic in topics:
                if not G.has_node(topic):
                    G.add_node(topic, type="topic")

                # Connect knowledge unit to topic
                G.add_edge(unit_id, topic, type="has_topic")

        # Second pass: Add connections between knowledge units
        for unit in knowledge_units:
            unit_id = unit["id"]
            metadata = unit.get("metadata", {})

            # Add connections from metadata
            related_units = metadata.get("related_to", [])
            if isinstance(related_units, list):
                for related_id in related_units:
                    if G.has_node(related_id):
                        G.add_edge(unit_id, related_id, type="related_to")

            # Connect units with similar topics
            unit_topics = G.nodes[unit_id].get("topics", [])
            for other_unit in knowledge_units:
                other_id = other_unit["id"]
                if other_id == unit_id:
                    continue

                other_topics = G.nodes[other_id].get("topics", [])

                # Find common topics
                common_topics = set(unit_topics).intersection(set(other_topics))

                # If units share topics, create a common_topic edge
                if common_topics and not G.has_edge(unit_id, other_id):
                    G.add_edge(unit_id, other_id, type="common_topic", topics=list(common_topics))

        return G

    def visualize_knowledge_graph(
        self,
        knowledge_units: list[dict[str, Any]],
        title: str = "Knowledge Graph",
        highlight_topics: list[str] | None = None,
        output_file: str | None = None,
    ) -> str:
        """
        Visualize the knowledge graph.

        Args:
            knowledge_units: List of knowledge units
            title: Title for the visualization
            highlight_topics: Optional list of topics to highlight
            output_file: Optional filename for the output

        Returns:
            Path to the saved visualization
        """
        G = self._create_knowledge_graph(knowledge_units)

        if not G.nodes:
            logger.warning("No nodes in graph, nothing to visualize")
            return None

        # Prepare visualization
        plt.figure(figsize=(12, 10))
        plt.title(title)

        # Node positions
        pos = nx.spring_layout(G, seed=42, k=0.3)

        # Node colors by type
        node_colors = []
        node_sizes = []
        for node in G.nodes:
            node_type = G.nodes[node].get("type", "unknown")
            if node_type == "knowledge_unit":
                node_colors.append("skyblue")
                node_sizes.append(300)
            elif node_type == "topic":
                if highlight_topics and node in highlight_topics:
                    node_colors.append("orange")
                else:
                    node_colors.append("lightgreen")
                node_sizes.append(200)
            else:
                node_colors.append("gray")
                node_sizes.append(100)

        # Edge colors by type
        edge_colors = []
        for u, v, data in G.edges(data=True):
            edge_type = data.get("type", "unknown")
            if edge_type == "related_to":
                edge_colors.append("red")
            elif edge_type == "has_topic":
                edge_colors.append("green")
            elif edge_type == "common_topic":
                edge_colors.append("blue")
            else:
                edge_colors.append("gray")

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=1.5, edge_color=edge_colors, alpha=0.7)

        # Add labels for topics
        topic_labels = {node: node for node in G.nodes if G.nodes[node].get("type") == "topic"}
        nx.draw_networkx_labels(G, pos, labels=topic_labels, font_size=10)

        # Add truncated content for knowledge units
        ku_labels = {
            node: G.nodes[node].get("content", "")[:20] + "..."
            for node in G.nodes
            if G.nodes[node].get("type") == "knowledge_unit"
        }
        nx.draw_networkx_labels(G, pos, labels=ku_labels, font_size=8)

        # Add legend
        legend_elements = [
            Patch(facecolor="skyblue", label="Knowledge Unit"),
            Patch(facecolor="lightgreen", label="Topic"),
            Patch(facecolor="orange", label="Highlighted Topic"),
            Patch(facecolor="red", label="Related To"),
            Patch(facecolor="green", label="Has Topic"),
            Patch(facecolor="blue", label="Common Topic"),
        ]
        plt.legend(handles=legend_elements, loc="upper left")

        # Set margins and turn off axis
        plt.margins(0.1)
        plt.axis("off")

        # Save the visualization
        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"knowledge_graph_{timestamp}.png"

        output_path = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Knowledge graph visualization saved to {output_path}")
        return str(output_path)

    def visualize_topic_network(
        self,
        knowledge_units: list[dict[str, Any]],
        title: str = "Topic Network",
        min_common_units: int = 1,
        output_file: str | None = None,
    ) -> str:
        """
        Visualize the network of topics and their connections.

        Args:
            knowledge_units: List of knowledge units
            title: Title for the visualization
            min_common_units: Minimum number of common units for topics to be connected
            output_file: Optional filename for the output

        Returns:
            Path to the saved visualization
        """
        # Extract topics from all knowledge units
        topics_to_units = {}

        for unit in knowledge_units:
            content = unit["content"]
            unit_id = unit["id"]

            # Extract topics from content
            topics = self._extract_topics(content)

            # Map topics to units
            for topic in topics:
                if topic not in topics_to_units:
                    topics_to_units[topic] = set()
                topics_to_units[topic].add(unit_id)

        # Create a graph of topic connections
        G = nx.Graph()

        # Add topics as nodes
        for topic, units in topics_to_units.items():
            G.add_node(topic, units=len(units), type="topic")

        # Connect topics that appear in the same units
        for topic1 in topics_to_units:
            for topic2 in topics_to_units:
                if topic1 >= topic2:  # Avoid duplicate edges and self-loops
                    continue

                # Find common units
                common_units = topics_to_units[topic1].intersection(topics_to_units[topic2])

                # If topics share sufficient units, connect them
                if len(common_units) >= min_common_units:
                    G.add_edge(
                        topic1, topic2, weight=len(common_units), common_units=len(common_units)
                    )

        if not G.nodes:
            logger.warning("No topics found, nothing to visualize")
            return None

        # Prepare visualization
        plt.figure(figsize=(14, 12))
        plt.title(title)

        # Node positions using force-directed layout
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)

        # Node sizes based on number of units
        node_sizes = [G.nodes[node]["units"] * 100 for node in G.nodes]

        # Edge widths based on weight
        edge_widths = [G[u][v]["weight"] * 0.5 for u, v in G.edges]

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.8)
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Add edge labels for common units
        edge_labels = {(u, v): f"{G[u][v]['common_units']}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        # Add legend for node size
        sizes = sorted(set(G.nodes[node]["units"] for node in G.nodes))
        legend_elements = [
            Patch(facecolor="lightblue", label=f"{size} Units: {size * 100} size") for size in sizes
        ]
        plt.legend(handles=legend_elements, loc="upper left")

        # Set margins and turn off axis
        plt.margins(0.1)
        plt.axis("off")

        # Save the visualization
        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"topic_network_{timestamp}.png"

        output_path = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Topic network visualization saved to {output_path}")
        return str(output_path)

    def visualize_knowledge_growth(
        self,
        knowledge_snapshots: list[dict[str, Any]],
        title: str = "Knowledge Growth Over Time",
        output_file: str | None = None,
    ) -> str:
        """
        Visualize how knowledge grows over time.

        Args:
            knowledge_snapshots: List of knowledge snapshots with timestamp and units
            title: Title for the visualization
            output_file: Optional filename for the output

        Returns:
            Path to the saved visualization
        """
        if not knowledge_snapshots:
            logger.warning("No knowledge snapshots provided, nothing to visualize")
            return None

        # Extract timestamps and counts
        timestamps = []
        unit_counts = []
        topic_counts = []
        connection_counts = []

        for snapshot in knowledge_snapshots:
            timestamp = snapshot.get("timestamp", "")
            units = snapshot.get("units", [])

            # Convert timestamp to datetime if it's a string
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp)
                except:
                    # If parsing fails, use the index as a proxy for time
                    timestamp = len(timestamps)

            # Count units
            unit_count = len(units)

            # Extract topics and connections
            all_topics = set()
            connection_count = 0

            for unit in units:
                # Extract topics
                content = unit.get("content", "")
                topics = self._extract_topics(content)
                all_topics.update(topics)

                # Count connections
                metadata = unit.get("metadata", {})
                related_units = metadata.get("related_to", [])
                if isinstance(related_units, list):
                    connection_count += len(related_units)

            # Record counts
            timestamps.append(timestamp)
            unit_counts.append(unit_count)
            topic_counts.append(len(all_topics))
            connection_counts.append(connection_count)

        # Create visualization
        plt.figure(figsize=(12, 8))
        plt.title(title)

        # Sort by timestamp if timestamps are datetime objects
        if all(isinstance(t, datetime) for t in timestamps):
            sorted_indices = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
            timestamps = [timestamps[i] for i in sorted_indices]
            unit_counts = [unit_counts[i] for i in sorted_indices]
            topic_counts = [topic_counts[i] for i in sorted_indices]
            connection_counts = [connection_counts[i] for i in sorted_indices]

        # Plot counts over time
        plt.plot(
            timestamps, unit_counts, marker="o", linestyle="-", linewidth=2, label="Knowledge Units"
        )
        plt.plot(timestamps, topic_counts, marker="s", linestyle="--", linewidth=2, label="Topics")
        plt.plot(
            timestamps,
            connection_counts,
            marker="^",
            linestyle="-.",
            linewidth=2,
            label="Connections",
        )

        plt.xlabel("Time")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Format x-axis if timestamps are datetime objects
        if all(isinstance(t, datetime) for t in timestamps):
            plt.gcf().autofmt_xdate()

        # Save the visualization
        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"knowledge_growth_{timestamp}.png"

        output_path = self.output_dir / output_file
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

        logger.info(f"Knowledge growth visualization saved to {output_path}")
        return str(output_path)

    def visualize_conversation_knowledge(
        self,
        conversation: list[dict[str, Any]],
        knowledge_snapshots: list[dict[str, Any]],
        title: str = "Knowledge Evolution During Conversation",
        output_file: str | None = None,
    ) -> str:
        """
        Visualize how knowledge evolves during a conversation.

        Args:
            conversation: List of conversation turns
            knowledge_snapshots: List of knowledge snapshots taken during conversation
            title: Title for the visualization
            output_file: Optional filename for the output

        Returns:
            Path to the saved visualization
        """
        if not conversation or not knowledge_snapshots:
            logger.warning("Conversation or knowledge snapshots missing, nothing to visualize")
            return None

        # Count turns in conversation
        user_turns = [turn for turn in conversation if turn.get("role") == "user"]
        num_turns = len(user_turns)

        # Create a multi-panel visualization
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(title, fontsize=16)

        # Panel 1: Knowledge growth during conversation
        ax1 = plt.subplot(2, 1, 1)

        # Extract unit counts from snapshots
        snapshot_labels = []
        unit_counts = []
        topic_counts = []

        for snapshot in knowledge_snapshots:
            stage = snapshot.get("stage", "")
            units = snapshot.get("knowledge", [])

            # Count units
            unit_count = len(units)

            # Extract topics
            all_topics = set()
            for unit in units:
                content = unit.get("content", "")
                topics = self._extract_topics(content)
                all_topics.update(topics)

            # Record counts
            snapshot_labels.append(stage)
            unit_counts.append(unit_count)
            topic_counts.append(len(all_topics))

        # Plot counts by stage
        x = range(len(snapshot_labels))
        ax1.bar(
            [i - 0.2 for i in x], unit_counts, width=0.4, color="skyblue", label="Knowledge Units"
        )
        ax1.bar([i + 0.2 for i in x], topic_counts, width=0.4, color="lightgreen", label="Topics")

        ax1.set_xlabel("Conversation Stage")
        ax1.set_ylabel("Count")
        ax1.set_xticks(x)
        ax1.set_xticklabels(snapshot_labels, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel 2: Topic evolution
        ax2 = plt.subplot(2, 1, 2)

        # Extract topics for each snapshot
        topic_evolution = {}

        for i, snapshot in enumerate(knowledge_snapshots):
            units = snapshot.get("knowledge", [])
            stage = snapshot.get("stage", f"Stage {i + 1}")

            # Count topic occurrences
            topic_counts = {}
            for unit in units:
                content = unit.get("content", "")
                topics = self._extract_topics(content)

                for topic in topics:
                    if topic not in topic_counts:
                        topic_counts[topic] = 0
                    topic_counts[topic] += 1

            # Record counts for this stage
            for topic, count in topic_counts.items():
                if topic not in topic_evolution:
                    topic_evolution[topic] = [0] * len(knowledge_snapshots)
                topic_evolution[topic][i] = count

        # Select top topics by maximum count
        top_topics = sorted(
            topic_evolution.keys(), key=lambda t: max(topic_evolution[t]), reverse=True
        )[
            :10
        ]  # Show top 10 topics

        # Plot topic evolution
        for topic in top_topics:
            ax2.plot(snapshot_labels, topic_evolution[topic], marker="o", linewidth=2, label=topic)

        ax2.set_xlabel("Conversation Stage")
        ax2.set_ylabel("Topic Occurrence Count")
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax2.grid(True, alpha=0.3)

        # Save the visualization
        if not output_file:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_file = f"conversation_knowledge_{timestamp}.png"

        output_path = self.output_dir / output_file
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for suptitle
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Conversation knowledge visualization saved to {output_path}")
        return str(output_path)


async def visualize_memory_manager(memory_manager: MemoryManager) -> dict[str, str]:
    """
    Visualize the current state of a memory manager.

    Args:
        memory_manager: The memory manager to visualize

    Returns:
        Dict with paths to generated visualizations
    """
    visualizer = MemoryVisualizer()

    # Get all knowledge units
    units = await memory_manager.list_knowledge()

    # Format for visualizer
    knowledge_units = []
    for unit in units:
        knowledge_units.append(
            {"id": unit.unique_id, "content": unit.original_chunk, "metadata": unit.metadata or {}}
        )

    # Generate visualizations
    viz_paths = {}

    # Knowledge graph
    kg_path = visualizer.visualize_knowledge_graph(knowledge_units, title="Current Knowledge Graph")
    viz_paths["knowledge_graph"] = kg_path

    # Topic network
    tn_path = visualizer.visualize_topic_network(knowledge_units, title="Current Topic Network")
    viz_paths["topic_network"] = tn_path

    return viz_paths


async def main():
    """Run the memory visualizer as a standalone tool."""
    parser = argparse.ArgumentParser(description="Memory visualizer for LTM agent")
    parser.add_argument("--input", type=str, help="Path to JSON file with knowledge units")
    parser.add_argument("--snapshots", type=str, help="Path to JSON file with knowledge snapshots")
    parser.add_argument("--conversation", type=str, help="Path to JSON file with conversation data")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    args = parser.parse_args()

    visualizer = MemoryVisualizer(output_dir=args.output_dir)

    if args.input:
        # Load knowledge units from file
        with open(args.input) as f:
            data = json.load(f)

        # Handle different possible formats
        knowledge_units = []
        if isinstance(data, list):
            knowledge_units = data
        elif isinstance(data, dict) and "knowledge" in data:
            knowledge_units = data["knowledge"]
        elif isinstance(data, dict) and "units" in data:
            knowledge_units = data["units"]

        # Generate visualizations
        if knowledge_units:
            kg_path = visualizer.visualize_knowledge_graph(
                knowledge_units, title="Knowledge Graph Visualization"
            )
            print(f"Knowledge graph saved to: {kg_path}")

            tn_path = visualizer.visualize_topic_network(
                knowledge_units, title="Topic Network Visualization"
            )
            print(f"Topic network saved to: {tn_path}")

    if args.snapshots:
        # Load knowledge snapshots from file
        with open(args.snapshots) as f:
            snapshots = json.load(f)

        # Handle different possible formats
        knowledge_snapshots = []
        if isinstance(snapshots, list):
            knowledge_snapshots = snapshots
        elif isinstance(snapshots, dict) and "snapshots" in snapshots:
            knowledge_snapshots = snapshots["snapshots"]

        # Generate growth visualization
        if knowledge_snapshots:
            growth_path = visualizer.visualize_knowledge_growth(
                knowledge_snapshots, title="Knowledge Growth Over Time"
            )
            print(f"Knowledge growth visualization saved to: {growth_path}")

    if args.conversation and args.snapshots:
        # Load conversation from file
        with open(args.conversation) as f:
            conversation_data = json.load(f)

        # Handle different possible formats
        conversation = []
        if isinstance(conversation_data, list):
            conversation = conversation_data
        elif isinstance(conversation_data, dict) and "conversation" in conversation_data:
            conversation = conversation_data["conversation"]

        # Load knowledge snapshots if not already loaded
        if "knowledge_snapshots" not in locals():
            with open(args.snapshots) as f:
                snapshots = json.load(f)

            # Handle different possible formats
            knowledge_snapshots = []
            if isinstance(snapshots, list):
                knowledge_snapshots = snapshots
            elif isinstance(snapshots, dict) and "snapshots" in snapshots:
                knowledge_snapshots = snapshots["snapshots"]

        # Generate conversation knowledge visualization
        if conversation and knowledge_snapshots:
            conv_path = visualizer.visualize_conversation_knowledge(
                conversation, knowledge_snapshots, title="Knowledge Evolution During Conversation"
            )
            print(f"Conversation knowledge visualization saved to: {conv_path}")

    # If no arguments provided, show usage
    if not (args.input or args.snapshots or args.conversation):
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
