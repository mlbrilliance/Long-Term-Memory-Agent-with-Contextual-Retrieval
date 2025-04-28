"""
Learning pathway functions for the LTM Agent.

This module provides functions for formatting action results and human feedback
for storage in the agent's memory, enabling the agent to learn from its actions
and from human guidance.
"""

import json
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_action_result(result: Any) -> str:
    """
    Format an action result for storage in the agent's memory.
    
    Args:
        result: The result of an agent action (can be of any type)
        
    Returns:
        str: A formatted string representation of the action result
    """
    timestamp = datetime.now().isoformat()
    formatted_parts = [f"ACTION RESULT: [{timestamp}]"]
    
    # Handle different types of results
    if result is None:
        formatted_parts.append("Result: None")
    elif isinstance(result, str):
        formatted_parts.append(f"Result: {result}")
    elif isinstance(result, dict):
        formatted_parts.append("Result Details:")
        # Format dictionary in a readable way
        for key, value in result.items():
            if isinstance(value, dict):
                formatted_parts.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                formatted_parts.append(f"  {key}:")
                for item in value:
                    formatted_parts.append(f"    - {item}")
            else:
                formatted_parts.append(f"  {key}: {value}")
    elif hasattr(result, "__dict__"):  # Handle custom objects
        formatted_parts.append(f"Result Type: {type(result).__name__}")
        # Extract attributes
        for attr, value in inspect.getmembers(result):
            # Skip private attributes and methods
            if not attr.startswith("_") and not inspect.ismethod(value):
                formatted_parts.append(f"  {attr}: {value}")
    else:
        # Handle primitive types and other cases
        formatted_parts.append(f"Result ({type(result).__name__}): {result}")
    
    return "\n".join(formatted_parts)


def format_human_feedback(feedback: Any) -> str:
    """
    Format human feedback for storage in the agent's memory.
    
    Args:
        feedback: The human feedback (can be of any type)
        
    Returns:
        str: A formatted string representation of the human feedback
    """
    timestamp = datetime.now().isoformat()
    formatted_parts = [f"HUMAN FEEDBACK: [{timestamp}]"]
    
    # Handle different types of feedback
    if feedback is None:
        formatted_parts.append("Feedback: None")
    elif isinstance(feedback, str):
        formatted_parts.append(f"Feedback: {feedback}")
    elif isinstance(feedback, dict):
        formatted_parts.append("Feedback Details:")
        # Format dictionary in a readable way
        for key, value in feedback.items():
            if isinstance(value, dict):
                formatted_parts.append(f"  {key}:")
                for sub_key, sub_value in value.items():
                    formatted_parts.append(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                formatted_parts.append(f"  {key}:")
                for item in value:
                    formatted_parts.append(f"    - {item}")
            else:
                formatted_parts.append(f"  {key}: {value}")
    else:
        # Handle other types
        formatted_parts.append(f"Feedback ({type(feedback).__name__}): {feedback}")
    
    return "\n".join(formatted_parts)
