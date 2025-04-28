"""
Tests for learning pathway functions.

This module contains tests for the functions that format action results and
human feedback for storage in the agent's memory.
"""

import pytest
import json
from typing import Dict, Any
from datetime import datetime

# Import the functions we want to test
from ltm_agent.agent.learning import format_action_result, format_human_feedback


class TestLearningFormatters:
    """Tests for learning pathway formatters."""
    
    def test_format_action_result_with_string(self):
        """Test formatting simple string action results."""
        # Simple string result
        result = "The task was completed successfully."
        
        formatted = format_action_result(result)
        
        expected_contents = [
            "ACTION RESULT",  # Should contain this identifier
            result,  # Should contain the original string
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_action_result_with_dict(self):
        """Test formatting dictionary action results."""
        # Dictionary result
        result = {
            "status": "success",
            "data": {
                "id": 123,
                "name": "Example",
                "metadata": {
                    "source": "API",
                    "timestamp": "2025-04-27T12:00:00Z"
                }
            }
        }
        
        formatted = format_action_result(result)
        
        expected_contents = [
            "ACTION RESULT",  # Should contain this identifier
            "status",  # Should contain key dictionary keys
            "success",  # Should contain important values
            "data",
            "Example",
            "API"
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_action_result_with_object(self):
        """Test formatting custom object action results."""
        # Define a simple custom class
        class CustomResult:
            def __init__(self, name, value):
                self.name = name
                self.value = value
                self.timestamp = datetime.now()
            
            def __str__(self):
                return f"{self.name}: {self.value} at {self.timestamp}"
        
        # Create an instance
        result = CustomResult("TestAction", 42)
        
        formatted = format_action_result(result)
        
        expected_contents = [
            "ACTION RESULT",  # Should contain this identifier
            "TestAction",  # Should contain the name
            "42"  # Should contain the value as string
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_action_result_with_None(self):
        """Test formatting None action results."""
        result = None
        
        formatted = format_action_result(result)
        
        expected_contents = [
            "ACTION RESULT",  # Should contain this identifier
            "None"  # Should handle None value gracefully
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_human_feedback_with_string(self):
        """Test formatting simple string human feedback."""
        # Simple string feedback
        feedback = "That's correct, well done."
        
        formatted = format_human_feedback(feedback)
        
        expected_contents = [
            "HUMAN FEEDBACK",  # Should contain this identifier
            feedback,  # Should contain the original string
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_human_feedback_with_dict(self):
        """Test formatting dictionary human feedback."""
        # Dictionary feedback
        feedback = {
            "rating": 4.5,
            "comment": "Good job, but could be more concise.",
            "areas_for_improvement": ["brevity", "clarity"]
        }
        
        formatted = format_human_feedback(feedback)
        
        expected_contents = [
            "HUMAN FEEDBACK",  # Should contain this identifier
            "rating",  # Should contain key dictionary keys
            "4.5",  # Should contain important values as strings
            "Good job",
            "brevity",
            "clarity"
        ]
        
        for content in expected_contents:
            assert content in formatted
    
    def test_format_human_feedback_with_structured_input(self):
        """Test formatting structured input with mixed types."""
        # Structured feedback with mixed types
        feedback = {
            "structured_assessment": {
                "correctness": True,
                "usefulness": 9,
                "creativity": 7
            },
            "textual_feedback": "The solution is technically correct but could be more elegant.",
            "suggestions": ["Consider using a different algorithm", "Add more comments"]
        }
        
        formatted = format_human_feedback(feedback)
        
        expected_contents = [
            "HUMAN FEEDBACK",  # Should contain this identifier
            "structured_assessment",  # Should contain nested keys
            "correctness",
            "True",
            "9",
            "technically correct",
            "Consider using a different algorithm"
        ]
        
        for content in expected_contents:
            assert content in formatted
