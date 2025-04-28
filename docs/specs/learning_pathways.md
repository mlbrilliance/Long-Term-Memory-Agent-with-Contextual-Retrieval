# Learning Pathways Specification

## Overview

This document specifies the design and interface for the learning pathway functions in the LTM Agent. These functions are responsible for formatting action results and human feedback into a standardized text format suitable for storage in the agent's memory.

## Functions

### `format_action_result(result: Any) -> str`

#### Purpose
Convert the result of an agent action into a formatted string suitable for storage in the agent's memory. This enables the agent to learn from its own actions and their outcomes.

#### Interface
- **Input:** `result: Any` - The result of an agent action, which can be of various types:
  - Simple string
  - Dictionary (potentially nested)
  - Custom object with attributes
  - None or other primitive values
- **Output:** `str` - A formatted string with a standardized structure that includes:
  - A clear identifier indicating this is an action result
  - The original content in a readable format
  - Timestamp information (when the action occurred)
  - Any other relevant metadata

#### Formatting Requirements
1. Must begin with a clear identifier (e.g., "ACTION RESULT:")
2. Must include the timestamp of when the formatting occurred
3. Must handle different input types gracefully:
   - For strings: Include directly with minimal processing
   - For dictionaries: Convert to a readable format (e.g., JSON or a formatted string representation)
   - For custom objects: Extract relevant attributes and format them consistently
   - For None or other primitive values: Format in a human-readable way
4. Should be concise yet comprehensive
5. Should maintain a consistent format across different input types

#### Example
```python
# Input: Dictionary result
result = {
    "status": "success",
    "data": {"id": 123, "name": "Example"}
}

# Expected output (example)
output = """ACTION RESULT: [2025-04-27T12:34:56]
Status: success
Data: 
  - ID: 123
  - Name: Example
"""
```

### `format_human_feedback(feedback: Any) -> str`

#### Purpose
Convert human feedback into a formatted string suitable for storage in the agent's memory. This enables the agent to learn from human guidance and corrections.

#### Interface
- **Input:** `feedback: Any` - Human feedback, which can be of various types:
  - Simple string (e.g., direct feedback message)
  - Dictionary (potentially with structured feedback components)
  - Other types that represent human input/feedback
- **Output:** `str` - A formatted string with a standardized structure that includes:
  - A clear identifier indicating this is human feedback
  - The feedback content in a readable format
  - Timestamp information
  - Any other relevant metadata

#### Formatting Requirements
1. Must begin with a clear identifier (e.g., "HUMAN FEEDBACK:")
2. Must include the timestamp of when the formatting occurred
3. Must handle different input types gracefully, similar to `format_action_result`
4. Should emphasize important feedback points (e.g., specific corrections or suggestions)
5. Should maintain a consistent format across different input types

#### Example
```python
# Input: Dictionary feedback
feedback = {
    "rating": 4.5,
    "comment": "Good job, but could be more concise."
}

# Expected output (example)
output = """HUMAN FEEDBACK: [2025-04-27T12:34:56]
Rating: 4.5/5.0
Comment: Good job, but could be more concise.
"""
```

## Implementation Considerations

1. **Timestamp Format**: Use ISO 8601 format for consistency (YYYY-MM-DDTHH:MM:SS)
2. **Nested Data Handling**: For deeply nested data structures, consider limiting the depth of recursion to avoid excessively long outputs
3. **Error Handling**: Gracefully handle unexpected input types or malformed data
4. **Text Length**: Consider implementing a maximum length for the formatted output to avoid overwhelming the memory storage
5. **Consistency**: Ensure consistency between the formatting of action results and human feedback to simplify processing and retrieval

## Usage Context

These functions will be used in the agent's learning loop, specifically:

1. After an agent performs an action, `format_action_result` will be called to format the result
2. When a human provides feedback, `format_human_feedback` will be called to format it
3. The formatted strings will be stored in the agent's memory (likely as `KnowledgeUnit` objects)
4. During retrieval, these formatted strings will be processed and used to inform future actions

## Test Criteria

1. **Correct Identification**: Output should be clearly identifiable as either action result or human feedback
2. **Type Handling**: All supported input types should be handled correctly
3. **Consistency**: Similar inputs should produce consistently formatted outputs
4. **Readability**: Output should be human-readable and logically structured
5. **Error Resilience**: Function should not fail on unexpected inputs
