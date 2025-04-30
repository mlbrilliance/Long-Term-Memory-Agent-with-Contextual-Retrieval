# Script Documentation

This directory contains various scripts for testing, evaluating, and demonstrating the Long-Term Memory Agent capabilities.

## Demonstration Scripts

- `minimal_demo.py`: A simplified demonstration of core memory features that avoids problematic fields. This is the most reliable demo script.

- `simple_demo.py`: Demonstrates memory features with more complex examples, but may have issues with certain fields like "last_accessed".

- `demo_consolidation.py`: Demonstrates memory consolidation features, but may encounter issues with certain field references.

- `basic_test.py`: A test script showing how to use the cross-referencer and consolidator classes.

## Evaluation Scripts

- `evaluate.py`: Main evaluation script for assessing the Long-Term Memory Agent's performance.

- `simple_evaluate.py`: Simplified version of the evaluation script for quick testing.

## Output Locations

All script outputs are saved to the following locations:

- Demo outputs: `../logs/demos/`
- Evaluation results: `../logs/evaluation/`
- Test outputs: `../logs/tests/`

## Running Scripts

To run any of the scripts, use the following command:

```
python scripts/<script_name>.py
```

For example, to run the minimal demo:

```
python scripts/minimal_demo.py
```
