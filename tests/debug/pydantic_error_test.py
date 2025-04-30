"""Test Pydantic v2 ValidationError with our exact data structure."""

import inspect

from pydantic import ValidationError

with open("error_test_result.txt", "w") as f:
    f.write(f"ValidationError loaded from: {inspect.getfile(ValidationError)}\n")
    f.write(f"ValidationError MRO: {ValidationError.__mro__}\n\n")

    # Create errors in our exact format
    errors = [
        {"loc": ("anthropic_api_key",), "msg": "Field required", "type": "value_error.missing"}
    ]

    # Test different formats
    f.write("==== Testing Error Formats ====\n")

    try:
        # Standard pydantic v2 format requires each error to be in the expected format
        # The error object needs to match PyLineError structure
        from pydantic_core import PythonError

        proper_errors = []
        for error in errors:
            proper_errors.append(
                PythonError(type=error["type"], loc=error["loc"], msg=error["msg"])
            )

        raise ValidationError.from_exception_data(
            title="Validation Error", line_errors=proper_errors
        )
    except Exception as e:
        f.write(f"Test result: {type(e).__name__}\n")
        f.write(f"Error message: {str(e)}\n\n")

print("Check error_test_result.txt for the detailed output")
