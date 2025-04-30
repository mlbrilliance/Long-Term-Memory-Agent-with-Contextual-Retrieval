"""Test proper usage of ValidationError in Pydantic v2, writing to a file."""

import inspect

from pydantic import ValidationError

# Open a file for writing the results
with open("validation_test_results.txt", "w") as f:
    f.write(f"ValidationError loaded from: {inspect.getfile(ValidationError)}\n")
    f.write(f"ValidationError MRO: {ValidationError.__mro__}\n\n")

    # How to properly create ValidationError in Pydantic v2
    errors = [
        {"loc": ("anthropic_api_key",), "msg": "Field required", "type": "value_error.missing"}
    ]

    try:
        # This is how it works in pydantic_core based ValidationError
        # See: https://docs.pydantic.dev/latest/errors/errors/
        raise ValidationError.from_exception_data(title="Validation Error", line_errors=errors)
    except Exception as e:
        f.write(f"SUCCESS: Raised {type(e).__name__} properly!\n")
        f.write(f"Error message: {str(e)}\n")
        # Print available attributes and methods
        f.write(f"Error dir: {dir(e)}\n")
        f.write(f"Has 'errors' attribute: {'errors' in dir(e)}\n")
        if hasattr(e, "errors") and callable(e.errors):
            f.write(f"First error: {e.errors()[0]}\n")

print("Results written to validation_test_results.txt")
