"""Test proper usage of ValidationError in Pydantic v2."""

import inspect

from pydantic import ValidationError

print("ValidationError loaded from:", inspect.getfile(ValidationError))
print("ValidationError MRO:", ValidationError.__mro__)

# How to properly create ValidationError in Pydantic v2
errors = [{"loc": ("anthropic_api_key",), "msg": "Field required", "type": "value_error.missing"}]

try:
    # This is how it works in pydantic_core based ValidationError
    # See: https://docs.pydantic.dev/latest/errors/errors/
    raise ValidationError.from_exception_data(title="Validation Error", line_errors=errors)
except Exception as e:
    print(f"SUCCESS: Raised {type(e).__name__} properly!")
    print(f"Error message: {str(e)}")
    # Print available attributes and methods
    print(f"Has 'errors' attribute: {'errors' in dir(e)}")
    if hasattr(e, "errors"):
        print(f"First error: {e.errors()[0]}")
