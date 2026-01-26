"""Example Python module for testing code parser."""


class Calculator:
    """A simple calculator class."""

    def __init__(self, name: str):
        """Initialize calculator with a name."""
        self.name = name
        self.history: list[str] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result


def create_calculator(name: str = "default") -> Calculator:
    """Factory function to create a calculator."""
    return Calculator(name)


async def async_calculate(operation: str, a: float, b: float) -> float:
    """Async calculation function."""
    calc = Calculator("async")
    if operation == "add":
        return calc.add(a, b)
    elif operation == "multiply":
        return calc.multiply(a, b)
    else:
        raise ValueError(f"Unknown operation: {operation}")
