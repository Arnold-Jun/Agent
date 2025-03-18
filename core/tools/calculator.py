import json
import math
import dspy

class Calculator(dspy.Module):
    def __init__(
            self,
            add: bool = True,
            subtract: bool = True,
            multiply: bool = True,
            divide: bool = True,
            exponentiate: bool = False,
            factorial: bool = False,
            is_prime: bool = False,
            square_root: bool = False,
    ):
        # 初始化模块，并设置模块名称为 "calculator"
        super().__init__(name="calculator")

        # 根据参数注册对应的运算函数
        if add:
            self.register(self.add)
        if subtract:
            self.register(self.subtract)
        if multiply:
            self.register(self.multiply)
        if divide:
            self.register(self.divide)
        if exponentiate:
            self.register(self.exponentiate)
        if factorial:
            self.register(self.factorial)
        if is_prime:
            self.register(self.is_prime)
        if square_root:
            self.register(self.square_root)

    def add(self, a: float, b: float) -> str:
        """Add two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a + b
        return json.dumps({"operation": "addition", "result": result})

    def subtract(self, a: float, b: float) -> str:
        """Subtract second number from first and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a - b
        return json.dumps({"operation": "subtraction", "result": result})

    def multiply(self, a: float, b: float) -> str:
        """Multiply two numbers and return the result.

        Args:
            a (float): First number.
            b (float): Second number.

        Returns:
            str: JSON string of the result.
        """
        result = a * b
        return json.dumps({"operation": "multiplication", "result": result})

    def divide(self, a: float, b: float) -> str:
        """Divide first number by second and return the result.

        Args:
            a (float): Numerator.
            b (float): Denominator.

        Returns:
            str: JSON string of the result.
        """
        if b == 0:
            return json.dumps({"operation": "division", "error": "Division by zero is undefined"})
        result = a / b
        return json.dumps({"operation": "division", "result": result})

    def exponentiate(self, a: float, b: float) -> str:
        """Raise first number to the power of the second number and return the result.

        Args:
            a (float): Base.
            b (float): Exponent.

        Returns:
            str: JSON string of the result.
        """
        result = math.pow(a, b)
        return json.dumps({"operation": "exponentiation", "result": result})

    def factorial(self, n: int) -> str:
        """Calculate the factorial of a number and return the result.

        Args:
            n (int): Number to calculate the factorial of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            return json.dumps({"operation": "factorial", "error": "Factorial of a negative number is undefined"})
        result = math.factorial(n)
        return json.dumps({"operation": "factorial", "result": result})

    def is_prime(self, n: int) -> str:
        """Check if a number is prime and return the result.

        Args:
            n (int): Number to check if prime.

        Returns:
            str: JSON string of the result.
        """
        if n <= 1:
            return json.dumps({"operation": "prime_check", "result": False})
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return json.dumps({"operation": "prime_check", "result": False})
        return json.dumps({"operation": "prime_check", "result": True})

    def square_root(self, n: float) -> str:
        """Calculate the square root of a number and return the result.

        Args:
            n (float): Number to calculate the square root of.

        Returns:
            str: JSON string of the result.
        """
        if n < 0:
            return json.dumps({"operation": "square_root", "error": "Square root of a negative number is undefined"})
        result = math.sqrt(n)
        return json.dumps({"operation": "square_root", "result": result})
