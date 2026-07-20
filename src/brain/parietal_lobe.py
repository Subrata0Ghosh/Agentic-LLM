"""
Parietal Lobe — src/brain/parietal_lobe.py
============================================
Brain Region: Parietal Cortex (IPS - Intraparietal Sulcus)

The parietal lobe handles spatial attention and numerical processing (math).
In this upgraded version, it runs a SymPy-based Code Interpreter utilizing the
local 0.5B model to translate natural language math into executable Python/SymPy.
"""

import re
import math
import sympy as sp

try:
    from generate import generate_text_api
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
    from generate import generate_text_api


class ParietalLobe:
    """
    Mathematical and symbolic reasoning engine representing the parietal cortex.
    Uses SymPy to solve calculus, algebra, and arithmetic queries.
    """

    def __init__(self):
        # Match standard mathematical equations or word problems containing math expressions
        self.math_pattern = re.compile(
            r"(\b(solve|calculate|evaluate|compute|equation|derivative|integral|sum|product|limit|factorial)\b)|([\d\+\-\*\/\%\^\(\)\=x]{3,})"
        )

    def is_math_query(self, query: str) -> bool:
        """Determines if a query requires mathematical or symbolic logic."""
        # Check if it has a math command word
        has_math_word = any(
            w in query.lower() for w in
            ["solve", "calculate", "evaluate", "compute", "derivative", "integral", "factorial", "limit", "equation"]
        )
        if has_math_word:
            return True

        # Check if there are digits AND at least one operator
        has_digits = len(re.findall(r"\d+", query)) >= 1
        has_ops = any(op in query for op in ["+", "-", "*", "/", "=", "^", "%"])
        return has_digits and has_ops

    def solve(self, query: str) -> dict:
        """
        Processes the query, translates it to Python/SymPy code via LLM,
        executes the code safely, and returns the result.
        """
        # Double check original query for obvious exploit attempts
        for forbidden in ["import ", "eval(", "exec(", "subprocess.", "open("]:
            if forbidden in query.lower():
                return {
                    "success": False,
                    "reason": "Exploit attempt detected in math query.",
                    "steps": []
                }

        system_prompt = (
            "You are a mathematical translation unit. Your task is to translate the user's math query into a single valid Python expression using SymPy. "
            "The expression must evaluate to the answer using SymPy symbols and functions. "
            "Define 'x' as a symbol (from sympy import symbols; x = symbols('x')) and use standard SymPy functions like: "
            "diff(expr, x) for derivatives, integrate(expr, x) for integrals, solve(eq, x) for solving equations (eq is left - right if there is an '='), "
            "limit(expr, x, val) for limits, or standard math operations. "
            "CRITICAL: Output ONLY the raw Python expression string. Do not include triple backticks, markdown formatting, explanation, or extra code. "
            "Examples:\n"
            "Query: derivative of x^2 + 3x\n"
            "Expression: diff(x**2 + 3*x, x)\n"
            "Query: integrate sin(x)\n"
            "Expression: integrate(sin(x), x)\n"
            "Query: solve x^2 - 9 = 0\n"
            "Expression: solve(x**2 - 9, x)\n"
            "Query: what is 500 * 12 / 3\n"
            "Expression: 500 * 12 / 3\n"
            "Query: limit of sin(x)/x as x goes to 0\n"
            "Expression: limit(sin(x)/x, x, 0)"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nExpression:"}
        ]

        steps = []
        try:
            # Generate the Python/SymPy expression
            raw_code = generate_text_api(
                messages, max_new_tokens=60,
                temperature=0.1, top_p=0.9
            )
            cleaned_code = raw_code.strip().replace("`", "").replace("python", "").strip()

            # Sanitization check to prevent executing arbitrary system code
            allowed_chars = re.compile(r"^[a-zA-Z0-9\+\-\*\/\(\)\,\.\s\%\[\]\_]+$")
            if not allowed_chars.match(cleaned_code):
                return {
                    "success": False,
                    "reason": "Unsafe characters detected in generated expression.",
                    "steps": [f"Translated code: {cleaned_code} rejected due to safety filter."]
                }

            # Double check there are no dangerous builtins (using word boundaries to avoid false positives like 'cos')
            for forbidden in ["import", "eval", "exec", "os", "sys", "subprocess", "open"]:
                if re.search(rf"\b{forbidden}\b", cleaned_code):
                    return {
                        "success": False,
                        "reason": f"Forbidden keyword '{forbidden}' detected.",
                        "steps": []
                    }
            if "__" in cleaned_code:
                return {
                    "success": False,
                    "reason": "Forbidden dunder sequence '__' detected.",
                    "steps": []
                }

            steps.append(f"Translated query to SymPy expression: {cleaned_code}")

            # Define standard variables
            x, y, z = sp.symbols('x y z')
            # Build clean evaluation namespace
            context = {name: getattr(sp, name) for name in dir(sp) if not name.startswith('_')}
            context.update({'x': x, 'y': y, 'z': z, 'sp': sp})

            # Execute
            result = eval(cleaned_code, {"__builtins__": None}, context)
            steps.append("Evaluated SymPy code successfully.")

            return {
                "success": True,
                "expression": cleaned_code,
                "result": str(result),
                "steps": steps
            }
        except Exception as e:
            return {
                "success": False,
                "reason": f"Calculation error: {str(e)}",
                "steps": steps
            }
