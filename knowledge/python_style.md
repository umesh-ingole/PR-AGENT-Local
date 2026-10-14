# Python style guide (project)
- Follow PEP 8: 4-space indentation, lines under 100 characters, snake_case for functions and variables.
- Never use bare `except:`; catch specific exceptions and log them.
- Do not use mutable default arguments such as `def f(x=[])`; use None and create inside.
- Use context managers (`with open(...)`) for files and connections.
- Never call eval() or exec() on user input; avoid pickle for untrusted data.
- Build SQL with parameterized queries, never string formatting.
- Use `subprocess.run([...], shell=False)` with a list of arguments, never shell=True with user input.
- Compare to None with `is`, not `==`.
- Add docstrings to public functions and classes.
