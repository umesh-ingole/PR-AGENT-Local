# Ruby style guide (project)
- Follow the community Ruby style guide: 2-space indentation, snake_case methods, CamelCase classes.
- Never pass user input to eval, system, exec, backticks, or send; use array-form system("cmd", arg).
- Use ActiveRecord placeholders or parameterized queries, never string interpolation in SQL.
- Prefer `unless` only for simple negations; avoid `unless/else`.
- Rescue specific exceptions, not bare `rescue` or `rescue Exception`.
- Use symbols for hash keys and frozen_string_literal comments in new files.
- Avoid mass-assignment of unpermitted params.
