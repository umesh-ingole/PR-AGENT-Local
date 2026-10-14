# Java style guide (project)
- Follow Google Java Style: 2-space indentation, braces on every if/for/while, UpperCamelCase classes, lowerCamelCase methods.
- Close resources with try-with-resources.
- Never swallow exceptions with an empty catch block; log or rethrow.
- Compare strings with .equals(), never ==. Put the literal first to avoid NullPointerException.
- Use PreparedStatement with placeholders for SQL; never concatenate user input.
- Do not use public mutable fields; expose accessors. Make fields final when possible.
- Avoid catching Exception or Throwable broadly.
- Do not hard-code credentials or secrets.
