# Security checklist (all languages)
- Injection: SQL, command, path traversal, template. Treat all external input as untrusted.
- Hard-coded secrets, API keys, passwords, tokens in source.
- Weak cryptography: MD5, SHA1, DES, ECB mode, hard-coded IVs; use modern primitives and a CSPRNG.
- Unsafe deserialization of untrusted data.
- Buffer overflow, use-after-free, double free, integer overflow in native code.
- Missing input validation, missing authorization checks, verbose error messages leaking internals.
- Race conditions (TOCTOU) on files and shared state.
