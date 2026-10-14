# C / C++ style guide (project)
- Never use gets(); use fgets() with a size limit. Avoid strcpy/strcat/sprintf; use strncpy/snprintf with bounds.
- Always check the return value of malloc/calloc/realloc and free exactly once; set pointers to NULL after free.
- Initialize all variables before use. Avoid uninitialized reads.
- Check array indexes against bounds before access; watch for off-by-one in loops (< not <=).
- Validate printf-style format strings; never pass user input as the format string.
- In C++ prefer std::string, std::vector, and smart pointers (std::unique_ptr) over raw new/delete and C arrays.
- Mark single-argument constructors explicit; follow the rule of three/five.
- Use const wherever a value must not change.
