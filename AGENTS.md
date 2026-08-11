# Test permissions

Pytest tests involving PDF extraction may encounter permission restrictions. Run them with elevated permissions, or skip those tests.

# Python environments

- If the default uv cache is not writable, use `UV_CACHE_DIR=.uv-cache`.
