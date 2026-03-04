"""Shared pytest configuration and fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (> 30s)")
    config.addinivalue_line("markers", "llm: marks tests that call external LLM APIs or load heavy models")
    config.addinivalue_line("markers", "integration: marks tests requiring external services")


def pytest_collection_modifyitems(config, items):
    """Auto-mark entire test_mcp_server.py as llm (initializes DigitalCerebellum → SentenceTransformer)."""
    for item in items:
        if "test_mcp_server" in str(item.fspath):
            item.add_marker(pytest.mark.llm)
        if "test_openclaw_benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
