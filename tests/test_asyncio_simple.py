"""
Simple test to verify that pytest-asyncio is working correctly.
"""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_asyncio_working():
    """Simple test to verify asyncio works."""
    await asyncio.sleep(0.1)
    assert True
