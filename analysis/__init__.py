"""
TACI Analysis Module
==================

Professional analysis tools for TACI evaluation results.

This module provides high-level interfaces to the core TACI analysis functionality
while maintaining compatibility with the existing codebase structure.
"""

# Import main analyzer class
try:
    from .insights_analyzer import TACIAnalyzer
    __all__ = ["TACIAnalyzer"]
except ImportError:
    # Fallback for development
    __all__ = []