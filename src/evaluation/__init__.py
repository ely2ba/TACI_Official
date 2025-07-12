"""
TACI Evaluation Module
====================

Multi-phase evaluation pipeline for AI task automation assessment.

This module contains the core evaluation phases:
- Phase 00: Wrapper validation (output format compliance)
- Phase 01: Schema validation (structured output verification)  
- Phase 02: Safety assessment (harmful content detection)
- Phase 03: Rubric scoring (multi-dimensional quality assessment)
- Phase 04: Composite analysis (final scoring and insights)
- Data Processing: Master table construction and data aggregation

Each phase can be run independently or as part of the full pipeline.

Directory Structure:
- phase_00_wrapper/: Format compliance checking
- phase_01_schema/: Structured output validation
- phase_02_safety/: Safety and harm detection
- phase_03_rubric/: Multi-dimensional quality scoring
- phase_04_composite/: Final composite scoring
- data_processing/: Data aggregation and master table building
"""

__version__ = "1.0.0"