#!/usr/bin/env python3
"""
TACI Path Mapping System
========================

Central configuration for all file paths in TACI.
This enables easy migration and backward compatibility.

Usage:
    from PATH_MAPPING import PATHS
    
    # Use new paths
    wrapper_script = PATHS.evaluation.phase_00.wrapper_checker
    
    # Get old paths for compatibility
    old_path = PATHS.legacy.graders.phase_00_wrapper
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# Project root
ROOT = Path(__file__).parent

@dataclass
class PathConfig:
    """Path configuration container"""
    
    # Current structure (new)
    evaluation: Dict[str, Any]
    data: Dict[str, Any]
    analysis: Dict[str, Any]
    scripts: Dict[str, Any]
    
    # Legacy structure (old)
    legacy: Dict[str, Any]

# Define all paths
PATHS = PathConfig(
    
    # NEW STRUCTURE
    evaluation={
        "phase_00": {
            "wrapper_checker": ROOT / "evaluation/phase_00_wrapper/phase_00_wrapper_checker.py",
            "wrapper_checker_paralegal": ROOT / "evaluation/phase_00_wrapper/phase_00_wrapper_checker_paralegal.py",
            "output_dir": ROOT / "evaluation/phase_00_wrapper",
        },
        "phase_01": {
            "schema_grader": ROOT / "evaluation/phase_01_schema/phase_01_schema_grader.py", 
            "schema_grader_paralegal": ROOT / "evaluation/phase_01_schema/phase_01_schema_grader_paralegal.py",
            "output_dir": ROOT / "evaluation/phase_01_schema",
        },
        "phase_02": {
            "safety_grader": ROOT / "evaluation/phase_02_safety/phase_02_safety_grader.py",
            "safety_grader_paralegal": ROOT / "evaluation/phase_02_safety/phase_02_safety_grader_paralegal.py", 
            "output_dir": ROOT / "evaluation/phase_02_safety",
        },
        "phase_03": {
            "rubric_grader": ROOT / "evaluation/phase_03_rubric/rubric_grader.py",
            "rubric_grader_paralegal": ROOT / "evaluation/phase_03_rubric/rubric_grader_paralegal.py",
            "output_dir": ROOT / "evaluation/phase_03_rubric",
        },
        "phase_04": {
            "composite_analyzer": ROOT / "evaluation/phase_04_composite/phase_04_grader.py",
            "composite_analyzer_paralegal": ROOT / "evaluation/phase_04_composite/final_composite_Paralegal.py",
            "output_dir": ROOT / "evaluation/phase_04_composite",
        }
    },
    
    data={
        "raw": {
            "onet": ROOT / "data/onet_raw",  # Legacy location for now
            "tasks": ROOT / "data/manifests",  # Legacy location for now
            "assets": ROOT / "assets",  # Legacy location for now
        },
        "results": {
            "outputs": ROOT / "runs",  # Legacy location for now
            "evaluated": ROOT / "graders",  # Legacy location for now
            "final": ROOT / "results",  # Legacy location for now
        }
    },
    
    analysis={
        "insights_analyzer": ROOT / "analysis/insights_analyzer.py",
        "visualizations": ROOT / "analysis/visualizations.py",
        "reports": ROOT / "analysis/reports",
    },
    
    scripts={
        "run_analysis": ROOT / "scripts/run_analysis.py",
        "utilities": ROOT / "scripts/utilities",
    },
    
    # LEGACY STRUCTURE (for backward compatibility)
    legacy={
        "graders": {
            "phase_00_wrapper": ROOT / "graders/phase_00_wrapper",
            "phase_01_schema": ROOT / "graders/phase_01_schema", 
            "phase_02_safety": ROOT / "graders/phase_02_safety",
            "phase_03_rubric": ROOT / "graders/phase_03_rubric",
            "phase_04_composite": ROOT / "graders/phase_04_composite",
            "phase_04_composite_paralegal": ROOT / "graders/phase_04_composite_paralegal",
        },
        "llm_client": ROOT / "llm_client",
        "utils": ROOT / "utils",
        "data_manifests": ROOT / "data/manifests",
        "data_onet": ROOT / "data/onet_raw",
        "runs": ROOT / "runs",
    }
)

# Helper functions for migration
def get_legacy_path(new_path: Path) -> Path:
    """Get legacy path for a new path"""
    # This would contain mapping logic
    # For now, return the path as-is
    return new_path

def get_new_path(legacy_path: Path) -> Path:
    """Get new path for a legacy path"""
    # This would contain reverse mapping logic
    return legacy_path

# File mapping for renamed files
FILE_MAPPING = {
    # Before -> After
    "phase_00_wrapper_checker_paralegal.py": "phase_00_wrapper_checker_paralegal.py",
    "phase_01_schema_grader_paralegal.py": "phase_01_schema_grader_paralegal.py", 
    "phase_02_safety_grader_paralegal.py": "phase_02_safety_grader_paralegal.py",
    "rubric_grader_paralegal.py": "rubric_grader_paralegal.py",
    "final_composite_Paralegal.py": "composite_analyzer_paralegal.py",
    "vison_garder_test.py": "vision_guard_test.py",
    "convert_mammograms 2.py": "convert_mammograms_v2.py",
}

if __name__ == "__main__":
    print("TACI Path Mapping Configuration")
    print("=" * 40)
    print(f"Project Root: {ROOT}")
    print(f"Evaluation Dir: {PATHS.evaluation['phase_00']['output_dir']}")
    print(f"Legacy Graders: {PATHS.legacy['graders']['phase_00_wrapper']}")