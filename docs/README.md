# TACI Documentation

## Overview

TACI (Task Automation Capability Index) is a comprehensive framework for evaluating AI model capabilities on real-world task automation.

## Quick Start

### Running Analysis

```bash
# Modern entry point
python scripts/run_analysis.py

# Or use original scripts directly
python taci_insights_analyzer.py
python graph.py
```

### Key Components

- **Evaluation Pipeline**: Multi-phase assessment (wrapper, schema, safety, rubric, composite)
- **Analysis Tools**: Systematic insights generation and visualization
- **Data Management**: O*NET integration and task sampling

## Directory Structure

```
fresh_TACI/
├── graders/           # Evaluation pipeline phases
├── llm_client/        # Model client interfaces
├── data/              # Raw data and manifests
├── runs/              # Model output results
├── analysis/          # Modern analysis tools
├── scripts/           # Utility scripts
└── docs/              # Documentation
```

## Analysis Tools

### Insights Analyzer
Systematic analysis of model performance:
- Structural reliability assessment
- Capability tier identification  
- Quality analysis across rubric dimensions
- Prompt robustness evaluation

### Visualization Suite
Publication-ready charts showing:
- Model capability progression
- Performance thresholds
- Quality comparisons

## Research Findings

TACI has identified critical thresholds in AI automation:
- **Professional Threshold**: 75+ score for production viability
- **Expert Level**: 80+ score for expert-level performance
- **Reliability Requirements**: <5% structural failure rate

## Contributing

Please maintain backward compatibility when making changes:
1. Keep existing file paths working
2. Add new functionality in `analysis/` or `scripts/`
3. Test with existing workflows