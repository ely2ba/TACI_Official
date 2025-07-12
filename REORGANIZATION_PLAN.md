# TACI Repository Reorganization Plan

## Current Issues
- Root directory cluttered with loose files (`graph.py`, `renaming.py`, etc.)
- Inconsistent naming (`phase_00_warpper_checker` typo)
- Mixed data/results locations (`data/runs/` vs `runs/`)
- Utility scripts scattered across multiple locations
- No clear separation between core framework and analysis tools

## Proposed New Structure

```
fresh_TACI/
├── README.md
├── LICENSE
├── requirements.txt              # NEW: Dependencies
├── setup.py                     # NEW: Package setup
│
├── taci/                        # NEW: Core TACI framework
│   ├── __init__.py
│   ├── core/                    # Core evaluation pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Main orchestrator
│   │   ├── phase_wrapper.py     # Phase 0: Wrapper validation
│   │   ├── phase_schema.py      # Phase 1: Schema validation  
│   │   ├── phase_safety.py      # Phase 2: Safety evaluation
│   │   ├── phase_rubric.py      # Phase 3: Rubric scoring
│   │   └── phase_composite.py   # Phase 4: Composite scoring
│   │
│   ├── models/                  # LLM client interfaces
│   │   ├── __init__.py
│   │   ├── base.py              # Base model interface
│   │   ├── anthropic_client.py  # Anthropic models
│   │   ├── openai_client.py     # OpenAI models
│   │   ├── gemini_client.py     # Google models
│   │   └── batch_runner.py      # Batch processing
│   │
│   ├── prompts/                 # Prompt management
│   │   ├── __init__.py
│   │   ├── generator.py         # Prompt generation
│   │   ├── templates/           # Template files
│   │   └── variants/            # Generated prompts
│   │
│   └── utils/                   # Core utilities
│       ├── __init__.py
│       ├── data_loader.py
│       ├── validator.py
│       └── extractor.py
│
├── analysis/                    # NEW: Analysis and visualization
│   ├── __init__.py
│   ├── insights_analyzer.py     # Systematic insights (your new tool)
│   ├── visualizations.py       # Graph generation
│   ├── statistical_tests.py    # Statistical analysis
│   └── reports/                 # Generated reports
│
├── data/                        # Consolidated data directory
│   ├── raw/                     # Raw input data
│   │   ├── onet/               # O*NET data
│   │   ├── tasks/              # Task manifests
│   │   └── assets/             # Images, references
│   │
│   ├── processed/              # Processed data
│   │   ├── prompts/            # Generated prompts by type
│   │   └── samples/            # Sampled tasks
│   │
│   └── results/                # All evaluation results
│       ├── raw_outputs/        # Raw model outputs
│       ├── graded/             # Graded results by phase
│       ├── composite/          # Final composite scores
│       └── analysis/           # Analysis outputs
│
├── experiments/                # NEW: Research experiments
│   ├── paralegal_study/        # Main paralegal automation study
│   ├── econometrics/           # Economic analysis
│   └── human_evaluation/       # Human audit data
│
├── configs/                    # Configuration files
│   ├── models.yaml            # Model configurations
│   ├── evaluation.yaml        # Evaluation parameters
│   └── schemas/               # Validation schemas
│
├── scripts/                    # Standalone scripts
│   ├── run_evaluation.py      # Main evaluation runner
│   ├── generate_report.py     # Report generation
│   ├── data_preprocessing.py  # Data preparation
│   └── utilities/             # Utility scripts
│
├── tests/                      # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
│
└── docs/                       # Documentation
    ├── api/                   # API documentation
    ├── tutorials/             # Usage tutorials
    ├── papers/                # Research papers/reports
    └── examples/              # Example usage
```

## Migration Strategy

### Phase 1: Create New Structure
1. Create new directory structure
2. Move core evaluation logic to `taci/core/`
3. Consolidate LLM clients to `taci/models/`
4. Move analysis tools to `analysis/`

### Phase 2: Data Consolidation  
1. Merge `data/` and `runs/` into unified `data/` structure
2. Organize results by evaluation phase
3. Clean up duplicate/temporary files

### Phase 3: Code Refactoring
1. Create proper Python package structure
2. Fix naming inconsistencies (`warpper` → `wrapper`)
3. Add `__init__.py` files and imports
4. Create main entry points

### Phase 4: Documentation & Polish
1. Update README with new structure
2. Add proper documentation
3. Create requirements.txt
4. Add setup.py for easy installation

## Benefits of New Structure

1. **Professional Package Layout**: Follows Python best practices
2. **Clear Separation of Concerns**: Core framework vs analysis vs experiments  
3. **Easier Navigation**: Logical grouping of related functionality
4. **Better Collaboration**: Clear where to add new features
5. **Cleaner Root**: Remove clutter from main directory
6. **Scalable**: Easy to add new models, evaluations, analyses

## Files to Relocate

### Move to `taci/core/`:
- `graders/phase_*/` → Organized by evaluation phase
- Core pipeline logic

### Move to `taci/models/`:
- `llm_client/` → Renamed and reorganized

### Move to `analysis/`:
- `graph.py` → `visualizations.py`
- `taci_insights_analyzer.py` → `insights_analyzer.py`

### Move to `scripts/`:
- `renaming.py`, `testingsafety.py`, `testingtokens.py`
- `utils/` scripts

### Consolidate in `data/`:
- `data/` + `runs/` + `results/` → Unified structure
- `prompts/` → `data/processed/prompts/`

Would you like me to start implementing this reorganization?