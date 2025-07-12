# TACI Repository Restructuring & Naming Migration Log

**Date**: 2025-06-30  
**Scope**: Complete repository structure and naming standardization  
**Status**: Planning Phase

## 🎯 Objectives

1. **Fix naming inconsistencies** (`warpper` → `wrapper`, inconsistent suffixes)
2. **Standardize file structure** (consistent patterns across phases)
3. **Improve organization** (logical grouping, clear hierarchy)
4. **Maintain functionality** (update all internal references)
5. **Document everything** (complete change tracking)

---

## 📊 Current Structure Analysis

### Identified Issues

#### 1. **Naming Inconsistencies**
- `phase_00_warpper_checker_one_occ.py` → Should be `wrapper`
- `phase_00_wrapper_checker.py` vs `phase_00_warpper_checker_one_occ.py`
- Mixed naming patterns: `_one_occ` vs `_paralegal`
- Inconsistent file extensions and suffixes

#### 2. **Structure Issues**
- Scattered utility scripts in root directory
- Mixed data locations (`data/` vs `runs/`)
- Inconsistent phase organization
- No clear module boundaries

#### 3. **Path Dependencies**
- Hard-coded paths in multiple files
- Relative imports that break easily
- Mixed absolute/relative path usage

---

## 🗂️ Proposed New Structure

```
fresh_TACI/
├── README.md
├── requirements.txt
├── setup.py
├── MIGRATION_LOG.md          # This file
│
├── taci/                     # Core TACI package
│   ├── __init__.py
│   ├── core/                 # Core evaluation pipeline
│   │   ├── __init__.py
│   │   ├── phase_00_wrapper.py      # Wrapper validation
│   │   ├── phase_01_schema.py       # Schema validation
│   │   ├── phase_02_safety.py       # Safety evaluation
│   │   ├── phase_03_rubric.py       # Rubric scoring
│   │   └── phase_04_composite.py    # Composite analysis
│   │
│   ├── clients/              # LLM client interfaces
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── anthropic_client.py
│   │   ├── openai_client.py
│   │   ├── gemini_client.py
│   │   └── batch_runner.py
│   │
│   ├── data/                 # Data processing
│   │   ├── __init__.py
│   │   ├── loaders.py
│   │   ├── samplers.py
│   │   └── processors.py
│   │
│   └── utils/                # Core utilities
│       ├── __init__.py
│       ├── validation.py
│       ├── caching.py
│       └── helpers.py
│
├── evaluation/               # Evaluation scripts (organized by phase)
│   ├── __init__.py
│   ├── phase_00_wrapper/
│   │   ├── wrapper_validator.py
│   │   └── wrapper_validator_paralegal.py
│   ├── phase_01_schema/
│   │   ├── schema_validator.py
│   │   └── schema_validator_paralegal.py
│   ├── phase_02_safety/
│   │   ├── safety_evaluator.py
│   │   └── safety_evaluator_paralegal.py
│   ├── phase_03_rubric/
│   │   ├── rubric_grader.py
│   │   └── rubric_grader_paralegal.py
│   └── phase_04_composite/
│       ├── composite_analyzer.py
│       └── composite_analyzer_paralegal.py
│
├── data/                     # All data (consolidated)
│   ├── raw/
│   │   ├── onet/            # O*NET occupational data
│   │   ├── tasks/           # Task manifests
│   │   └── assets/          # Images, references
│   ├── processed/
│   │   ├── prompts/         # Generated prompts
│   │   └── samples/         # Sampled data
│   └── results/
│       ├── outputs/         # Raw model outputs
│       ├── evaluated/       # Graded results
│       └── final/           # Final composite scores
│
├── experiments/              # Research experiments
│   ├── paralegal_study/
│   ├── econometrics/
│   └── benchmarks/
│
├── scripts/                  # Utility scripts
│   ├── run_evaluation.py    # Main evaluation runner
│   ├── generate_prompts.py  # Prompt generation
│   ├── process_results.py   # Results processing
│   └── utilities/           # Helper scripts
│
├── analysis/                 # Analysis tools
│   ├── insights_analyzer.py
│   ├── visualizations.py
│   └── reports/
│
├── configs/                  # Configuration files
│   ├── models.yaml
│   ├── evaluation.yaml
│   └── schemas/
│
├── tests/                    # Test suite
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
└── docs/                     # Documentation
    ├── api/
    ├── tutorials/
    └── examples/
```

---

## 📋 Migration Plan

### Phase 1: Create New Structure ✅
- [x] Create new directory hierarchy
- [x] Add package `__init__.py` files
- [x] Create migration tracking system

### Phase 2: Fix Naming Issues
- [ ] Fix `warpper` → `wrapper` typo
- [ ] Standardize `_one_occ` vs `_paralegal` naming
- [ ] Consistent file naming patterns
- [ ] Update all internal references

### Phase 3: Reorganize Files
- [ ] Move graders to evaluation/ structure
- [ ] Consolidate LLM clients
- [ ] Organize data directories
- [ ] Move utility scripts

### Phase 4: Update Path References
- [ ] Create path configuration system
- [ ] Update hardcoded paths
- [ ] Add backward compatibility
- [ ] Test all functionality

### Phase 5: Clean Up & Document
- [ ] Remove old files after verification
- [ ] Update documentation
- [ ] Create migration helpers
- [ ] Final testing

---

## 🔄 Detailed File Changes

### Files to Rename (Phase 2)

| Current Path | New Path | Issue Fixed |
|-------------|----------|-------------|
| `graders/phase_00_wrapper/phase_00_warpper_checker_one_occ.py` | `evaluation/phase_00_wrapper/wrapper_validator_paralegal.py` | Typo + standardization |
| `graders/phase_00_wrapper/phase_00_wrapper_checker.py` | `evaluation/phase_00_wrapper/wrapper_validator.py` | Standardization |
| `graders/phase_01_schema/phase_01_schema_grader_one_occ.py` | `evaluation/phase_01_schema/schema_validator_paralegal.py` | Standardization |
| `graders/phase_01_schema/phase_01_schema_grader.py` | `evaluation/phase_01_schema/schema_validator.py` | Standardization |
| `graders/phase_02_safety/phase_02_safety_one_occ.py` | `evaluation/phase_02_safety/safety_evaluator_paralegal.py` | Standardization |
| `graders/phase_02_safety/phase_02_safety_grader.py` | `evaluation/phase_02_safety/safety_evaluator.py` | Standardization |
| `graders/phase_03_rubric/rubric_grader_one_occ.py` | `evaluation/phase_03_rubric/rubric_grader_paralegal.py` | Standardization |
| `graders/phase_03_rubric/rubric_grader.py` | `evaluation/phase_03_rubric/rubric_grader.py` | No change needed |
| `graders/phase_04_composite/phase_04_grader.py` | `evaluation/phase_04_composite/composite_analyzer.py` | Standardization |
| `graders/phase_04_composite_paralegal/final_composite_Paralegal.py` | `evaluation/phase_04_composite/composite_analyzer_paralegal.py` | Standardization |

### Files to Move (Phase 3)

| Current Path | New Path | Reason |
|-------------|----------|---------|
| `llm_client/run_batch_*.py` | `taci/clients/` | Core functionality |
| `prompt_gen/` | `taci/data/` | Data processing |
| `sampling/` | `taci/data/` | Data processing |
| `utils/` | `scripts/utilities/` | Utility scripts |
| `data/onet_raw/` | `data/raw/onet/` | Better organization |
| `data/manifests/` | `data/raw/tasks/` | Better organization |
| `runs/` | `data/results/outputs/` | Consolidation |
| `results/` | `data/results/final/` | Consolidation |

### Path Updates Required

| File | Current Reference | New Reference |
|------|------------------|---------------|
| `taci_insights_analyzer.py` | `graders/phase_00_to_03_condenser/` | `data/results/evaluated/` |
| `llm_client/run_batch_*.py` | `data/manifests/` | `data/raw/tasks/` |
| `utils/build_*_catalog.py` | `graders/phase_*/` | `evaluation/phase_*/` |

---

## ⚠️ Risk Mitigation

### Backup Strategy
1. **Git commit** before any changes
2. **Parallel testing** - keep old structure until verified
3. **Incremental migration** - one phase at a time
4. **Rollback plan** - documented reversal steps

### Compatibility Measures
1. **Symlinks** for critical paths during transition
2. **Path configuration** file for easy updates
3. **Import shims** for backward compatibility
4. **Extensive testing** at each phase

### Validation Steps
1. **Functionality testing** - run key workflows
2. **Path verification** - check all imports work
3. **Data integrity** - ensure no data loss
4. **Performance testing** - verify no regressions

---

## 📊 Progress Tracking

### Completed ✅
- [x] Structure analysis
- [x] Migration plan creation
- [x] Risk assessment
- [x] Initial directory structure

### In Progress 🔄
- [ ] Naming standardization

### Pending ⏳
- [ ] File reorganization
- [ ] Path updates
- [ ] Testing & validation
- [ ] Documentation updates

---

## 🤝 Next Steps

1. **Review this plan** - Ensure all issues are captured
2. **Execute Phase 2** - Fix naming issues first
3. **Test incrementally** - Verify each change
4. **Update documentation** - Keep this log current
5. **Plan Phase 3** - Detailed file reorganization

---

*This log will be updated as migration progresses. Each change will be documented with before/after states and verification steps.*