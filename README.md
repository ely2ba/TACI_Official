# TACI (Task-AI Capability Index) - Alpha Showcase

A novel research framework establishing systematic methodology for evaluating AI automation potential across professional occupations through statistically rigorous 5-phase assessment pipeline. TACI provides the first comprehensive benchmark for measuring AI model capabilities on real-world occupational tasks with production-grade engineering and academic rigor.

> **🚧 Alpha Version**: This is a demonstration/showcase version of TACI. The internal production system includes 40+ occupations, 15+ model providers, and advanced experimental capabilities.

## Overview

TACI evaluates AI models through **7,600+ lines of research-grade Python** implementing:
- **Multi-modal assessment** across TEXT, GUI, and VISION tasks with computer vision IoU scoring
- **Statistical rigor** via 3-vote self-consistency validation and bootstrap confidence intervals  
- **Production-scale orchestration** of Anthropic Claude, OpenAI GPT, Google Gemini, and Meta Llama APIs
- **Systematic evaluation methodology** through 5-phase pipeline with weighted composite scoring
- **Economic impact modeling** for automation potential across 20+ professional occupations

📂 Reviewer’s Guide

This repository is large (7k+ LOC) because it’s a full research pipeline.
For a quick review of coding style and methodology, here are 3 compact entry points:

1️⃣ Rubric-Based Scoring

File: src/evaluation/phase_03_rubric/rubric_grader_paralegal.py

# System Architecture & Workflow

```
═══════════════════════════════════════════════════════════════════════════════════════════════════
                                    DATA INGESTION STAGE                                          
═══════════════════════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │Task_Statements  │     │ Task_Ratings    │     │Occupation_Data   │
    │     .txt        │     │     .txt        │     │     .txt         │
    └────────┬────────┘     └────────┬────────┘     └────────┬─────────┘
             │                       │                        │
             ▼                       ▼                        ▼
    ┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │  Extract Tasks  │     │ Extract Ratings │     │Extract Occupation│
    │                 │     │                 │     │     Titles       │
    └────────┬────────┘     └────────┬────────┘     └────────┬─────────┘
             │                       │                        │
             └───────────┬───────────┘                        │
                         ▼                                     │
              ┌──────────────────────────┐                    │
              │  Merge Tasks + Ratings   │                    │
              │  on SOC + tmdTaskID      │◄───────────────────┘
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │ Join Occupation Titles   │
              │     to Task SOCs         │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │ Clean/Standardize Titles │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  Assign LLM (tmdSOC-     │
              │        Twist)            │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │ Assign Task Importance   │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │   Filter Target SOCs     │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │     Write Manifest:      │
              │  sampled_tasks.csv       │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │   Parse Manifest CSV     │
              └──────────┬───────────────┘
                         │
═══════════════════════════╪═══════════════════════════════════════════════════════════════════════
                           ▼            PROMPT GENERATION STAGE                                    
═══════════════════════════════════════════════════════════════════════════════════════════════════

              ┌──────────────────────────┐
              │    For Each Task        │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │   For Each Modality     │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │ For Each Prompt Variant │
              │      (v0, v1, v2)          │
              └──────────┬───────────────┘
                         │
         ┌───────────────┼───────────────┬───────────────┐
         ▼               ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │   Make   │    │   Make   │    │   Make   │    │   Make   │
   │   TEXT   │    │   GUI    │    │  VISION  │    │  MANUAL  │
   │  Prompt  │    │ Prompt + │    │ Prompt + │    │  Prompt  │
   │          │    │selectors │    │  images  │    │          │
   └─────┬────┘    └─────┬────┘    └─────┬────┘    └─────┬────┘
         ▼               ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
   │  Save to │    │  Save to │    │  Save to │    │  Save to │
   │ prompts/ │    │ prompts/ │    │ prompts/ │    │ prompts/ │
   │   text   │    │   gui    │    │  vision  │    │  manual  │
   └─────┬────┘    └─────┬────┘    └─────┬────┘    └─────┬────┘
         │               │               │               │
═════════╪═══════════════╪═══════════════╪═══════════════╪═════════════════════════════════════════
         └───────────────┴───────────────┴───────────────┘
                                 ▼      INFERENCE LOOP                                            
═══════════════════════════════════════════════════════════════════════════════════════════════════

              ┌──────────────────────────┐
              │ For Each Model Provider │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │    For Each Model       │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │ For Each Modality (loop)│
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  For Each Prompt File   │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  Attach Modality Extras │
              │    (by archetype)       │
              └──────────┬───────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────┐      ┌─────────────────┐
   │   If VISION:    │      │    If GUI:      │
   │ Attach images   │      │Attach selectors │
   │(auto by vision  │      │  (auto by GUI   │
   │  archetype)     │      │   archetype)    │
   └────────┬────────┘      └────────┬────────┘
            └────────────┬────────────┘
                         ▼
              ┌──────────────────────────┐
              │     Call Model API      │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │  Log meta (buttons,     │
              │     temp, etc)          │
              └──────────┬───────────────┘
                         ▼
              ┌──────────────────────────┐
              │   Store to results/     │
              └──────────┬───────────────┘
                         │
═══════════════════════════╪═══════════════════════════════════════════════════════════════════════
                           ▼        VALIDATION & QUALITY CHECKS                                    
═══════════════════════════════════════════════════════════════════════════════════════════════════

              ┌──────────────────────────┐
              │  Wrapper Strict Check   │
              └──────────┬───────────────┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   ┌─────────────────────┐  ┌──────────────────────┐
   │wrapper_text_gui_per │  │ bad_outputs_strict   │
   │    _output.csv      │  │       .csv           │
   └─────────────────────┘  └──────────────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │ Wrapper Rescore Check   │
              └──────────┬───────────────┘
                         │
         ┌───────────────┴───────────────┐
         ▼                               ▼
   ┌──────────────┐              ┌──────────────┐
   │Schema (Text/ │              │    Schema    │
   │GUI) Strict   │              │   (Vision)   │
   └──────┬───────┘              │    Strict    │
          ▼                      └──────┬───────┘
   ┌──────────────┐                     ▼
   │Schema (Text/ │              ┌──────────────┐
   │GUI) Rescored │              │    Schema    │
   └──────┬───────┘              │   (Vision)   │
          ▼                      │   Rescored   │
   ┌──────────────┐              └──────┬───────┘
   │schema_text_  │                     ▼
   │failures.csv  │              ┌──────────────┐
   └──────────────┘              │   vision_    │
          │                      │failures.csv  │
          │                      └──────────────┘
          │                             │
          └──────────┬──────────────────┘
                     ▼
           ┌──────────────────┐
           │  Safety Strict   │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │ Safety Rescored  │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │Safety fail log   │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │Candidate Filter: │
           │ pass schema +    │
           │    safety        │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │   phase_03_      │
           │candidates.csv    │
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │ Rubric Grader    │
           │(e.g. GPT-4o-mini)│
           └────────┬─────────┘
                    ▼
           ┌──────────────────┐
           │phase_03_rubric_  │
           │ per_output.csv   │
           └────────┬─────────┘
                    │
═══════════════════════╪═══════════════════════════════════════════════════════════════════════════
                       ▼           AGGREGATION STAGE                                              
═══════════════════════════════════════════════════════════════════════════════════════════════════

         ┌──────────────────────────────────┐
         │ Combine/merge all pre-phase      │◄──┐
         │         outputs                  │   │
         └────────────┬─────────────────────┘   │
                      ▼                          │
         ┌──────────────────────────────────┐   │
         │    master_per_output.csv         │   │
         └────────────┬─────────────────────┘   │
                      ▼                          │ (All previous outputs feed here)
         ┌──────────────────────────────────┐   │
         │  Aggregate completions/models/   │   │
         │    occupation summaries          │   │
         └────────────┬─────────────────────┘   │
                      ▼                          │
         ┌──────────────────────────────────┐   │
         │ phase_04_composite_per_output    │───┘
         │            .csv                  │
         └────────────┬─────────────────────┘
                      │
═══════════════════════╪═══════════════════════════════════════════════════════════════════════════
                      ▼        DOWNSTREAM ARTIFACTS                                               
═══════════════════════════════════════════════════════════════════════════════════════════════════

         ┌────────────┴────────────┬────────────────┐
         ▼                         ▼                ▼
   ┌──────────────┐      ┌──────────────┐   ┌──────────────┐
   │    task_     │      │  job_level_  │   │ automation_  │
   │ capability_  │      │  scores.csv  │   │classification│
   │  scores.csv  │      │              │   │    .csv      │
   └──────┬───────┘      └──────┬───────┘   └──────┬───────┘
          └──────────────────────┼──────────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │  Build occupation-panel     │
                   │ (TACI + skills/important/  │
                   │          etc)               │
                   └─────────────┬───────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │   occupations_panel.csv     │
                   └─────────────┬───────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │     Do/Exhibit StudyN       │
                   └─────────────┬───────────────┘
                                 ▼
                   ┌─────────────────────────────┐
                   │   Analyse tables/plots      │
                   └─────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════════════════
```

## Legend

- `┌─────┐` = Process/Action boxes
- `▼ ▲ ◄ ►` = Flow direction arrows  
- `───` = Connections between processes
- `═══` = Stage separators
- Files ending in `.txt` or `.csv` = Data files
- Indented boxes = Sub-processes or outputs

## Directory Structure

```
TACI_Official/
├── README.md                           # This file - project overview and documentation
├── LICENSE                             # Project license
├── requirements.txt                    # Python dependencies
├── PATH_MAPPING.py                     # Path mapping utilities for reorganization
├── MIGRATION_*.md                      # Migration and reorganization documentation
├── REORGANIZATION_PLAN.md              # Detailed plan for code restructuring
├── SAFE_REORGANIZATION_PLAN.md         # Safe migration strategy
│
├── assets/                             # Static assets and reference images
│   └── images/                         # Images organized by archetype
│       ├── chest_xray/                 # Medical imaging samples (2 files)
│       ├── classification/             # Classification task images (2 files)
│       └── parcel_qc/                  # Quality control images (2 files)
│
├── ci/                                 # Continuous integration configuration
│
├── config/                             # Configuration files
│   ├── archetype_rules.yml             # GUI task archetype classification rules
│   ├── gui_selectors.json              # GUI element selectors for automation
│   └── vision_archetypes.yml           # VISION task archetype classification rules
│
├── dashboard/                          # Web dashboard components (if applicable)
│
├── data/                               # Data storage and manifests
│   ├── labour/                         # Labor economics data
│   ├── manifests/                      # Task manifests and metadata
│   │   ├── modality_cache*.json        # Cached modality classifications
│   │   └── sampled_tasks_comprehensive.csv # Main task manifest
│   ├── onet_raw/                       # Raw O*NET occupational data
│   │   ├── Occupation_Data.txt         # O*NET occupation definitions
│   │   ├── Task_Ratings.txt            # Task importance ratings
│   │   └── Task_Statements.txt         # Task descriptions
│   ├── panel/                          # Panel data for analysis
│   └── vision_demo/                    # Vision task demonstration data
│
├── docs/                               # Documentation
│   ├── README.md                       # Additional documentation
│   ├── TACI_AdvisorBrief_v01.pdf      # Project brief and methodology
│   ├── api/                            # API documentation
│   ├── examples/                       # Usage examples
│   └── tutorials/                      # Step-by-step tutorials
│
├── outputs/                            # Model evaluation results
│   ├── anthropic/                      # Claude model outputs
│   │   ├── claude-3-5-sonnet-20240620/ # Organized by model version
│   │   └── claude-3-opus-20240229/     # Multiple evaluation runs per task
│   ├── gemini/                         # Google Gemini outputs
│   │   ├── gemini-2.0-flash/          # Multiple model versions
│   │   └── gemini-2.5-flash-preview-05-20/
│   ├── groq_batch_rate_limited/        # Groq API results (rate limited)
│   ├── openai/                         # OpenAI model outputs
│   ├── results/                        # Processed results and analysis
│   └── runs/                           # Individual evaluation runs
│
├── prompts/                            # Generated prompts organized by type
│   ├── gui/                            # GUI task prompts (6 files)
│   ├── manual/                         # Manually created prompts (15 files)
│   ├── one_occ/                        # Single occupation prompts
│   ├── text/                           # Text-based task prompts (72 files)
│   └── vision/                         # Vision task prompts (6 files)
│
├── scripts/                            # Standalone utility scripts
│   ├── run_analysis.py                 # Main analysis runner
│   ├── tests/                          # Test scripts
│   │   └── test_dummy.py               # Basic test file
│   └── update_paths.py                 # Path update utilities
│
├── src/                                # Source code - main framework
│   ├── analysis/                       # Analysis and visualization tools
│   │   ├── __init__.py
│   │   ├── insights_analyzer.py        # Systematic insights extraction
│   │   ├── taci_insights_analyzer.py   # TACI-specific analysis
│   │   ├── visualizations.py           # Graph and chart generation
│   │   └── Econometrics/               # Economic analysis
│   │       ├── epochs.py               # Time-series analysis
│   │       └── results/                # Econometric results
│   │
│   ├── data_pipeline/                  # Data processing pipeline
│   │   ├── prompt_gen/                 # Prompt generation
│   │   │   ├── generate_prompts.py     # Main prompt generator
│   │   │   └── prompts_one_occ.py      # Single occupation prompts
│   │   └── sampling/                   # Task sampling and selection
│   │       └── sample_tasks.py         # Task sampling with modality classification
│   │
│   ├── evaluation/                     # 5-phase evaluation pipeline
│   │   ├── __init__.py
│   │   ├── data_processing/            # Data preparation for evaluation
│   │   │   ├── build_master_table.py   # Consolidate evaluation results
│   │   │   └── build_master_table_paralegal.py # Paralegal-specific tables
│   │   │
│   │   ├── phase_00_wrapper/           # Phase 0: Wrapper validation
│   │   │   ├── phase_00_wrapper_checker.py # General wrapper validation
│   │   │   └── phase_00_wrapper_checker_paralegal.py # Paralegal wrapper validation
│   │   │
│   │   ├── phase_01_schema/            # Phase 1: Schema validation
│   │   │   ├── phase_01_schema_grader.py # JSON schema validation
│   │   │   ├── phase_01_schema_grader_paralegal.py # Paralegal schema validation
│   │   │   ├── phase_01_vision.py      # Vision-specific schema validation
│   │   │   └── schemas/                # JSON schemas for validation
│   │   │       └── GUI.json            # GUI task schema
│   │   │
│   │   ├── phase_02_safety/            # Phase 2: Safety evaluation
│   │   │   ├── phase_02_safety_grader.py # General safety assessment
│   │   │   └── phase_02_safety_grader_paralegal.py # Paralegal safety assessment
│   │   │
│   │   ├── phase_03_rubric/            # Phase 3: Rubric-based scoring
│   │   │   ├── phase_03_filter.py      # Result filtering
│   │   │   ├── phase_03_filter_one_occ.py # Single occupation filtering
│   │   │   ├── rubric_grader.py        # General rubric evaluation
│   │   │   ├── rubric_grader_paralegal.py # Paralegal rubric evaluation
│   │   │   └── test.py                 # Testing utilities
│   │   │
│   │   └── phase_04_composite/         # Phase 4: Composite scoring
│   │       ├── final_composite_Paralegal.py # Paralegal final scoring
│   │       ├── phase_04_grader.py      # General composite scoring
│   │       └── weights.json            # Scoring weights configuration
│   │
│   ├── execution/                      # Model execution and batch processing
│   │   └── llm_client/                 # LLM client interfaces
│   │       ├── run_batch_anthropic.py  # Anthropic Claude batch runner
│   │       ├── run_batch_gemini.py     # Google Gemini batch runner
│   │       ├── run_batch_llama3.py     # Llama 3 batch runner
│   │       ├── run_batch_openai.py     # OpenAI batch runner
│   │       └── test_runners/           # Testing and specialized runners
│   │           ├── o3_failed.py        # Handle failed O3 model runs
│   │           ├── o3_special.py       # Special O3 model configurations
│   │           ├── one_occ.py          # Single occupation testing
│   │           ├── test_batch_anthropic.py # Anthropic testing
│   │           ├── test_batch_gemini.py # Gemini testing
│   │           └── test_one.py         # Single task testing
│   │
│   └── utils/                          # Utility functions and scripts
│       ├── assign_archetypes.py        # Assign GUI archetypes to tasks
│       ├── auto_tag_vision.py          # Assign VISION archetypes to tasks
│       ├── build_manual_stub_catalog.py # Manual task catalog generation
│       ├── build_vision_stub_catalog.py # Vision task catalog generation
│       ├── check_prompt_wrappers.py    # Validate prompt wrappers
│       ├── convert_mammograms*.py      # Medical image conversion utilities
│       ├── count_phase2_calls.py       # Count safety evaluation calls
│       ├── fix_bad_runs.py             # Repair corrupted evaluation runs
│       ├── graph.py                    # Graph generation utilities
│       ├── image_changes_*.py          # Image processing utilities
│       ├── renaming.py                 # File and variable renaming utilities
│       ├── robust_extractor.py         # Robust data extraction
│       ├── sample_for_human_review.py  # Sample tasks for human evaluation
│       ├── testingsafety.py            # Safety testing utilities
│       ├── testingtokens.py            # Token counting and testing
│       └── vision_guard_test.py        # Vision safety testing
│
├── tests/                              # Test data and human evaluation
│   ├── human audit samples/            # Human-audited task samples
│   └── old_runs_on_anth_and_llama_with_extr/ # Historical evaluation runs
│
├── vision_refs/                        # Vision task reference data
│   ├── ce0be4e8_GT.json               # Ground truth for vision tasks
│   └── d21fc252_GT.json               # Vision evaluation references
│
├── website/                            # Web interface (if applicable)
│   ├── index.html                      # Main web interface
│   ├── script.js                       # JavaScript functionality
│   └── styles.css                      # Web styling
│
├── taci_analysis_report.json           # Generated analysis report
├── taci_paralegal_*.png                # Visualization outputs
└── wrapper_per_output_paralegal.csv    # Paralegal evaluation wrapper data
```

## Key Components

### Data Pipeline (`src/data_pipeline/`)
- **Task sampling and classification** across 20+ occupations with NLP-powered modality detection
- **Multi-variant prompt generation** for TEXT, GUI, VISION, and MANUAL task types

### Evaluation Pipeline (`src/evaluation/`)
**5-Phase Statistical Validation Framework:**
- **Phase 0**: **Wrapper compliance** - Multi-vendor response format validation with strict/rescued extraction
- **Phase 1**: **Schema validation** - Formal JSON Schema compliance + computer vision IoU scoring  
- **Phase 2**: **Safety assessment** - OpenAI moderation API with custom risk weighting and thresholds
- **Phase 3**: **Multi-axis rubric scoring** - 6-dimensional evaluation (accuracy, coverage, depth, style, utility, specificity) with 3-vote consensus
- **Phase 4**: **Weighted composite scoring** - AHP methodology with bootstrap confidence intervals on 0-100 scale

### Model Execution (`src/execution/`)
- **Multi-provider API orchestration** with enterprise-grade error handling and retry logic
- **Batch processing systems** supporting Anthropic, OpenAI, Google, and Meta LLM providers
- **Comprehensive provenance tracking** with git commit logging and experimental reproducibility

### Analysis Tools (`src/analysis/`)
**Research-Grade Analytics Framework:**
- **`insights_analyzer.py`**: Systematic performance pattern extraction with statistical significance testing
- **`visualizations.py`**: Multi-dimensional capability visualization and comparative analysis
- **`Econometrics/`**: Economic impact modeling and automation potential assessment for policy research

### Configuration (`config/`)
- **`archetype_rules.yml`**: Rules for classifying GUI tasks into archetypes
- **`vision_archetypes.yml`**: Rules for classifying VISION tasks into archetypes
- **`gui_selectors.json`**: GUI element selectors for automation tasks

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Set Environment Variables**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export ANTHROPIC_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   ```

3. **Sample Tasks**:
   ```bash
   python src/data_pipeline/sampling/sample_tasks.py
   ```

4. **Generate Prompts**:
   ```bash
   python src/data_pipeline/prompt_gen/generate_prompts.py
   ```

5. **Run Evaluation**:
   ```bash
   python src/execution/llm_client/run_batch_anthropic.py
   ```

6. **Analyze Results**:
   ```bash
   python src/analysis/insights_analyzer.py
   ```

## Research Contribution

**First Systematic AI Occupational Capability Benchmark**

> **Note**: This repository represents the **alpha/showcase version** of TACI for demonstration and open research. The internal production version includes expanded occupational coverage, additional model providers, and advanced experimental features.

TACI establishes novel methodology for quantitative assessment of AI automation potential across professional domains. This demonstration version includes:

- **Paralegal occupation proof-of-concept** with comprehensive task coverage across all major LLM providers
- **Multi-modal evaluation framework** integrating text analysis, GUI automation, and visual document processing
- **Statistical validation pipeline** demonstrating reproducible methodology for occupational AI assessment
- **Economic impact modeling** framework supporting automation potential quantification

**Internal Research Extensions**: The production TACI system evaluates 40+ occupations across 15+ model providers with advanced experimental features including adaptive prompting, cross-occupational transfer analysis, and longitudinal capability tracking.

**Academic Impact**: Framework designed for reproducible research with full experimental provenance, supporting publication-quality analysis of AI capabilities across cognitive work categories.

**Policy Applications**: Quantitative automation impact assessment enabling evidence-based workforce transition planning and technology deployment strategies.

## File Naming Conventions

- **Task UIDs**: 8-character MD5 hash of SOC code + Task ID (e.g., `07080553`)
- **Model Outputs**: `{uid}_v{variant}_t{temperature}_{run}.json`
- **Prompts**: `{uid}_v{variant}.json`
- **Results**: Organized by model provider and version
