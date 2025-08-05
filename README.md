# TACI (Task-AI Capability Index)

A comprehensive evaluation framework for assessing AI model capabilities across real-world occupational tasks from O*NET data. TACI evaluates models on TEXT, GUI, VISION, and MULTI-modal tasks through a rigorous 5-phase pipeline.

## Overview

TACI systematically evaluates AI models by:
- Sampling tasks from 20+ diverse occupations using O*NET occupational data
- Classifying tasks by modality (TEXT/GUI/VISION) using GPT-4 consensus voting
- Running models through a 5-phase evaluation pipeline
- Analyzing performance patterns and generating insights

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
- **`sampling/sample_tasks.py`**: Samples tasks from O*NET data across 20+ occupations, classifies modality using GPT-4 consensus voting
- **`prompt_gen/generate_prompts.py`**: Generates evaluation prompts for different task types and modalities

### Evaluation Pipeline (`src/evaluation/`)
- **Phase 0**: Wrapper validation - ensures model responses are properly formatted
- **Phase 1**: Schema validation - validates JSON structure against defined schemas
- **Phase 2**: Safety evaluation - assesses responses for safety and appropriateness
- **Phase 3**: Rubric scoring - evaluates task completion quality using detailed rubrics
- **Phase 4**: Composite scoring - combines all phases into final capability scores

### Model Execution (`src/execution/`)
- Batch runners for major LLM providers (Anthropic, OpenAI, Google, Meta)
- Specialized test runners for debugging and single-task evaluation
- Support for vision tasks with image inputs

### Analysis Tools (`src/analysis/`)
- **`insights_analyzer.py`**: Systematic extraction of performance insights
- **`visualizations.py`**: Generate graphs and charts for results
- **`Econometrics/`**: Economic analysis of automation potential

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

## Research Focus

Current research emphasizes paralegal task automation, with comprehensive evaluation across:
- Legal document analysis and generation
- Client communication and case management
- Regulatory compliance and research
- Multi-modal task performance (text + GUI + vision)

## File Naming Conventions

- **Task UIDs**: 8-character MD5 hash of SOC code + Task ID (e.g., `07080553`)
- **Model Outputs**: `{uid}_v{variant}_t{temperature}_{run}.json`
- **Prompts**: `{uid}_v{variant}.json`
- **Results**: Organized by model provider and version