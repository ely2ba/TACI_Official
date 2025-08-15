# TACI (Task-AI Capability Index)

A novel research framework establishing systematic methodology for evaluating AI automation potential across professional occupations through statistically rigorous 5-phase assessment pipeline. TACI provides the first comprehensive benchmark for measuring AI model capabilities on real-world occupational tasks with production-grade engineering and academic rigor.

## Overview

TACI evaluates AI models through **7,600+ lines of research-grade Python** implementing:
- **Multi-modal assessment** across TEXT, GUI, and VISION tasks with computer vision IoU scoring
- **Statistical rigor** via 3-vote self-consistency validation and bootstrap confidence intervals  
- **Production-scale orchestration** of Anthropic Claude, OpenAI GPT, Google Gemini, and Meta Llama APIs
- **Systematic evaluation methodology** through 5-phase pipeline with weighted composite scoring
- **Economic impact modeling** for automation potential across 20+ professional occupations

# Process Flowchart

## Visual Diagram


flowchart TD
    Start1[New_SessionID,lib]
    Start2[New_PollApp,lib]
    
    Start1 --> Unvote[Unvotation_ballot.txt]
    Start2 --> Unvote
    
    Unvote --> Extract[Extract Unvotation Titles]
    
    Extract --> Merge[Merge Titles + Change<br/>in order set to blank]
    
    Merge --> SetHelp[Set help text to<br/>help.html]
    
    SetHelp --> Random[Randomize Files]
    
    Random --> Delete[Delete old vote data files]
    
    Delete --> Initiate[Initiate Text Initialization]
    
    Initiate --> Ready[Ready For Input]
    
    Ready --> InputWait{Input detected<br/>session_ballot.txt}
    
    InputWait --> Prompt[Prompt Username Entry]
    
    Prompt --> PollEntry[Poll Entry Page]
    
    PollEntry --> PollStart[Poll Entry Functions]
    
    PollStart --> FullPage[Full Page Prompt Display]
    
    %% Multiple parallel processes
    FullPage --> Process1[Make POLL Project]
    FullPage --> Process2[Make GUI Prompt + selections]
    FullPage --> Process3[Make ORIGIN Prompt + image]
    FullPage --> Process4[Make MANUAL Prompt]
    
    Process1 --> Preselected1{Want to preselect/select}
    Process2 --> Preselected2{Want to preselect/select}
    Process3 --> Preselected3{Want to preselect/select}
    Process4 --> Preselected4{Want to preselect/select}
    
    %% Poll Entry convergence
    Preselected1 --> PollEntryScrum[Poll Entry Scrum]
    Preselected2 --> PollEntryScrum
    Preselected3 --> PollEntryScrum
    Preselected4 --> PollEntryScrum
    
    PollEntryScrum --> PollForm[Poll Form Display]
    
    PollForm --> PollFormListener[Poll Form Listener]
    
    PollFormListener --> Submit[Submit]
    
    Submit --> AttachPrompt[Attach Manifest Prompt to<br/>submission]
    
    %% Branching logic
    AttachPrompt --> CheckReplace{If replace was dragged<br/>onto to form submission}
    AttachPrompt --> CheckDelete{If the delete selection<br/>went to form submission}
    
    CheckReplace --> Replace[Replace File]
    CheckDelete --> Delete2[Delete Entry]
    
    Replace --> LogAction[Log code actions done, etc?]
    Delete2 --> LogAction
    
    LogAction --> ReadyLoop{Want to restart?}
    
    ReadyLoop -->|Yes| Random
    ReadyLoop -->|No| SessionEnd[Session Data export]
    
    %% Nested sections with different processes
    SessionEnd --> NestedSection1[Dynamic Reshuffle Library]
    SessionEnd --> NestedSection2[Summary History Report]
    SessionEnd --> NestedSection3[Summary Report Request]
    
    NestedSection1 --> SubProcess1[creating_new_poll_with_output.txt]
    NestedSection2 --> SubProcess2[reference_text_inference.txt]
    NestedSection3 --> SubProcess3[survey_questions.txt]
    
    SubProcess1 --> Library[Library Dump]
    SubProcess2 --> Library
    SubProcess3 --> Library
    
    Library --> NestedLoop1{Select Priorities?}
    
    NestedLoop1 -->|Yes| Priority1[Priority UI display poll selection]
    NestedLoop1 -->|No| Continue1[Continue]
    
    Priority1 --> SetPriority[Set priority]
    SetPriority --> NestedLoop1
    
    Continue1 --> CandidateFilter[Candidate Filter pass internal<br/>process]
    
    CandidateFilter --> Pruned[pruned_lib_candidates.csv]
    
    Pruned --> ScoreSub[Priority Subset lib SET descriptor]
    
    ScoreSub --> PruneMore[Prune_lib_items_poll_output.csv]
    
    PruneMore --> FinalMerge[Merge output of<br/>prune_outputs.txt]
    
    FinalMerge --> GenerateReport[Generate report]
    
    GenerateReport --> ConfigureOutput[Configure output to<br/>add polls to existing report]
    
    ConfigureOutput --> OutputFormat{Output_lib_pollsumdb_poll_output.html}
    
    OutputFormat --> Browser[browser.lib}
    OutputFormat --> Schedule[cron_job_scheduler_daemon.txt]
    OutputFormat --> WebLink[web_server.lib]
    OutputFormat --> EmailReminder[auto_email_reminders]
    
    Browser --> NextPhase[Next poll libraries]
    Schedule --> NextPhase
    WebLink --> NextPhase
    EmailReminder --> NextPhase
    
    NextPhase --> Complete[Finalized executables]
```

## Process Description

This flowchart represents a polling/voting system workflow with the following main phases:

### 1. **Initialization Phase**
- Creates new session and poll application instances
- Processes unvotation ballot data
- Extracts and merges titles
- Sets up help text and randomizes files
- Cleans up old vote data

### 2. **User Input Phase**
- Waits for input detection (session_ballot.txt)
- Prompts for username entry
- Displays poll entry page

### 3. **Poll Configuration Phase**
- Creates multiple parallel processes:
  - POLL Project setup
  - GUI Prompt with selections
  - ORIGIN Prompt with image
  - MANUAL Prompt
- Each process allows for preselection options

### 4. **Poll Execution Phase**
- Consolidates selections in Poll Entry Scrum
- Displays poll form
- Listens for form submissions
- Attaches manifest prompts to submissions

### 5. **Action Processing Phase**
- Handles replace/delete operations
- Logs code actions
- Offers restart option

### 6. **Report Generation Phase**
- Exports session data
- Generates three types of reports:
  - Dynamic Reshuffle Library
  - Summary History Report
  - Summary Report Request

### 7. **Library Processing Phase**
- Dumps data to library
- Allows priority selection (optional)
- Applies candidate filtering
- Prunes and merges outputs

### 8. **Output Distribution Phase**
- Generates final report
- Configures output for existing reports
- Distributes through multiple channels:
  - Browser interface
  - Scheduled cron jobs
  - Web server
  - Email reminders

### 9. **Completion**
- Finalizes executables for next poll libraries

## Implementation Notes

- Decision points are indicated with diamond shapes in the flowchart
- Parallel processes converge at the Poll Entry Scrum
- The system includes both manual and automated pathways
- Loop-back functionality allows for process restart at various stages

## Technical Innovation

### **Statistical Sophistication**
- **3-vote consensus validation** across evaluation phases with deterministic seeding
- **Bootstrap confidence intervals** (95% CI) for uncertainty quantification
- **AHP-weighted composite scoring** preserving pairwise importance ratios
- **Multi-tier safety assessment** with custom risk thresholds and moderation API integration

### **Production Engineering**
- **Enterprise-grade API orchestration** with exponential backoff, retry logic, and graceful degradation
- **Vendor-agnostic response parsing** supporting 4+ LLM provider formats with unified extraction
- **Comprehensive caching system** using SHA-256 integrity checking and persistent storage
- **Git commit provenance tracking** ensuring full experimental reproducibility

### **Research Methodology** 
- **Formal JSON Schema validation** with Draft7Validator compliance testing
- **Computer vision evaluation** using Intersection-over-Union (IoU) metrics for spatial tasks
- **NLP-powered modality classification** via spaCy linguistic analysis and GPT-4 consensus
- **Econometric analysis framework** supporting automation impact research and policy analysis

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
**5-Phase Statistical Validation Framework:**
- **Phase 0**: **Wrapper compliance** - Multi-vendor response format validation with strict/rescued extraction
- **Phase 1**: **Schema validation** - Formal JSON Schema compliance + computer vision IoU scoring  
- **Phase 2**: **Safety assessment** - OpenAI moderation API with custom risk weighting and thresholds
- **Phase 3**: **Multi-axis rubric scoring** - 6-dimensional evaluation (accuracy, coverage, depth, style, utility, specificity) with 3-vote consensus
- **Phase 4**: **Weighted composite scoring** - AHP methodology with bootstrap confidence intervals on 0-100 scale

### Model Execution (`src/execution/`)
**Production-Grade LLM Orchestration:**
- **Enterprise API integration** for Anthropic Claude, OpenAI GPT, Google Gemini, Meta Llama
- **Robust batch processing** with exponential backoff, retry logic, and error recovery
- **Multi-modal support** including vision tasks with image inputs and spatial reasoning
- **Comprehensive logging** with git commit tracking and response provenance

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

TACI establishes novel methodology for quantitative assessment of AI automation potential across professional domains. Current research emphasizes paralegal task automation as proof-of-concept, with comprehensive evaluation across:

- **Legal document analysis and generation** with domain-specific accuracy metrics
- **Client communication and case management** workflows with multi-turn interaction assessment  
- **Regulatory compliance and research** tasks requiring factual precision and legal reasoning
- **Multi-modal professional workflows** integrating text analysis, GUI automation, and visual document processing

**Academic Impact**: Framework designed for reproducible research with full experimental provenance, supporting publication-quality analysis of AI capabilities across cognitive work categories.

**Policy Applications**: Quantitative automation impact assessment enabling evidence-based workforce transition planning and technology deployment strategies.

## File Naming Conventions

- **Task UIDs**: 8-character MD5 hash of SOC code + Task ID (e.g., `07080553`)
- **Model Outputs**: `{uid}_v{variant}_t{temperature}_{run}.json`
- **Prompts**: `{uid}_v{variant}.json`
- **Results**: Organized by model provider and version
