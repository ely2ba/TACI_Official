# SAFE TACI Repository Reorganization Plan

## 🚨 Current Dependencies Found

### Hard-coded paths that would break:
- **graders/**: 7+ files reference this path
- **data/**: 10+ files reference data/manifests, data/onet_raw
- **runs/**: 8+ files reference runs/openai, runs/gemini
- **assets/**: Referenced in visualization and image processing

## 🎯 SAFE Incremental Approach

Instead of massive restructuring, let's do **targeted improvements** that don't break existing functionality:

### Phase 1: Clean Root Directory (SAFE)
- Create `scripts/` folder for loose scripts
- Move standalone files that aren't referenced by other code:
  ```bash
  # SAFE to move (no dependencies found):
  mv renaming.py scripts/
  mv testingsafety.py scripts/
  mv testingtokens.py scripts/
  ```

### Phase 2: Add Professional Structure (ADDITIVE)
- Create new directories alongside existing ones
- Add new files without moving existing ones:
  ```
  fresh_TACI/
  ├── [existing structure unchanged]
  ├── analysis/           # NEW: Professional analysis tools
  │   ├── insights_analyzer.py  # Copy of taci_insights_analyzer.py
  │   └── visualizations.py     # Copy of graph.py
  ├── docs/              # NEW: Documentation
  └── requirements.txt   # NEW: Dependencies
  ```

### Phase 3: Create Modern Entry Points (NON-BREAKING)
- Add new convenience scripts that work with existing paths
- Keep all original files in place
- Add modern Python package structure

### Phase 4: Gradual Migration (OPTIONAL)
- Only if you want to migrate later
- Update paths one file at a time
- Test each change

## 🛡️ What We WON'T Break

### Keep These Paths Unchanged:
- `graders/phase_*/` - Used by 7+ Python files
- `data/manifests/` - Used by LLM clients
- `data/onet_raw/` - Used by sampling code
- `runs/` - Used by test runners and utilities
- `prompts/` - Generated and used by multiple systems

### Files with Hard Dependencies:
- `llm_client/` scripts
- `graders/` analysis scripts  
- `utils/` processing scripts
- `sampling/` data preparation

## ✅ SAFE Actions We Can Take Now

### 1. Clean Root Directory
```bash
# Create scripts directory
mkdir -p scripts/utilities

# Move standalone scripts (no dependencies)
mv renaming.py scripts/utilities/
mv testingsafety.py scripts/utilities/
mv testingtokens.py scripts/utilities/
```

### 2. Add Professional Structure
```bash
# Create analysis module
mkdir -p analysis
cp taci_insights_analyzer.py analysis/insights_analyzer.py
cp graph.py analysis/visualizations.py

# Add package files
touch analysis/__init__.py
```

### 3. Add Documentation
```bash
# Create docs structure
mkdir -p docs/{api,tutorials,examples}

# Add requirements and setup
# (create requirements.txt, setup.py)
```

### 4. Create Convenience Entry Points
```bash
# Add modern entry point that calls existing code
# scripts/run_analysis.py -> calls taci_insights_analyzer.py
```

## 🎯 Benefits of Safe Approach

1. **Zero Breakage**: All existing code continues to work
2. **Professional Polish**: Adds modern Python package structure
3. **Better Organization**: New code goes in logical places
4. **Incremental**: Can evolve gradually over time
5. **Backward Compatible**: Original scripts still work

## 📋 Implementation Checklist

- [ ] Create `scripts/utilities/` and move standalone files
- [ ] Create `analysis/` module with copies of key analysis tools
- [ ] Add `requirements.txt` and `setup.py`
- [ ] Create `docs/` structure
- [ ] Add modern entry points that call existing code
- [ ] Update README with new structure (but keep old paths working)

## 🔄 Future Migration (Optional)

If you later want to do full reorganization:
1. Create path mapping in a config file
2. Add path resolution utilities
3. Update files one by one with backward compatibility
4. Deprecate old paths gradually
5. Remove old structure only when confident

This approach gives you **professional organization** without the **risk of breaking everything**.