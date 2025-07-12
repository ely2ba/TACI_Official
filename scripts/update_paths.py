#!/usr/bin/env python3
"""
Path Update Script
=================

Updates hardcoded paths in Python files to use new structure.
Can be run safely multiple times.

Usage:
    python scripts/update_paths.py --dry-run    # Show what would change
    python scripts/update_paths.py --execute    # Actually make changes
"""

import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Project root
ROOT = Path(__file__).parent.parent

# Path replacements (old -> new)
PATH_REPLACEMENTS = {
    # Graders to evaluation
    r'"evaluation/phase_00_wrapper/': r'"evaluation/phase_00_wrapper/',
    r'"evaluation/phase_01_schema/': r'"evaluation/phase_01_schema/',
    r'"evaluation/phase_02_safety/': r'"evaluation/phase_02_safety/', 
    r'"evaluation/phase_03_rubric/': r'"evaluation/phase_03_rubric/',
    r'"evaluation/phase_04_composite/': r'"evaluation/phase_04_composite/',
    r'"evaluation/phase_04_composite/': r'"evaluation/phase_04_composite/',
    
    # Single quotes
    r"'evaluation/phase_00_wrapper/": r"'evaluation/phase_00_wrapper/",
    r"'evaluation/phase_01_schema/": r"'evaluation/phase_01_schema/",
    r"'evaluation/phase_02_safety/": r"'evaluation/phase_02_safety/",
    r"'evaluation/phase_03_rubric/": r"'evaluation/phase_03_rubric/",
    r"'evaluation/phase_04_composite/": r"'evaluation/phase_04_composite/",
    r"'evaluation/phase_04_composite/": r"'evaluation/phase_04_composite/",
    
    # Path objects
    r'Path\("graders/': r'Path("evaluation/',
    r"Path\('graders/": r"Path('evaluation/",
    
    # File references that were renamed
    r'phase_00_warpper_checker_one_occ\.py': r'phase_00_wrapper_checker_paralegal.py',
    r'phase_01_schema_grader_one_occ\.py': r'phase_01_schema_grader_paralegal.py', 
    r'phase_02_safety_one_occ\.py': r'phase_02_safety_grader_paralegal.py',
    r'rubric_grader_one_occ\.py': r'rubric_grader_paralegal.py',
}

def find_files_to_update() -> List[Path]:
    """Find Python files that contain graders/ references"""
    files = []
    
    # Search all Python files
    for py_file in ROOT.rglob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')
            if 'graders/' in content:
                files.append(py_file)
        except (UnicodeDecodeError, PermissionError):
            pass
            
    return files

def update_file_paths(file_path: Path, dry_run: bool = True) -> Tuple[bool, List[str]]:
    """Update paths in a single file"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        changes = []
        
        # Apply all replacements
        for old_pattern, new_pattern in PATH_REPLACEMENTS.items():
            matches = re.findall(old_pattern, content)
            if matches:
                content = re.sub(old_pattern, new_pattern, content)
                changes.append(f"  {old_pattern} → {new_pattern} ({len(matches)} matches)")
        
        # Write changes if not dry run
        if not dry_run and content != original_content:
            file_path.write_text(content, encoding='utf-8')
            
        return content != original_content, changes
        
    except Exception as e:
        return False, [f"  ERROR: {e}"]

def main():
    parser = argparse.ArgumentParser(description="Update TACI file paths")
    parser.add_argument("--dry-run", action="store_true", default=True,
                       help="Show changes without applying them")
    parser.add_argument("--execute", action="store_true", 
                       help="Actually make the changes")
    
    args = parser.parse_args()
    
    if args.execute:
        dry_run = False
        print("🔧 EXECUTING path updates...")
    else:
        dry_run = True
        print("👀 DRY RUN - showing what would change...")
    
    print("=" * 50)
    
    # Find files to update
    files_to_update = find_files_to_update()
    print(f"Found {len(files_to_update)} files with graders/ references")
    
    total_files_changed = 0
    total_changes = 0
    
    # Update each file
    for file_path in files_to_update:
        changed, changes = update_file_paths(file_path, dry_run)
        
        if changed:
            total_files_changed += 1
            total_changes += len(changes)
            
            print(f"\n📁 {file_path.relative_to(ROOT)}")
            for change in changes:
                print(change)
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"  Files with changes: {total_files_changed}")
    print(f"  Total replacements: {total_changes}")
    
    if dry_run:
        print(f"\n💡 Run with --execute to apply changes")
    else:
        print(f"\n✅ Changes applied successfully!")

if __name__ == "__main__":
    main()