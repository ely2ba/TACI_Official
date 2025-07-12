#!/usr/bin/env python3
"""
TACI Analysis Runner
==================

Modern entry point for TACI analysis that works with existing code structure.
This script provides a clean interface while maintaining compatibility.

Usage:
    python scripts/run_analysis.py              # Run insights analysis
    python scripts/run_analysis.py --graph      # Generate graph  
    python scripts/run_analysis.py --both       # Run both
"""

import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run TACI analysis")
    parser.add_argument("--graph", action="store_true", help="Generate visualization")
    parser.add_argument("--both", action="store_true", help="Run both analysis and graph")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print("🔬 TACI Analysis Runner")
    print("=" * 50)
    
    # Run insights analysis (default)
    if not args.graph or args.both:
        print("\n📊 Running systematic insights analysis...")
        insights_script = project_root / "taci_insights_analyzer.py"
        
        if insights_script.exists():
            try:
                subprocess.run([sys.executable, str(insights_script)], 
                             cwd=project_root, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Analysis failed: {e}")
                return 1
        else:
            print("❌ Insights analyzer not found")
            return 1
    
    # Generate graph
    if args.graph or args.both:
        print("\n📈 Generating visualization...")
        graph_script = project_root / "graph.py"
        
        if graph_script.exists():
            try:
                subprocess.run([sys.executable, str(graph_script)], 
                             cwd=project_root, check=True)
                print("✅ Graph generated successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Graph generation failed: {e}")
                return 1
        else:
            print("❌ Graph script not found")
            return 1
    
    print("\n✅ Analysis complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())