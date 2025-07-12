#!/usr/bin/env python3
"""
TACI Insights Analyzer
======================
Systematic analysis of AI model performance across paralegal automation tasks.
Automatically generates the key insights discovered in the TACI benchmark data.

Usage: python3 taci_insights_analyzer.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class TACIAnalyzer:
    def __init__(self, base_path="/Users/elyb/Desktop/Projects/TACI_Official"):
        self.base_path = Path(base_path)
        self.data = {}
        self.insights = []
        
    def load_data(self):
        """Load all relevant TACI data files"""
        try:
            # Load master output data
            master_file = self.base_path / "outputs/results/evaluated/master_per_output_paralegal.csv"
            self.data['master'] = pd.read_csv(master_file)
            
            # Load model grader matrix
            matrix_file = self.base_path / "outputs/results/evaluated/model_grader_matrix_paralegal.csv"
            self.data['matrix'] = pd.read_csv(matrix_file)
            
            # Load occupation summary
            summary_file = self.base_path / "outputs/results/evaluated/occupation_summary_paralegal.csv"
            self.data['summary'] = pd.read_csv(summary_file)
            
            print("✓ Successfully loaded all TACI data files")
            return True
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            return False
    
    def analyze_structural_reliability(self):
        """Analyze wrapper and schema failure rates across models using raw task data"""
        print("\n" + "="*60)
        print("STRUCTURAL RELIABILITY ANALYSIS")
        print("="*60)
        
        if 'master' not in self.data:
            return
            
        master = self.data['master']
        
        # Analyze per-task failures by model
        for model in master['model'].unique():
            model_data = master[master['model'] == model]
            total_tasks = len(model_data)
            
            # Count wrapper failures (wrapper_strict = 0)
            wrapper_failures = len(model_data[model_data['wrapper_strict'] == 0])
            wrapper_fail_rate = (wrapper_failures / total_tasks) * 100
            
            # Count schema failures (schema_strict = 0)
            schema_failures = len(model_data[model_data['schema_strict'] == 0])
            schema_fail_rate = (schema_failures / total_tasks) * 100
            
            # Count safety failures (safety_strict = 1, meaning failed safety)
            safety_failures = len(model_data[model_data['safety_strict'] == 1])
            safety_fail_rate = (safety_failures / total_tasks) * 100
            
            print(f"\n{model}:")
            print(f"  Total Tasks: {total_tasks}")
            print(f"  Wrapper Failures: {wrapper_failures}/{total_tasks} ({wrapper_fail_rate:.1f}%)")
            print(f"  Schema Failures:  {schema_failures}/{total_tasks} ({schema_fail_rate:.1f}%)")
            print(f"  Safety Failures:  {safety_failures}/{total_tasks} ({safety_fail_rate:.1f}%)")
            
            # Generate insights based on failure rates
            if wrapper_fail_rate > 10:
                self.insights.append(f"🚨 {model} has {wrapper_fail_rate:.1f}% wrapper failure rate - UNRELIABLE for production")
            elif wrapper_fail_rate > 5:
                self.insights.append(f"⚠️  {model} has {wrapper_fail_rate:.1f}% wrapper failure rate - concerning reliability")
            elif wrapper_fail_rate == 0:
                self.insights.append(f"✅ {model} achieves perfect wrapper compliance (0% failures)")
                
            if schema_fail_rate > 10:
                self.insights.append(f"🚨 {model} has {schema_fail_rate:.1f}% schema failure rate - UNRELIABLE structure")
            elif schema_fail_rate == 0:
                self.insights.append(f"✅ {model} achieves perfect schema compliance (0% failures)")
                
            # Combined structural reliability assessment
            total_structural_failures = wrapper_failures + schema_failures
            total_fail_rate = (total_structural_failures / total_tasks) * 100
            
            if total_fail_rate > 10:
                self.insights.append(f"❌ {model} FAILS structural reliability test ({total_fail_rate:.1f}% total failures)")
            elif total_fail_rate == 0:
                self.insights.append(f"🎯 {model} demonstrates PERFECT structural reliability")
    
    def analyze_capability_tiers(self):
        """Identify capability tiers and breakthrough thresholds"""
        print("\n" + "="*60)
        print("CAPABILITY TIER ANALYSIS")
        print("="*60)
        
        if 'summary' not in self.data:
            return
            
        summary = self.data['summary'].sort_values('mean_score')
        
        print("\nModel Performance Ranking:")
        for _, row in summary.iterrows():
            model = row['model']
            score = row['mean_score']
            ci_low = row['ci_low']
            ci_high = row['ci_high']
            
            # Determine tier
            if score >= 80:
                tier = "EXPERT"
                color = "🔵"
            elif score >= 75:
                tier = "PROFESSIONAL"
                color = "🟢"
            elif score >= 70:
                tier = "APPROACHING"
                color = "🟡"
            else:
                tier = "UNRELIABLE"
                color = "🔴"
            
            print(f"  {color} {model:<15} {score:5.1f} ({ci_low:.1f}-{ci_high:.1f}) - {tier}")
        
        # Find breakthrough jumps (sorted by score)
        summary_sorted = summary.sort_values('mean_score')
        scores = summary_sorted['mean_score'].values
        models = summary_sorted['model'].values
        
        print(f"\nBreakthrough Analysis:")
        for i in range(1, len(scores)):
            jump = scores[i] - scores[i-1]
            print(f"  {models[i-1]} → {models[i]}: +{jump:.1f} points")
            
            if jump > 5:  # Significant jump threshold
                self.insights.append(f"🚀 BREAKTHROUGH: {models[i]} shows +{jump:.1f} point jump over {models[i-1]}")
            elif jump > 3:
                self.insights.append(f"📈 Significant improvement: {models[i]} (+{jump:.1f} over {models[i-1]})")
                
        # Identify threshold crossings
        for _, row in summary.iterrows():
            score = row['mean_score']
            model = row['model']
            
            if 75 <= score < 80:
                self.insights.append(f"🎯 {model} crosses PROFESSIONAL THRESHOLD (75+ score)")
            elif score >= 80:
                self.insights.append(f"🌟 {model} achieves EXPERT LEVEL (80+ score)")
        
        # Identify the performance cliff between viable and non-viable models
        professional_models = summary[summary['mean_score'] >= 75]
        non_professional = summary[summary['mean_score'] < 75]
        
        if len(professional_models) > 0 and len(non_professional) > 0:
            best_non_prof = non_professional['mean_score'].max()
            worst_prof = professional_models['mean_score'].min()
            gap = worst_prof - best_non_prof
            
            best_non_prof_model = non_professional[non_professional['mean_score'] == best_non_prof]['model'].iloc[0]
            worst_prof_model = professional_models[professional_models['mean_score'] == worst_prof]['model'].iloc[0]
            
            self.insights.append(f"🏔️  CAPABILITY CLIFF: {gap:.1f} point gap between {best_non_prof_model} ({best_non_prof:.1f}) and {worst_prof_model} ({worst_prof:.1f})")
    
    def analyze_rubric_quality(self):
        """Analyze rubric dimension performance using raw task data"""
        print("\n" + "="*60)
        print("RUBRIC QUALITY ANALYSIS")
        print("="*60)
        
        if 'master' not in self.data or 'matrix' not in self.data:
            return
            
        master = self.data['master']
        matrix = self.data['matrix']
        
        # First show aggregated rubric scores from matrix
        dimensions = ['accuracy_mean', 'coverage_mean', 'depth_mean', 'style_mean', 'utility_mean', 'specificity_mean']
        
        print("\nRubric Performance by Model (Aggregated):")
        model_quality = {}
        
        for _, row in matrix.iterrows():
            model = row['model']
            avg_rubric = np.mean([row[dim] for dim in dimensions if pd.notna(row[dim])])
            model_quality[model] = avg_rubric
            
            print(f"\n{model}:")
            print(f"  Overall Rubric: {avg_rubric:.2f}/5.0 ({avg_rubric/5*100:.1f}% quality)")
            
            for dim in dimensions:
                if pd.notna(row[dim]):
                    clean_dim = dim.replace('_mean', '').title()
                    print(f"  {clean_dim:<12}: {row[dim]:.2f}/5.0")
        
        # Now analyze raw rubric score distribution
        print(f"\nRubric Score Distribution Analysis:")
        for model in master['model'].unique():
            model_data = master[master['model'] == model]
            rubric_scores = model_data['rubric_score'].dropna()
            
            if len(rubric_scores) > 0:
                mean_score = rubric_scores.mean()
                std_score = rubric_scores.std()
                min_score = rubric_scores.min()
                max_score = rubric_scores.max()
                
                # Count perfect scores (5.0)
                perfect_scores = len(rubric_scores[rubric_scores == 5.0])
                poor_scores = len(rubric_scores[rubric_scores <= 2.0])
                
                print(f"\n{model}:")
                print(f"  Mean: {mean_score:.2f} ± {std_score:.2f}")
                print(f"  Range: {min_score:.1f} - {max_score:.1f}")
                print(f"  Perfect scores (5.0): {perfect_scores}/{len(rubric_scores)} ({perfect_scores/len(rubric_scores)*100:.1f}%)")
                print(f"  Poor scores (≤2.0): {poor_scores}/{len(rubric_scores)} ({poor_scores/len(rubric_scores)*100:.1f}%)")
        
        # Generate quality insights
        for model, avg_rubric in model_quality.items():
            if avg_rubric >= 4.8:
                self.insights.append(f"⭐ {model} achieves EXPERT-LEVEL quality ({avg_rubric:.2f}/5.0 - {avg_rubric/5*100:.1f}%)")
            elif avg_rubric >= 4.4:
                self.insights.append(f"✨ {model} reaches PROFESSIONAL quality ({avg_rubric:.2f}/5.0 - {avg_rubric/5*100:.1f}%)")
            elif avg_rubric < 3.5:
                self.insights.append(f"⚠️  {model} shows concerning quality gaps ({avg_rubric:.2f}/5.0 - {avg_rubric/5*100:.1f}%)")
        
        # Identify the biggest quality gaps
        sorted_models = sorted(model_quality.items(), key=lambda x: x[1])
        if len(sorted_models) >= 2:
            worst_model, worst_score = sorted_models[0]
            best_model, best_score = sorted_models[-1]
            quality_gap = best_score - worst_score
            
            if quality_gap > 1.0:
                self.insights.append(f"📊 MASSIVE quality gap: {best_model} ({best_score:.2f}) vs {worst_model} ({worst_score:.2f}) = {quality_gap:.2f} point difference")
    
    def analyze_prompt_robustness(self):
        """Analyze model sensitivity to prompt variants and temperature"""
        print("\n" + "="*60)
        print("PROMPT ROBUSTNESS ANALYSIS")
        print("="*60)
        
        if 'master' not in self.data:
            return
            
        master = self.data['master']
        
        # Group by model and variant for rubric scores
        print("\nPrompt Variant Sensitivity (Rubric Scores):")
        model_robustness = {}
        
        for model in master['model'].unique():
            model_data = master[master['model'] == model]
            variant_analysis = model_data.groupby('variant')['rubric_score'].agg(['mean', 'std', 'count']).reset_index()
            
            if len(variant_analysis) > 1:
                score_range = variant_analysis['mean'].max() - variant_analysis['mean'].min()
                model_robustness[model] = score_range
                
                print(f"\n{model}:")
                for _, row in variant_analysis.iterrows():
                    variant = row['variant']
                    mean_score = row['mean']
                    std_score = row['std']
                    count = row['count']
                    print(f"  Variant {variant}: {mean_score:.2f} ± {std_score:.2f} ({count} tasks)")
                print(f"  Range: {score_range:.2f} points")
        
        # Temperature sensitivity analysis
        print(f"\nTemperature Sensitivity Analysis:")
        for model in master['model'].unique():
            model_data = master[master['model'] == model]
            temp_analysis = model_data.groupby('temp')['rubric_score'].agg(['mean', 'std', 'count']).reset_index()
            
            if len(temp_analysis) > 1:
                temp_range = temp_analysis['mean'].max() - temp_analysis['mean'].min()
                print(f"\n{model}:")
                for _, row in temp_analysis.iterrows():
                    temp = row['temp']
                    mean_score = row['mean']
                    print(f"  Temp {temp}: {mean_score:.2f}")
                print(f"  Temperature range: {temp_range:.2f} points")
                
                if temp_range > 0.5:
                    self.insights.append(f"🌡️  {model} shows temperature sensitivity ({temp_range:.2f} point range)")
        
        # Robustness insights
        if model_robustness:
            print(f"\nPrompt Variant Sensitivity Ranking:")
            for model, range_val in sorted(model_robustness.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model:<20}: {range_val:.2f} point range")
                
                if range_val > 1.0:
                    self.insights.append(f"📊 {model} shows HIGH prompt sensitivity ({range_val:.2f} point range)")
                elif range_val < 0.3:
                    self.insights.append(f"🎯 {model} demonstrates ROBUST performance across prompts ({range_val:.2f} range)")
                    
        # Identify most/least robust models
        if model_robustness:
            most_sensitive = max(model_robustness.items(), key=lambda x: x[1])
            most_robust = min(model_robustness.items(), key=lambda x: x[1])
            
            self.insights.append(f"🏆 Most robust model: {most_robust[0]} ({most_robust[1]:.2f} range)")
            self.insights.append(f"⚠️  Most sensitive model: {most_sensitive[0]} ({most_sensitive[1]:.2f} range)")
    
    def generate_executive_summary(self):
        """Generate executive summary of key findings"""
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY - KEY INSIGHTS")
        print("="*60)
        
        print("\n🔍 SYSTEMATIC ANALYSIS FINDINGS:")
        
        if not self.insights:
            print("  No significant insights detected.")
            return
            
        # Categorize insights
        reliability_insights = [i for i in self.insights if "wrapper" in i.lower() or "structural" in i.lower()]
        breakthrough_insights = [i for i in self.insights if "breakthrough" in i.lower() or "threshold" in i.lower()]
        quality_insights = [i for i in self.insights if "quality" in i.lower() or "expert" in i.lower()]
        robustness_insights = [i for i in self.insights if "prompt" in i.lower() or "robust" in i.lower()]
        
        categories = [
            ("🏗️  STRUCTURAL RELIABILITY", reliability_insights),
            ("🚀 CAPABILITY BREAKTHROUGHS", breakthrough_insights),
            ("⭐ QUALITY ASSESSMENT", quality_insights),
            ("🎯 PROMPT ROBUSTNESS", robustness_insights)
        ]
        
        for category_name, category_insights in categories:
            if category_insights:
                print(f"\n{category_name}:")
                for insight in category_insights:
                    print(f"  • {insight}")
    
    def save_analysis_report(self):
        """Save detailed analysis to JSON file"""
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "summary_stats": {},
            "insights": self.insights,
            "model_rankings": []
        }
        
        if 'summary' in self.data:
            summary = self.data['summary'].sort_values('mean_score', ascending=False)
            for _, row in summary.iterrows():
                report["model_rankings"].append({
                    "model": row['model'],
                    "score": float(row['mean_score']),
                    "ci_low": float(row['ci_low']),
                    "ci_high": float(row['ci_high'])
                })
        
        output_file = self.base_path / "taci_analysis_report.json"
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {output_file}")
    
    def run_full_analysis(self):
        """Run complete TACI analysis pipeline"""
        print("🔬 TACI SYSTEMATIC INSIGHTS ANALYZER")
        print("=====================================")
        
        if not self.load_data():
            print("❌ Analysis failed - could not load data files")
            return
        
        # Run all analysis modules
        self.analyze_structural_reliability()
        self.analyze_capability_tiers()
        self.analyze_rubric_quality()
        self.analyze_prompt_robustness()
        
        # Generate summary and save report
        self.generate_executive_summary()
        self.save_analysis_report()
        
        print(f"\n✅ Analysis complete! Found {len(self.insights)} key insights.")
        print("🎯 TACI successfully identified breakthrough AI capabilities!")

def main():
    """Main execution function"""
    analyzer = TACIAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()