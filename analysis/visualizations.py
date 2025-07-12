import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
#  TACI progression – Paralegals & Legal Assistants (six models)
#  Enhanced with breakthrough insights and capability thresholds
# ------------------------------------------------------------------

models = [
    {   # 1 ─ GPT-3.5-turbo
        "label": "GPT-3.5-turbo",
        "score": 61.13,
        "ci": (59.00, 63.14),
        "ctx": "16k",
        "price": "$0.0005/k",
        "why": "UNRELIABLE FOR PRODUCTION\n• 62% legal accuracy (frequent errors)\n• Highly prompt-sensitive\n• Generic responses lacking precision",
        "annotate": True,
        "tier": "unreliable",
        "structural_fails": "12%",
        "rubric_avg": 3.1
    },
    {   # 2 ─ GPT-4
        "label": "GPT-4",
        "score": 63.58,
        "ci": (61.43, 65.77),
        "ctx": "8k",
        "price": "$0.03/k", 
        "why": "STRUCTURAL FAILURES\n• 11% wrapper/schema failures\n• Shallow legal analysis (3.5/5 depth)\n• Cannot be trusted in automated workflows",
        "annotate": True,
        "tier": "unreliable",
        "structural_fails": "11%",
        "rubric_avg": 3.4
    },
    {   # 3 ─ GPT-4 Turbo
        "label": "GPT-4 Turbo",
        "score": 64.62,
        "ci": (61.49, 67.43),
        "ctx": "128k",
        "price": "$0.01/k",
        "why": "MINIMAL IMPROVEMENT\n• Fixes most structural issues\n• Still lacks legal depth\n• 68% quality threshold not met",
        "annotate": True,
        "tier": "marginal",
        "structural_fails": "1%",
        "rubric_avg": 3.5
    },
    {   # 4 ─ GPT-4o
        "label": "GPT-4o",
        "score": 72.63,
        "ci": (70.34, 74.76),
        "ctx": "128k",
        "price": "$0.0025/k",
        "why": "APPROACHING VIABILITY\n• Better legal reasoning (4.1/5 depth)\n• Consistent structural compliance\n• Still below professional threshold",
        "annotate": True,
        "tier": "approaching",
        "structural_fails": "0%",
        "rubric_avg": 4.1
    },
    {   # 5 ─ GPT-4.1
        "label": "GPT-4.1",
        "score": 78.01,
        "ci": (75.93, 79.87),
        "ctx": "1048k",
        "price": "$0.002/k",
        "why": "PROFESSIONAL THRESHOLD\n• 89% legal quality (4.5/5 rubric)\n• Perfect structural reliability\n• First viable paralegal assistant",
        "annotate": True,
        "tier": "professional",
        "structural_fails": "0%",
        "rubric_avg": 4.5
    },
    {   # 6 ─ O3
        "label": "O3-2025",
        "score": 83.54,
        "ci": (82.34, 84.43),
        "ctx": "200k",
        "price": "$0.025/k",
        "why": "EXPERT-LEVEL CAPABILITY\n• 98% legal quality (4.9/5 rubric)\n• Near-perfect legal analysis depth\n• Exceeds many human paralegals",
        "annotate": True,
        "tier": "expert",
        "structural_fails": "6%*",
        "rubric_avg": 4.9
    },
]

# -------------------- data prep --------------------
dx = 1.8
x = np.arange(len(models)) * dx
means     = [m["score"] for m in models]
ci_lows   = [m["ci"][0] for m in models]
ci_highs  = [m["ci"][1] for m in models]
yerr = [[m - lo for m, lo in zip(means, ci_lows)],
        [hi - m for m, hi in zip(means, ci_highs)]]
labels = [m["label"] for m in models]

# Define vibrant capability tiers and colors
tier_colors = {
    "unreliable": "#E74C3C",    # Bold red - unreliable
    "marginal": "#F39C12",      # Bold orange - marginal 
    "approaching": "#F1C40F",   # Bold yellow - approaching
    "professional": "#2ECC71",  # Bold green - professional threshold
    "expert": "#3498DB"         # Bold blue - expert level
}

# -------------------- plot -------------------------
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fig, ax = plt.subplots(facecolor='white')
ax.set_facecolor('white')

# Add vibrant capability threshold zones with gradients
ax.axhspan(55, 70, alpha=0.15, color='#E74C3C', label='Unreliable Zone')
ax.axhspan(70, 75, alpha=0.18, color='#F39C12', label='Approaching Viability')
ax.axhspan(75, 80, alpha=0.2, color='#2ECC71', label='Professional Threshold')
ax.axhspan(80, 90, alpha=0.22, color='#3498DB', label='Expert Level')

# Professional threshold line with vibrant styling
ax.axhline(y=75, color='#2ECC71', linestyle='--', linewidth=4, alpha=0.8, 
           label='Professional Viability Threshold')

# Main progression line with dynamic styling
ax.plot(x, means, color="#34495E", linewidth=5, zorder=3, alpha=0.9, 
        solid_capstyle='round', solid_joinstyle='round')
ax.errorbar(x, means, yerr=yerr, fmt='none',
            ecolor="#95A5A6", elinewidth=3,
            capsize=10, capthick=3, zorder=2, alpha=0.8)

# Vibrant color-coded scatter points with enhanced shadows
shadow_offset = 0.08
for xi, yi, m in zip(x, means, models):
    color = tier_colors[m["tier"]]
    size = 400 if m["tier"] in ["professional", "expert"] else 320
    edge_width = 4 if m["tier"] in ["professional", "expert"] else 3
    
    # Main point
    ax.scatter(xi, yi, s=size, color=color,
               edgecolor="white", lw=edge_width, zorder=4)

# Enhanced annotation offsets and styling
offset = {
    "GPT-3.5-turbo": 8,
    "GPT-4": -5,
    "GPT-4 Turbo": 8,
    "GPT-4o": -10,
    "GPT-4.1": 7,
    "O3-2025": -10,
}

for xi, yi, m in zip(x, means, models):
    if not m["annotate"]:
        continue
    dy = offset[m["label"]]
    va = 'bottom' if dy > 0 else 'top'
    
    # Vibrant annotation styling based on tier
    box_color = tier_colors[m["tier"]]
    text_color = "white"
    
    ax.annotate(m["why"], xy=(xi, yi),
                xytext=(xi, yi + dy),
                ha='center', va=va,
                fontsize=11.5, fontweight='700' if m["tier"] in ["professional", "expert"] else '600',
                color=text_color, 
                arrowprops=dict(arrowstyle='->', lw=3.5, color=box_color, alpha=0.9),
                bbox=dict(boxstyle="round,pad=0.6,rounding_size=0.2", fc=box_color, 
                          edgecolor="white", linewidth=2, alpha=0.95))

# Enhanced context & price captions with performance metrics
for xi, m in zip(x, models):
    caption = f"{m['ctx']} • {m['price']}\nRubric: {m['rubric_avg']}/5.0 • Fails: {m['structural_fails']}"
    ax.annotate(caption, (xi, 51.5),
                ha='center', va='bottom',
                fontsize=9.5, color="#34495E", fontweight='500',
                bbox=dict(boxstyle="round,pad=0.4", fc="#ECF0F1", 
                          edgecolor="#BDC3C7", linewidth=1.5, alpha=0.95))

# Bold typography and enhanced labels
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, rotation=20, ha='right', color="#2C3E50", fontweight='600')
ax.set_ylabel("TACI Paralegal Automation Score (0–100, 95% CI)", 
              fontsize=15, fontweight='700', color="#2C3E50", labelpad=18)
ax.set_title("AI Legal Assistant Evolution: From Unreliable to Expert-Level\nTACI Paralegal Benchmark Results", 
             fontsize=20, weight='800', pad=30, color="#2C3E50")

# Enhanced Y-axis with meaningful labels
ax.set_ylim(50, 88)
ax.set_xlim(x[0]-1.2, x[-1]+1.2)

# Bold threshold labels on the right
ax.text(x[-1]+0.9, 77.5, "Professional\nThreshold", rotation=90, ha='center', va='center',
        fontsize=12, fontweight='700', color='#2ECC71', alpha=0.9)
ax.text(x[-1]+0.9, 82, "Expert\nLevel", rotation=90, ha='center', va='center',
        fontsize=12, fontweight='700', color='#3498DB', alpha=0.9)

# Enhanced grid with better contrast
ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=1.2, color='#D5DBDB')
ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=1, color='#D5DBDB')

# Bold breakthrough callouts
ax.annotate('BREAKTHROUGH\n+5.4 points', xy=(x[4], means[4]), xytext=(x[4]-0.8, means[4]+3),
            fontsize=12, fontweight='800', color='white',
            arrowprops=dict(arrowstyle='->', lw=4, color='#2ECC71', alpha=0.9),
            bbox=dict(boxstyle="round,pad=0.5", fc='#2ECC71', alpha=0.95, 
                      edgecolor="white", linewidth=2))

ax.annotate('EXPERT LEAP\n+5.5 points', xy=(x[5], means[5]), xytext=(x[5]-0.8, means[5]-4),
            fontsize=12, fontweight='800', color='white',
            arrowprops=dict(arrowstyle='->', lw=4, color='#3498DB', alpha=0.9),
            bbox=dict(boxstyle="round,pad=0.5", fc='#3498DB', alpha=0.95,
                      edgecolor="white", linewidth=2))

# Sleek footnote
fig.text(0.08, 0.03, "* O3 wrapper failures due to verbose reasoning chains, not capability deficits", 
         fontsize=10, style='italic', color='#7F8C8D', fontweight='400')

# Final styling touches with more contrast
ax.tick_params(axis='both', which='major', labelsize=11.5, colors='#34495E', length=8, width=1.5)
ax.spines['bottom'].set_color('#85929E')
ax.spines['left'].set_color('#85929E')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)

plt.tight_layout(pad=2)
# plt.savefig("taci_paralegal_evolution_sleek.png", dpi=300, bbox_inches='tight', 
#             facecolor='#fafafa', edgecolor='none')
plt.show()
