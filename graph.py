import matplotlib.pyplot as plt
import numpy as np

# ─── data ──────────────────────────────────────────────────────────────────────
steps = [0, 250, 400, 1000, 1500, 5000, 6000, 10000, 14000, 18000]
means = [-54.12, -43.69, -42.03, -43.46, -44.21, -45.48,
         -46.47, -39.42, -41.80, -41.75]
stds  = [9.72, 15.81, 19.20, 17.67, 15.05, 15.61,
         25.33, 47.71, 39.97, 41.15]
mins  = [-120.85, -107.04, -109.39, -107.24, -104.95, -103.60,
         -103.52, -112.02, -110.02, -104.78]
maxs  = [-27.07, 28.06, 55.62, 48.55, 14.38, 30.14,
         182.95, 336.32, 261.75, 238.72]

# ─── plotting ──────────────────────────────────────────────────────────────────
x = np.arange(len(steps))

plt.figure(figsize=(12, 6))

# mean with ±1 std vertical error bars
plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5, label='Mean ± 1 std')

# min / max triangles
plt.scatter(x, mins, marker='v', label='Min', alpha=0.9)
plt.scatter(x, maxs, marker='^', label='Max', alpha=0.9)

# light vertical guide-lines at each tick (visual cue like your example)
for xi in x:
    plt.axvline(x=xi, color='gray', linestyle='--', linewidth=0.4, alpha=0.4)

plt.xticks(x, [str(s) for s in steps])
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.title('Evaluation statistics – DAgger')
plt.legend()
plt.grid(axis='y', linestyle=':', linewidth=0.4)
plt.tight_layout()
plt.savefig('dagger_stats.png', dpi=300)
