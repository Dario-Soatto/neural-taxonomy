import matplotlib.pyplot as plt
import pandas as pd

# Data from the EM run
data = {
    'Iteration': ['Initial', '1', '2', '3', '4', '5'],
    'logL_total': [-366.96, -305.51, -457.32, -413.03, -428.57, -272.26],
    'AIC': [753.91, 633.02, 938.64, 852.06, 885.13, 574.52],
    'BIC': [796.06, 679.38, 989.21, 906.85, 944.14, 637.74],
    'PPL': [2.083, 1.842, 2.496, 2.284, 2.356, 1.724],
    'ELBO': [-1151.29, -1207.38, -1313.42, -1470.49, -1604.94, -1876.59]
}

df = pd.DataFrame(data)

# Print table
print("\n" + "="*70)
print("EM Score Evolution")
print("="*70)
print(df.to_string(index=False))
print("="*70 + "\n")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('EM Schema Refinement - Score Evolution', fontsize=14, fontweight='bold')

x = range(len(df))
labels = df['Iteration']

# Plot 1: logL_total
ax1 = axes[0, 0]
ax1.plot(x, df['logL_total'], 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax1.set_ylabel('logL_total')
ax1.set_title('Log-Likelihood (↑ better)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=df['logL_total'].iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial')

# Plot 2: AIC and BIC
ax2 = axes[0, 1]
ax2.plot(x, df['AIC'], 'o-', color='#3498db', linewidth=2, markersize=8, label='AIC')
ax2.plot(x, df['BIC'], 's-', color='#9b59b6', linewidth=2, markersize=8, label='BIC')
ax2.set_ylabel('Score')
ax2.set_title('AIC / BIC (↓ better)')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Perplexity
ax3 = axes[1, 0]
ax3.plot(x, df['PPL'], 'o-', color='#e74c3c', linewidth=2, markersize=8)
ax3.set_ylabel('Perplexity')
ax3.set_xlabel('Iteration')
ax3.set_title('Perplexity (↓ better)')
ax3.set_xticks(x)
ax3.set_xticklabels(labels)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=df['PPL'].iloc[0], color='gray', linestyle='--', alpha=0.5)

# Plot 4: ELBO
ax4 = axes[1, 1]
ax4.plot(x, df['ELBO'], 'o-', color='#f39c12', linewidth=2, markersize=8)
ax4.set_ylabel('ELBO')
ax4.set_xlabel('Iteration')
ax4.set_title('ELBO (↑ better, but diverged here)')
ax4.set_xticks(x)
ax4.set_xticklabels(labels)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('em_scores_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

print("Chart saved to: em_scores_evolution.png")