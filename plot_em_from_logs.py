import re
import matplotlib.pyplot as plt
import pandas as pd

LOG_FILE = "logs_temp.txt"

# Regex patterns
e_step_re = re.compile(
    r"\[E-step\].*?logL_total=([+-]?\d+\.?\d*),.*?AIC=([+-]?\d+\.?\d*),\s*BIC=([+-]?\d+\.?\d*),.*?PPL_tok=([+-]?\d+\.?\d*),.*?ELBO=([+-]?\d+\.?\d*)"
)
iter_re = re.compile(
    r"\[Iter (\d+)\].*?logL_total=([+-]?\d+\.?\d*),.*?AIC=([+-]?\d+\.?\d*),\s*BIC=([+-]?\d+\.?\d*),.*?PPL_tok=([+-]?\d+\.?\d*),.*?ELBO=([+-]?\d+\.?\d*)"
)

rows = []

with open(LOG_FILE) as f:
    for line in f:
        m = e_step_re.search(line)
        if m:
            rows.append({
                "Iteration": "Initial",
                "logL_total": float(m.group(1)),
                "AIC": float(m.group(2)),
                "BIC": float(m.group(3)),
                "PPL_tok": float(m.group(4)),
                "ELBO": float(m.group(5)),
            })
            continue
        m = iter_re.search(line)
        if m:
            rows.append({
                "Iteration": int(m.group(1)),
                "logL_total": float(m.group(2)),
                "AIC": float(m.group(3)),
                "BIC": float(m.group(4)),
                "PPL_tok": float(m.group(5)),
                "ELBO": float(m.group(6)),
            })

df = pd.DataFrame(rows)

# Print table
print("\n" + "=" * 80)
print("EM Score Evolution")
print("=" * 80)
print(df.to_string(index=False))
print("=" * 80 + "\n")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EM Schema Refinement – Score Evolution (from logs_temp.txt)",
             fontsize=14, fontweight="bold")

x = range(len(df))
labels = df["Iteration"].astype(str)

# 1. Log-Likelihood
ax = axes[0, 0]
ax.plot(x, df["logL_total"], "o-", color="#2ecc71", linewidth=2, markersize=6)
ax.axhline(y=df["logL_total"].iloc[0], color="gray", linestyle="--", alpha=0.5, label="Initial")
ax.set_ylabel("logL_total")
ax.set_title("Log-Likelihood (↑ better)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(); ax.grid(True, alpha=0.3)

# 2. AIC / BIC
ax = axes[0, 1]
ax.plot(x, df["AIC"], "o-", color="#3498db", linewidth=2, markersize=6, label="AIC")
ax.plot(x, df["BIC"], "s-", color="#9b59b6", linewidth=2, markersize=6, label="BIC")
ax.set_ylabel("Score")
ax.set_title("AIC / BIC (↓ better)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.legend(); ax.grid(True, alpha=0.3)

# 3. Perplexity
ax = axes[1, 0]
ax.plot(x, df["PPL_tok"], "o-", color="#e74c3c", linewidth=2, markersize=6)
ax.axhline(y=df["PPL_tok"].iloc[0], color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("PPL_tok")
ax.set_xlabel("Iteration")
ax.set_title("Perplexity (↓ better)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.grid(True, alpha=0.3)

# 4. ELBO
ax = axes[1, 1]
ax.plot(x, df["ELBO"], "o-", color="#f39c12", linewidth=2, markersize=6)
ax.set_ylabel("ELBO")
ax.set_xlabel("Iteration")
ax.set_title("ELBO (↑ better)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "em_scores_from_logs.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"Chart saved to: {out}")