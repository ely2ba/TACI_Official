import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Load your composite‐output CSV (update the path as needed)
path = "graders/phase_04_composite_paralegal/phase_04_composite_per_output_paralegal.csv"
df = pd.read_csv(path)

# 2. Filter to the three models you care about
models = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]
df = df[df["model"].isin(models)].copy()

# 3. Map each model into your three epochs & launch dates
model_to_epoch = {
    "gpt-4":         (1, "2023-03-14"),
    "gpt-3.5-turbo": (2, "2024-01-25"),
    "gpt-4o":        (3, "2024-06-08"),
}
df["epoch"] = df["model"].map(lambda m: model_to_epoch[m][0])
df["launch_date"] = pd.to_datetime(df["model"].map(lambda m: model_to_epoch[m][1]))

# 4. Prepare for fixed‐effects regression: within‐task demeaning
df["uid"] = df["uid"].astype("category")
df["epoch"] = df["epoch"].astype("category")

# Compute demeaned composite score
df["demeaned_score"] = df["composite_score"] - df.groupby("uid")["composite_score"].transform("mean")

# Create epoch dummies (drop epoch 1 as base)
epoch_dummies = pd.get_dummies(df["epoch"], prefix="E", drop_first=True)

# 5. Run OLS on the demeaned outcome with epoch dummies
X = sm.add_constant(epoch_dummies)
ols_res = sm.OLS(df["demeaned_score"], X).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["uid"]}
)

print(ols_res.summary())

# 6. Event‐study plot of the epoch effects
coefs = ols_res.params.drop("const")
ses   = ols_res.bse.drop("const")
# Extract numeric epoch from dummy names, e.g. "E_2" → 2
epochs = [int(name.split("_")[-1]) for name in coefs.index]

plt.figure(figsize=(6, 4))
plt.errorbar(
    epochs,
    coefs,
    yerr=ses,
    fmt="o",
    capsize=5
)
plt.xticks(epochs, ["Epoch 2 vs 1", "Epoch 3 vs 1"])
plt.xlabel("Epoch Comparison")
plt.ylabel("Δ Demeaned Composite Score")
plt.title("TACI Event‐Study: Paralegal Tasks")
plt.grid(True)
plt.tight_layout()
plt.show()
