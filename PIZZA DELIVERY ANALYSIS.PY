import pandas as pd
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading Excel file...")
df = pd.read_excel("PIZZA DELIVERY ANALYSIS.xlsx", sheet_name=None)
print("Sheets loaded:", df.keys())

# Z-Test (Sheet1)
print("\n--- Z-Test (Known SD) ---")
z_data = df['Sheet1'].iloc[0:15]
local_z = pd.to_numeric(z_data.iloc[:, 1], errors='coerce').dropna().values
branded_z = pd.to_numeric(z_data.iloc[:, 2], errors='coerce').dropna().values
mean_local = np.mean(local_z)
mean_branded = np.mean(branded_z)
n = len(local_z)
sigma_local = 2.5
sigma_branded = 3
z = (mean_local - mean_branded) / np.sqrt(sigma_local**2/n + sigma_branded**2/n)
p_z = norm.cdf(z)
print(f"Z = {z:.4f}, p-value = {p_z:.4f}")

# Bar chart of mean delivery times (Sheet1)
plt.figure(figsize=(6, 4))
means = [mean_local, mean_branded]
labels = ['Local', 'Branded']
colors = ['skyblue', 'salmon']
plt.bar(labels, means, color=colors)
plt.title('Mean Delivery Times (Sheet1)')
plt.ylabel('Minutes')
plt.tight_layout()
plt.show()


# t-Test (Equal Variances, Sheet2)
print("\n--- t-Test (Equal Variances) ---")
t_data = df['Sheet2'].iloc[0:15]
local_t = pd.to_numeric(t_data.iloc[:, 1], errors='coerce').dropna().values
branded_t = pd.to_numeric(t_data.iloc[:, 2], errors='coerce').dropna().values
mean_local = np.mean(local_t)
mean_branded = np.mean(branded_t)
var_local = np.var(local_t, ddof=1)
var_branded = np.var(branded_t, ddof=1)
sp2 = ((n - 1)*var_local + (n - 1)*var_branded) / (2*n - 2)
t_stat = (mean_local - mean_branded) / np.sqrt(sp2 * (2/n))
df_t = 2*n - 2
p_t = 2 * t.cdf(t_stat, df_t)
print(f"t = {t_stat:.4f}, df = {df_t}, p-value = {p_t:.4f}")

# Paired t-Test (Sheet3)
print("\n--- Paired t-Test ---")
paired_data = df['Sheet3'].iloc[0:15]
local_p = pd.to_numeric(paired_data.iloc[:, 1], errors='coerce').dropna().values
branded_p = pd.to_numeric(paired_data.iloc[:, 2], errors='coerce').dropna().values
diff = local_p - branded_p
mean_diff = np.mean(diff)
sd_diff = np.std(diff, ddof=1)
t_paired = mean_diff / (sd_diff / np.sqrt(n))
df_p = n - 1
p_paired = 2 * t.cdf(t_paired, df_p)
print(f"t = {t_paired:.4f}, df = {df_p}, p-value = {p_paired:.4f}")

# Box plot of delivery times (Sheet2)
plt.figure(figsize=(6, 4))
sns.boxplot(data=[local_t, branded_t], palette='pastel')
plt.xticks([0, 1], ['Local', 'Branded'])
plt.title('Delivery Time Distribution (Sheet2)')
plt.ylabel('Minutes')
plt.tight_layout()
plt.show()

# Box plot of paired differences (Sheet3)
plt.figure(figsize=(6, 4))
sns.boxplot(data=diff, color='lightgreen')
plt.title('Paired Differences (Local - Branded)')
plt.ylabel('Minutes')
plt.tight_layout()
plt.show()
