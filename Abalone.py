import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Load the data
abalone_df = pd.read_csv("abalone.csv")

# Perform PCA
pca = PCA(n_components=7, center=True, whiten=True)
pca_data = pca.fit_transform(abalone_df.iloc[:,1:8])

# Scree plot
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1, 8), pca.explained_variance_ratio_, '-o', linewidth=2)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Proportion of Variance Explained")
ax.set_ylim(0,1)
ax.set_xticks(range(1, 8))
plt.show()

# Biplot
fig, ax = plt.subplots(figsize=(8,8))
sns.set_style("white")
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], alpha=0.7, color='lightgray', ax=ax)
for i in range(7):
    ax.arrow(0, 0, pca.components_[0,i], pca.components_[1,i], color='black', head_width=0.05)
    ax.text(pca.components_[0,i]*1.15, pca.components_[1,i]*1.15, abalone_df.columns[i+1], color='black', ha='center', va='center', fontsize=12)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
plt.show()

# Cluster analysis
kmeans = KMeans(n_clusters=3, n_init=25, random_state=123)
kmeans.fit(abalone_df.iloc[:,1:8])
abalone_cluster = kmeans.predict(abalone_df.iloc[:,1:8])

fig, ax = plt.subplots(figsize=(8,8))
sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=abalone_cluster, palette='tab10', alpha=0.7, ax=ax)
plt.show()

# Linear model
lm_shucked_weight = LinearRegression().fit(abalone_df.iloc[:,[3,5,7]], abalone_df.iloc[:,6])
print(lm_shucked_weight.summary())

# Model diagnostics
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8,8))
sns.regplot(x=lm_shucked_weight.fittedvalues, y=lm_shucked_weight.resid, ax=axes[0,0], scatter_kws={'alpha':0.5})
sns.histplot(lm_shucked_weight.resid, ax=axes[0,1], kde=True)
sns.scatterplot(x=abalone_df.iloc[:,6], y=lm_shucked_weight.resid, ax=axes[1,0], alpha=0.5)
sns.scatterplot(x=abalone_df.iloc[:,6], y=lm_shucked_weight.fittedvalues, ax=axes[1,1], alpha=0.5)
plt.show()
