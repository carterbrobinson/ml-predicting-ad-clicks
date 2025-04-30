import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

# === LOAD RAW MERGED DATA ===
df_raw = pd.read_csv("final_train.csv")

# === BASIC EXPLORATION ===
print("Raw shape:", df_raw.shape)
print(df_raw.head())
print(df_raw.tail())

print("\nDtypes:")
print(df_raw.dtypes)

print("\nMissing values:")
print(df_raw.isnull().sum())

print("\nSummary stats:")
print(df_raw.describe())

print("\nInfo:")
df_raw.info()

# === CLASS BALANCE ===
print("\nClass balance:")
print(df_raw['clicked'].value_counts(normalize=True))

# === TIMESTAMP PREVIEW ===
print("\nTimestamp preview:")
print(df_raw[['timestamp', 'publish_time']].head())

# === MISSING DATA PERCENTAGE BAR CHART ===
missing_percent = df_raw.isnull().mean() * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_percent.values, y=missing_percent.index)
plt.title("Missing Data Percentage by Feature")
plt.xlabel("Percentage Missing")
plt.ylabel("Feature")
plt.show()

# === CLICK DISTRIBUTION PLOT ===
sns.countplot(data=df_raw, x='clicked')
plt.title("Click Distribution")
plt.show()

# === HISTOGRAMS FOR NUMERICAL COLUMNS ===
numerical_cols = ['platform', 'campaign_id', 'advertiser_id', 'publisher_id', 'page_view_count']

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df_raw, x=col, hue='clicked', kde=True, bins=30)
    plt.title(f"{col} Distribution by Clicked")
    plt.show()

# === CORRELATION HEATMAP ===
plt.figure(figsize=(10, 8))
sns.heatmap(df_raw[numerical_cols + ['clicked']].corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# === CLICK RATE BY PLATFORM ===
plt.figure(figsize=(6, 4))
sns.barplot(data=df_raw, x='platform', y='clicked')
plt.title("Click Rate by Platform")
plt.ylabel("Average Clicked (Click Rate)")
plt.show()

# === PAGE VIEW COUNT VS CLICKED ===
plt.figure(figsize=(6, 4))
sns.boxplot(data=df_raw, x='clicked', y='page_view_count')
plt.title("Page View Count by Clicked")
plt.show()

# === TOP 10 ADVERTISERS BY CTR ===
top_ads = df_raw.groupby('advertiser_id')['clicked'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 4))
sns.barplot(x=top_ads.index.astype(str), y=top_ads.values)
plt.title("Top 10 Advertisers by CTR")
plt.xlabel("Advertiser ID")
plt.ylabel("Click-Through Rate")
plt.show()

# === TOP 10 CAMPAIGNS BY CTR ===
top_campaigns = df_raw.groupby('campaign_id')['clicked'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 4))
sns.barplot(x=top_campaigns.index.astype(str), y=top_campaigns.values)
plt.title("Top 10 Campaigns by CTR")
plt.xlabel("Campaign ID")
plt.ylabel("Click-Through Rate")
plt.show()

# === CLICK RATE BY TOP 10 PUBLISHERS ===
df_pub = df_raw[df_raw['publisher_id'].notnull()]
top_publishers = df_pub['publisher_id'].value_counts().head(10).index

plt.figure(figsize=(10, 5))
sns.barplot(data=df_pub[df_pub['publisher_id'].isin(top_publishers)], 
            x='publisher_id', y='clicked')
plt.title("CTR by Top 10 Publishers")
plt.ylabel("Click Rate")
plt.show()

# === LOAD AND CHECK PREPROCESSED DATA ===
df_ready = pd.read_csv('final-preprocessed-train.csv', index_col=0)

print("\nPreprocessed shape:", df_ready.shape)
print("Class distribution in preprocessed data:")
print(df_ready['clicked'].value_counts(normalize=True))