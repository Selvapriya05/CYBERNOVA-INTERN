import pandas as pd

# ============================================================
# STEP 1 - LOAD DATASET
# ============================================================

file_path = "../dataset/Streaming_Content.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("MOVIE STREAMING PLATFORM ANALYSIS")
print("=" * 70)

print("\nDATASET LOADED SUCCESSFULLY")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

# ============================================================
# STEP 6 - DATA UNDERSTANDING & CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA UNDERSTANDING & CLEANING")
print("=" * 70)

# ------------------------------------------------------------
# 1. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMISSING VALUES:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 2. CHECK DUPLICATES
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()

print("\nDUPLICATE ROWS:")
print(duplicate_count)


# ------------------------------------------------------------
# 3. CHECK CONTENT TYPE
# ------------------------------------------------------------

print("\nCONTENT TYPE:")
print(df["type"].value_counts())


# ------------------------------------------------------------
# 4. CHECK RATINGS
# ------------------------------------------------------------

print("\nCONTENT RATINGS:")
print(df["rating"].value_counts(dropna=False))


# ------------------------------------------------------------
# 5. CHECK RELEASE YEAR
# ------------------------------------------------------------

print("\nRELEASE YEAR RANGE:")
print("Minimum Year:", df["release_year"].min())
print("Maximum Year:", df["release_year"].max())


# ------------------------------------------------------------
# 6. CHECK COUNTRIES
# ------------------------------------------------------------

print("\nTOP COUNTRIES:")
print(
    df["country"]
    .value_counts()
    .head(15)
)


# ------------------------------------------------------------
# 7. CHECK CATEGORIES
# ------------------------------------------------------------

print("\nTOP CATEGORIES:")
print(
    df["listed_in"]
    .value_counts()
    .head(15)
)


# ------------------------------------------------------------
# 8. CHECK DATE ADDED
# ------------------------------------------------------------

print("\nDATE ADDED SAMPLE:")
print(df["date_added"].head(10))


# ------------------------------------------------------------
# 9. CHECK ZERO / INVALID RELEASE YEARS
# ------------------------------------------------------------

print("\nINVALID RELEASE YEARS:")

invalid_years = df[
    (df["release_year"] <= 0)
]

print("Count:", len(invalid_years))


# ------------------------------------------------------------
# 10. BASIC CLEANING
# ------------------------------------------------------------

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Fill missing text fields
text_columns = [
    "director",
    "cast",
    "country",
    "rating"
]

for column in text_columns:
    df[column] = df[column].fillna("Unknown")


# Convert date_added to datetime
df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)


# Remove rows with invalid release year
df = df[df["release_year"] > 0].copy()


print("\n" + "=" * 70)
print("CLEANING COMPLETED")
print("=" * 70)

print("\nCleaned Shape:")
print(df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# ============================================================
# STEP 7 - FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# 1. YEAR CONTENT WAS ADDED
# ------------------------------------------------------------

df["added_year"] = df["date_added"].dt.year


# ------------------------------------------------------------
# 2. MONTH CONTENT WAS ADDED
# ------------------------------------------------------------

df["added_month"] = df["date_added"].dt.month_name()


# ------------------------------------------------------------
# 3. CONTENT AGE
# ------------------------------------------------------------

df["content_age"] = 2021 - df["release_year"]


# ------------------------------------------------------------
# 4. DURATION VALUE
# ------------------------------------------------------------

df["duration_value"] = (
    df["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)


# ------------------------------------------------------------
# 5. DURATION UNIT
# ------------------------------------------------------------

df["duration_unit"] = (
    df["duration"]
    .str.extract(r"([A-Za-z]+)")
)


# ------------------------------------------------------------
# 6. PRIMARY COUNTRY
# ------------------------------------------------------------

df["primary_country"] = (
    df["country"]
    .str.split(",")
    .str[0]
    .str.strip()
)


# ------------------------------------------------------------
# 7. PRIMARY CATEGORY
# ------------------------------------------------------------

df["primary_category"] = (
    df["listed_in"]
    .str.split(",")
    .str[0]
    .str.strip()
)


# ------------------------------------------------------------
# 8. CONTENT STATUS
# ------------------------------------------------------------

df["content_status"] = df["type"]


# ------------------------------------------------------------
# 9. RECENT CONTENT FLAG
# ------------------------------------------------------------

df["is_recent_content"] = (
    df["release_year"] >= 2018
).map({
    True: "Recent",
    False: "Older"
})


# ------------------------------------------------------------
# 10. DISPLAY NEW COLUMNS
# ------------------------------------------------------------

print("\nNEW COLUMNS CREATED:")

new_columns = [
    "added_year",
    "added_month",
    "content_age",
    "duration_value",
    "duration_unit",
    "primary_country",
    "primary_category",
    "content_status",
    "is_recent_content"
]

print(df[new_columns].head(10))


# ------------------------------------------------------------
# 11. CHECK NEW DATA
# ------------------------------------------------------------

print("\nCONTENT TYPE:")
print(df["content_status"].value_counts())

print("\nTOP PRIMARY COUNTRIES:")
print(
    df["primary_country"]
    .value_counts()
    .head(10)
)

print("\nTOP PRIMARY CATEGORIES:")
print(
    df["primary_category"]
    .value_counts()
    .head(10)
)

print("\nCONTENT STATUS:")
print(
    df["is_recent_content"]
    .value_counts()
)


# ------------------------------------------------------------
# 12. CREATE OUTPUT FOLDERS
# ------------------------------------------------------------

import os

os.makedirs("../output/charts", exist_ok=True)
os.makedirs("../output/analysis_results", exist_ok=True)
os.makedirs("../output/cleaned_data", exist_ok=True)


# ------------------------------------------------------------
# 13. SAVE CLEANED DATASET
# ------------------------------------------------------------

cleaned_file = "../output/cleaned_data/streaming_content_cleaned.csv"

df.to_csv(
    cleaned_file,
    index=False
)

print("\nCLEANED DATASET SAVED:")
print(cleaned_file)

print("\nFINAL SHAPE:")
print(df.shape)

print("\n" + "=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

# ============================================================
# STEP 8 - BUSINESS ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS ANALYSIS")
print("=" * 70)


# ============================================================
# 1. CONTENT TYPE ANALYSIS
# ============================================================

content_type_analysis = (
    df.groupby("content_status")
    .size()
    .reset_index(name="Total_Content")
)

content_type_analysis["Percentage"] = (
    content_type_analysis["Total_Content"]
    / content_type_analysis["Total_Content"].sum()
    * 100
)

print("\n1. CONTENT TYPE ANALYSIS")
print(content_type_analysis)


# ============================================================
# 2. CONTENT BY COUNTRY
# ============================================================

country_analysis = (
    df.groupby("primary_country")
    .size()
    .reset_index(name="Total_Content")
    .sort_values("Total_Content", ascending=False)
)

print("\n2. TOP 15 COUNTRIES BY CONTENT")
print(country_analysis.head(15))


# ============================================================
# 3. CONTENT ADDED BY YEAR
# ============================================================

year_added_analysis = (
    df.dropna(subset=["added_year"])
    .groupby("added_year")
    .size()
    .reset_index(name="Total_Content")
    .sort_values("added_year")
)

print("\n3. CONTENT ADDED BY YEAR")
print(year_added_analysis)


# ============================================================
# 4. CONTENT BY PRIMARY CATEGORY
# ============================================================

category_analysis = (
    df.groupby("primary_category")
    .size()
    .reset_index(name="Total_Content")
    .sort_values("Total_Content", ascending=False)
)

print("\n4. TOP 15 CATEGORIES")
print(category_analysis.head(15))


# ============================================================
# 5. CONTENT BY RATING
# ============================================================

rating_analysis = (
    df.groupby("rating")
    .size()
    .reset_index(name="Total_Content")
    .sort_values("Total_Content", ascending=False)
)

print("\n5. CONTENT BY RATING")
print(rating_analysis)


# ============================================================
# 6. RELEASE YEAR ANALYSIS
# ============================================================

release_year_analysis = (
    df.groupby("release_year")
    .size()
    .reset_index(name="Total_Content")
    .sort_values("release_year")
)

print("\n6. RELEASE YEAR ANALYSIS")
print(release_year_analysis.tail(20))


# ============================================================
# 7. MOVIE DURATION ANALYSIS
# ============================================================

movie_duration_analysis = (
    df[df["type"] == "Movie"]
    .groupby("duration_value")
    .size()
    .reset_index(name="Total_Movies")
    .sort_values("duration_value")
)

print("\n7. MOVIE DURATION ANALYSIS")
print(movie_duration_analysis.head(20))


# ============================================================
# 8. TV SHOW SEASONS ANALYSIS
# ============================================================

tv_seasons_analysis = (
    df[df["type"] == "TV Show"]
    .groupby("duration_value")
    .size()
    .reset_index(name="Total_TV_Shows")
    .sort_values("duration_value")
)

print("\n8. TV SHOW SEASONS ANALYSIS")
print(tv_seasons_analysis.head(20))


# ============================================================
# 9. MOVIE vs TV SHOW BY COUNTRY
# ============================================================

country_type_analysis = (
    df.groupby(
        ["primary_country", "content_status"]
    )
    .size()
    .reset_index(name="Total_Content")
)

print("\n9. MOVIE vs TV SHOW BY COUNTRY")
print(
    country_type_analysis
    .sort_values("Total_Content", ascending=False)
    .head(20)
)


# ============================================================
# 10. RECENT vs OLDER CONTENT
# ============================================================

recent_analysis = (
    df.groupby("is_recent_content")
    .size()
    .reset_index(name="Total_Content")
)

recent_analysis["Percentage"] = (
    recent_analysis["Total_Content"]
    / recent_analysis["Total_Content"].sum()
    * 100
)

print("\n10. RECENT vs OLDER CONTENT")
print(recent_analysis)


# ============================================================
# 11. CONTENT ADDED BY MONTH
# ============================================================

month_added_analysis = (
    df.dropna(subset=["added_month"])
    .groupby("added_month")
    .size()
    .reset_index(name="Total_Content")
)

print("\n11. CONTENT ADDED BY MONTH")
print(month_added_analysis)


# ============================================================
# 12. KEY BUSINESS METRICS
# ============================================================

total_content = len(df)

total_movies = (
    (df["type"] == "Movie").sum()
)

total_tv_shows = (
    (df["type"] == "TV Show").sum()
)

recent_content = (
    (df["is_recent_content"] == "Recent").sum()
)

average_movie_duration = (
    df.loc[
        df["type"] == "Movie",
        "duration_value"
    ].mean()
)

average_tv_seasons = (
    df.loc[
        df["type"] == "TV Show",
        "duration_value"
    ].mean()
)


print("\n" + "=" * 70)
print("KEY BUSINESS METRICS")
print("=" * 70)

print("\nTotal Content:", total_content)
print("Total Movies:", total_movies)
print("Total TV Shows:", total_tv_shows)
print("Recent Content:", recent_content)
print(
    "Average Movie Duration:",
    round(average_movie_duration, 2),
    "minutes"
)
print(
    "Average TV Show Seasons:",
    round(average_tv_seasons, 2)
)


# ============================================================
# 13. SAVE ANALYSIS RESULTS
# ============================================================

country_analysis.to_csv(
    "../output/analysis_results/01_content_by_country.csv",
    index=False
)

content_type_analysis.to_csv(
    "../output/analysis_results/02_content_type_analysis.csv",
    index=False
)

year_added_analysis.to_csv(
    "../output/analysis_results/03_content_added_by_year.csv",
    index=False
)

category_analysis.to_csv(
    "../output/analysis_results/04_content_by_category.csv",
    index=False
)

rating_analysis.to_csv(
    "../output/analysis_results/05_content_by_rating.csv",
    index=False
)

release_year_analysis.to_csv(
    "../output/analysis_results/06_release_year_analysis.csv",
    index=False
)

movie_duration_analysis.to_csv(
    "../output/analysis_results/07_movie_duration.csv",
    index=False
)

tv_seasons_analysis.to_csv(
    "../output/analysis_results/08_tv_show_seasons.csv",
    index=False
)

country_type_analysis.to_csv(
    "../output/analysis_results/09_country_content_type.csv",
    index=False
)

recent_analysis.to_csv(
    "../output/analysis_results/10_recent_vs_older.csv",
    index=False
)

month_added_analysis.to_csv(
    "../output/analysis_results/11_content_added_by_month.csv",
    index=False
)


print("\nAnalysis files saved successfully.")

print("\n" + "=" * 70)
print("BUSINESS ANALYSIS COMPLETED")
print("=" * 70)

# ============================================================
# STEP 9 - PYTHON VISUALIZATIONS
# ============================================================

import matplotlib.pyplot as plt

chart_path = "../output/charts"
os.makedirs(chart_path, exist_ok=True)


# ============================================================
# CHART 1 - MOVIES VS TV SHOWS
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    content_type_analysis["content_status"],
    content_type_analysis["Total_Content"]
)

plt.title("Movies vs TV Shows")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/01_movies_vs_tv_shows.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 2 - TOP 10 COUNTRIES
# ============================================================

top_countries = (
    country_analysis
    .head(10)
    .sort_values("Total_Content")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_countries["primary_country"],
    top_countries["Total_Content"]
)

plt.title("Top 10 Countries by Content")
plt.xlabel("Number of Titles")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/02_top_10_countries.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 3 - CONTENT ADDED BY YEAR
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    year_added_analysis["added_year"],
    year_added_analysis["Total_Content"],
    marker="o"
)

plt.title("Content Added by Year")
plt.xlabel("Year")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/03_content_added_by_year.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 4 - TOP 10 CATEGORIES
# ============================================================

top_categories = (
    category_analysis
    .head(10)
    .sort_values("Total_Content")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_categories["primary_category"],
    top_categories["Total_Content"]
)

plt.title("Top 10 Content Categories")
plt.xlabel("Number of Titles")
plt.ylabel("Category")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/04_top_10_categories.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 5 - CONTENT BY RATING
# ============================================================

top_ratings = (
    rating_analysis
    .head(10)
    .sort_values("Total_Content")
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_ratings["rating"],
    top_ratings["Total_Content"]
)

plt.title("Top Content Ratings")
plt.xlabel("Number of Titles")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/05_content_by_rating.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 6 - RECENT VS OLDER CONTENT
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    recent_analysis["is_recent_content"],
    recent_analysis["Total_Content"]
)

plt.title("Recent vs Older Content")
plt.xlabel("Content Status")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/06_recent_vs_older.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 7 - RELEASE YEAR TREND
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    release_year_analysis["release_year"],
    release_year_analysis["Total_Content"]
)

plt.title("Content Distribution by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/07_release_year_trend.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 8 - MOVIE DURATION
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df.loc[
        df["type"] == "Movie",
        "duration_value"
    ].dropna(),
    bins=20
)

plt.title("Movie Duration Distribution")
plt.xlabel("Duration (Minutes)")
plt.ylabel("Number of Movies")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/08_movie_duration_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 9 - TV SHOW SEASONS
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df.loc[
        df["type"] == "TV Show",
        "duration_value"
    ].dropna(),
    bins=15
)

plt.title("TV Show Season Distribution")
plt.xlabel("Number of Seasons")
plt.ylabel("Number of TV Shows")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/09_tv_show_seasons.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 10 - CONTENT ADDED BY MONTH
# ============================================================

month_order = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

month_added_analysis["added_month"] = pd.Categorical(
    month_added_analysis["added_month"],
    categories=month_order,
    ordered=True
)

month_added_analysis = (
    month_added_analysis
    .sort_values("added_month")
)

plt.figure(figsize=(12, 5))

plt.bar(
    month_added_analysis["added_month"],
    month_added_analysis["Total_Content"]
)

plt.title("Content Added by Month")
plt.xlabel("Month")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/10_content_added_by_month.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 11 - MOVIE VS TV SHOW BY TOP COUNTRIES
# ============================================================

top_country_names = (
    country_analysis
    .head(10)["primary_country"]
    .tolist()
)

country_type_top = country_type_analysis[
    country_type_analysis["primary_country"].isin(
        top_country_names
    )
]

pivot_country = country_type_top.pivot(
    index="primary_country",
    columns="content_status",
    values="Total_Content"
).fillna(0)

pivot_country = pivot_country.sort_values(
    by=pivot_country.columns.tolist(),
    ascending=False
)

pivot_country.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Movies vs TV Shows Across Top Countries")
plt.xlabel("Country")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    f"{chart_path}/11_movies_vs_tv_shows_by_country.png",
    dpi=300
)

plt.close()


# ============================================================
# CHART 12 - TOP COUNTRIES BY CONTENT TYPE
# ============================================================

movie_country = (
    df[df["type"] == "Movie"]
    .groupby("primary_country")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(
    movie_country.index,
    movie_country.values
)

plt.title("Top 10 Countries by Movie Content")
plt.xlabel("Number of Movies")
plt.ylabel("Country")

plt.tight_layout()

plt.savefig(
    f"{chart_path}/12_top_movie_countries.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PYTHON VISUALIZATIONS COMPLETED")
print("=" * 70)

print("\nCharts saved in:")
print("../output/charts/")

print("\nTotal charts created: 12")