# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-glm-tools Demo
# MAGIC
# MAGIC This notebook demonstrates both subpackages on synthetic UK motor insurance data.
# MAGIC
# MAGIC - `insurance_glm_tools.cluster`: Band vehicle age using R2VF factor clustering
# MAGIC - `insurance_glm_tools.nested`: Fit a nested GLM with entity embeddings

# COMMAND ----------

# MAGIC %pip install insurance-glm-tools

# COMMAND ----------

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 5_000

# Synthetic UK motor data
vehicle_age = rng.integers(0, 16, n)
ncd_years = rng.integers(0, 10, n)
age_band = rng.choice(["17-25", "26-35", "36-50", "51-65", "66+"], n)
vehicle_make = rng.choice(
    ["Ford", "Vauxhall", "Toyota", "BMW", "Audi", "Honda", "Nissan", "VW",
     "Renault", "Peugeot", "Volkswagen", "Mercedes", "Skoda", "Seat", "Kia"],
    n
)
exposure = rng.uniform(0.1, 1.0, n)

# True structure: three vehicle age bands
va_true_rate = np.where(vehicle_age <= 3, -2.0, np.where(vehicle_age <= 8, -1.7, -1.3))
ncd_true_rate = np.where(ncd_years <= 2, 0.0, np.where(ncd_years <= 5, -0.3, -0.6))
mu = exposure * np.exp(va_true_rate + ncd_true_rate)
y = rng.poisson(mu).astype(float)

X = pd.DataFrame({
    "vehicle_age": vehicle_age,
    "ncd_years": ncd_years,
    "age_band": age_band,
    "vehicle_make": vehicle_make,
})

print(f"Dataset: {n:,} policies, {y.sum():.0f} claims, freq={y.sum()/exposure.sum():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: R2VF Factor Clustering

# COMMAND ----------

from insurance_glm_tools.cluster import FactorClusterer

fc = FactorClusterer(family="poisson", lambda_="bic", min_exposure=200, n_lambda=30)
fc.fit(X, y, exposure=exposure, ordinal_factors=["vehicle_age", "ncd_years"])

print("Vehicle age groupings:")
print(fc.level_map("vehicle_age").to_df().to_string())
print()
print("NCD years groupings:")
print(fc.level_map("ncd_years").to_df().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### BIC path

# COMMAND ----------

path_df = fc.diagnostic_path.to_df()
print(f"Best lambda: {fc.best_lambda:.6f}")
print(f"Vehicle age groups: {fc.level_map('vehicle_age').n_groups}")
print(f"NCD groups: {fc.level_map('ncd_years').n_groups}")
display(path_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Refit on merged encoding

# COMMAND ----------

X_merged = fc.transform(X)
result = fc.refit_glm(X_merged, y, exposure=exposure)
print(result.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 2: Nested GLM with entity embeddings (no spatial)

# COMMAND ----------

from insurance_glm_tools.nested import NestedGLMPipeline

pipeline = NestedGLMPipeline(
    base_formula="age_band",
    embedding_epochs=10,
    embedding_hidden_sizes=(32,),
    random_state=42,
)
pipeline.fit(
    X, y, exposure,
    high_card_cols=["vehicle_make"],
    base_formula_cols=["age_band"],
)

print("Outer GLM relativities:")
display(pipeline.relativities())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Embedding vectors for vehicle makes

# COMMAND ----------

frames = pipeline.embedding_trainer_.get_embedding_frame()
make_embeddings = frames["vehicle_make"]
print(f"Embedding shape: {make_embeddings.shape}")
display(make_embeddings.head(10))
