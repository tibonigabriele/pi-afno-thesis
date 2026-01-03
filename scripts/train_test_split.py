# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2025 Gabriele Tiboni
# MSc Thesis — Computer Engineering, University of Padua (UniPD)
#
# Rights and licensing:
# - This source file is released for academic/research dissemination as part of a
#   Master’s thesis project.
# - If this repository provides a LICENSE file, this file is distributed under
#   those terms. Otherwise, all rights are reserved by the author.
#
# Notes on originality:
# - The logic implemented here (random shuffle with a fixed seed and a 90/10 split)
#   is a standard data-preparation pattern widely used across the community and
#   does not appear to be derived from any specific publicly available script.
# -----------------------------------------------------------------------------

import pandas as pd

src = "data/spx_spy_options.csv"
train_out = "data/spx_spy_options_train.csv"
test_out  = "data/spx_spy_options_test.csv"

df = pd.read_csv(src)

# Split: 90% train, 10% test (fixed seed for reproducibility).
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

n_train = int(0.90 * len(df))
df_train = df.iloc[:n_train].copy()
df_test  = df.iloc[n_train:].copy()

df_train.to_csv(train_out, index=False)
df_test.to_csv(test_out, index=False)

print(f"Saved: {train_out} (rows={len(df_train)})")
print(f"Saved: {test_out} (rows={len(df_test)})")