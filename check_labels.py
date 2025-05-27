"""Модуль для проверки меток в датасете."""

import pickle


with open("data/train_data/ninaprodb1train.pkl", "rb") as f:
    df = pickle.load(f)

print("unique restimulus:", sorted(df["restimulus"].unique()))
print("max restimulus:", df["restimulus"].max())

for lbl in sorted(df["restimulus"].unique()):
    count = (df["restimulus"] == lbl).sum()
    print(f"метка {lbl}: {count} строк")

print("\nПримеры для метки 0 (первые 5 строк):")
print(df[df["restimulus"] == "0"].head())

print("\nПримеры для метки 255 (первые 5 строк):")
print(df[df["restimulus"] == "255"].head())
