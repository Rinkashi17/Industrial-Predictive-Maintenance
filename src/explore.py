
from pathlib import Path

import pandas as pd


# Caminho raiz do projeto
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"


def main():
    # Carrega o dataset
    df = pd.read_csv(DATA_PATH)

    # Visão geral
    print("\n=== DIMENSÕES DO DATASET ===")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    print("\n=== PRIMEIRAS 5 LINHAS ===")
    print(df.head())

    print("\n=== NOMES DAS COLUNAS ===")
    print(df.columns.tolist())

    print("\n=== TIPOS DE DADOS ===")
    print(df.dtypes)

    print("\n=== VALORES AUSENTES ===")
    print(df.isnull().sum())

    print("\n=== DISTRIBUIÇÃO DE FALHAS ===")
    if "Machine failure" in df.columns:
        print(df["Machine failure"].value_counts())
        print("\nPercentual por classe:")
        print(df["Machine failure"].value_counts(normalize=True) * 100)
    else:
        print("Coluna 'Machine failure' não encontrada.")


if __name__ == "__main__":
    main()
