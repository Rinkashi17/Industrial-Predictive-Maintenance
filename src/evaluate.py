
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"

TARGET = "Machine failure"

EXCLUDED_COLUMNS = [
    TARGET,
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "UDI",
    "Product ID",
]


def main():
    print("Carregando dataset e modelo...")

    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    y = df[TARGET]
    X = df.drop(columns=EXCLUDED_COLUMNS)

    # Reproduz a mesma separação utilizada no treinamento.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"\nRegistros para teste: {len(X_test)}")
    print("Avaliando previsões...\n")

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, predictions))

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Sem falha", "Com falha"],
            zero_division=0,
        )
    )

    average_precision = average_precision_score(
        y_test, probabilities
    )
    print(f"Average precision: {average_precision:.4f}")

    # Monta uma tabela para investigar os resultados.
    results = X_test.copy()
    results.insert(0, "UDI", df.loc[X_test.index, "UDI"])
    results["Falha real"] = y_test
    results["Falha prevista"] = predictions
    results["Probabilidade de falha (%)"] = (
        probabilities * 100
    ).round(2)

    false_negatives = results[
        (results["Falha real"] == 1)
        & (results["Falha prevista"] == 0)
    ]

    false_positives = results[
        (results["Falha real"] == 0)
        & (results["Falha prevista"] == 1)
    ]

    print("\n=== FALSOS NEGATIVOS ===")
    print("Falhas reais que o modelo não identificou:")
    print(f"Total: {len(false_negatives)}")

    if not false_negatives.empty:
        # Mostra primeiro os casos com maior probabilidade estimada.
        print(
            false_negatives.sort_values(
                "Probabilidade de falha (%)",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
    else:
        print("Nenhum caso encontrado.")

    print("\n=== FALSOS POSITIVOS ===")
    print("Casos sem falha que o modelo classificou como falha:")
    print(f"Total: {len(false_positives)}")

    if not false_positives.empty:
        print(
            false_positives.sort_values(
                "Probabilidade de falha (%)",
                ascending=False,
            )
            .head(10)
            .to_string(index=False)
        )
    else:
        print("Nenhum caso encontrado.")

    print("\nAvaliação concluída.")
    print(
        "Esta análise é experimental e não comprova "
        "a causa física das falhas."
    )


if __name__ == "__main__":
    main()
