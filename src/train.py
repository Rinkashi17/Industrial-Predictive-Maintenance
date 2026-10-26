
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# Caminhos do projeto
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "failure_model.joblib"


def main():
    # 1. Carregar os dados
    df = pd.read_csv(DATA_PATH)

    # 2. Definir o que queremos prever
    target = "Machine failure"
    y = df[target]

    # 3. Separar as entradas do modelo
    # Removemos a resposta, os indicadores de falha
    # e os identificadores dos registros/produtos.
    excluded_columns = [
        target,
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "UDI",
        "Product ID",
    ]

    X = df.drop(columns=excluded_columns)

    # Identificar colunas categóricas e numéricas
    categorical_columns = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Transformar categorias em colunas numéricas
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )

    # Dividir os dados: 80% treino e 20% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Registros para treino: {len(X_train)}")
    print(f"Registros para teste: {len(X_test)}")

    # Preparação dos dados + modelo em um único pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # 6. Treinar
    print("\nTreinando o modelo...")
    model.fit(X_train, y_train)

    # 7. Fazer previsões nos dados de teste
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 8. Avaliar o desempenho
    print("\n=== MATRIZ DE CONFUSÃO ===")
    print(confusion_matrix(y_test, predictions))

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Sem falha", "Com falha"],
            zero_division=0,
        )
    )

    print(
        "Average precision:",
        round(average_precision_score(y_test, probabilities), 4),
    )

    # 9. Salvar o modelo treinado
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
