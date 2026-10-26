
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "failure_model.joblib"


def main():
    model = joblib.load(MODEL_PATH)

    print("=== DADOS DA MÁQUINA ===")

    machine_type = input("Tipo da máquina (L, M ou H): ").strip().upper()

    if machine_type not in {"L", "M", "H"}:
        print("Tipo inválido. Escolha L, M ou H.")
        return

    try:
        air_temperature = float(input("Temperatura do ar (K): "))
        process_temperature = float(input("Temperatura do processo (K): "))
        rotational_speed = int(input("Velocidade de rotação (rpm): "))
        torque = float(input("Torque (Nm): "))
        tool_wear = float(input("Desgaste da ferramenta (min): "))
    except ValueError:
        print("Entrada inválida. Digite números nos campos numéricos.")
        return

    machine = {
        "Type": machine_type,
        "Air temperature [K]": air_temperature,
        "Process temperature [K]": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }

    X_new = pd.DataFrame([machine])

    prediction = int(model.predict(X_new)[0])
    failure_probability = float(model.predict_proba(X_new)[0, 1])

    print("\n=== RESULTADO ===")

    if prediction == 1:
        print("Classificação: COM FALHA")
    else:
        print("Classificação: SEM FALHA")

    print(f"Probabilidade estimada de falha: {failure_probability:.1%}")
    print("\nDemonstração experimental com dados informados pelo usuário.")
    print("O resultado não deve ser usado como diagnóstico industrial.")


if __name__ == "__main__":
    main()
