import pandas as pd
from joblib import load
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from decimal import Decimal


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)

@st.cache_data
def load_model(path: str):
    return load(path)

def draw_boxplots(data: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        yes_data = data.loc[data["label"] == 1].drop(columns=["label"])
        fig1, ax1 = plt.subplots()
        sns.boxplot(data=yes_data, ax=ax1)
        ax1.set_ylabel("Значения")
        ax1.set_title("Распределение получивших автомат")
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        no_data = data.loc[data["label"] == 0].drop(columns=["label"])
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=no_data, ax=ax2)
        ax2.set_ylabel("Значения")
        ax2.set_title("Распределение НЕ получивших автомат")
        st.pyplot(fig2)
        plt.close(fig2)

def print_metrics(acc, f1, precision, recall):
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label="Accuracy", value=acc)
        st.metric(label="F1-score", value=f1)
    with c2:
        st.metric(label="Precision", value=precision)
        st.metric(label="Recall", value=recall)

def input_scores():
    c1, c2 = st.columns(2)
    test1 = c1.number_input("Тест 1")
    test2 = c2.number_input("Тест 2")
    test3 = c1.number_input("Тест 3")
    test4 = c2.number_input("Тест 4")
    return [test1, test2, test3, test4]

def main():
    st.write('**Программа предназначена для студентов 3-го курса ВМК. С помощью нее вы узнаете вероятность получения автомата по Оптимальному управлению в 5-м семестре. Модель была обучена на данных прошлых лет**')

    data = load_data("clean_data.csv")
    draw_boxplots(data)

    st.write('**Была использована модель логистической регрессии. Получились следующие метрики:**')
    acc, precision, recall, f1_score = 0.94, 0.93, 1.00, 0.96
    print_metrics(acc, f1_score, precision, recall)

    st.header("Введите свои баллы за тесты:")
    scores = input_scores()

    model = load_model("ou_model.joblib")

 
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        try:
            feature_names = model[-1].feature_names_in_
        except Exception:
            feature_names = ["Test1", "Test2", "Test3", "Test4"]

    X = pd.DataFrame([scores], columns=list(feature_names))
    prob = float(model.predict_proba(X)[0][1])
    st.success(f"Вероятность получить автомат: {prob:.3f}")

if __name__ == "__main__":
    main()
