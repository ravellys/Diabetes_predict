import joblib
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def pre_tratamento_dados(df, feature_saida):
    x = df.drop(columns=[feature_saida])  # entrada
    y = df[feature_saida]  # saida

    # Normalização dos dados
    from sklearn.preprocessing import MinMaxScaler
    normaliza = MinMaxScaler()  # objeto para a normalização
    entradas_normalizadas = normaliza.fit_transform(x)
    return entradas_normalizadas, y, normaliza


def modelo_ml(model, x, y):
    # treino / teste
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)

    return clf, acc


def selecao_melhor_modelo(modelos, df, feature_saida):
    x, y, normal = pre_tratamento_dados(df, feature_saida)

    lista_modelos = list(modelos.keys())
    melhor = modelo_ml(modelos[lista_modelos[0]], x=x, y=y)

    for modelo_name in lista_modelos[1:]:
        modelo = modelo_ml(modelos[modelo_name], x=x, y=y)
        if modelo[1] > melhor[1]:
            melhor = modelo

    return melhor, normal


modelos = dict(
    KNeighborsClassifier=KNeighborsClassifier(n_neighbors=5),
    MLPClassifier=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 10), random_state=1),
    DecisionTreeClassifier=DecisionTreeClassifier(random_state=1)
)

if __name__ == '__main__' :
    df = pd.read_csv(
        'https://raw.githubusercontent.com/ravellys/Diabetes_predict/main/dataset/pima-indians-diabetes.csv',
        header=None)
    df.columns = ['NUM_GRAV', 'CONCENTRACAO_GLICOSE', 'PRESSSAO_DIASTOLICA', 'ESPESSURA_TRICEPS', 'INSULINA', 'IMC',
                  'HISTORICO_FAMILIAR', 'IDADE', 'CLASSIFICACAO']

    melhor_modelo = selecao_melhor_modelo(modelos, df=df, feature_saida='CLASSIFICACAO')
    nome_arquivo = 'melhor_modelo_ml.sav'
    joblib.dump(melhor_modelo[0][0], nome_arquivo)
    nome_arquivo = 'data_normaliza.sav'
    joblib.dump(melhor_modelo[1], nome_arquivo)
