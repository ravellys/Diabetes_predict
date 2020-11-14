import joblib
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


def previsao_diabetes(lista_formulario):
    prever = np.array(lista_formulario).reshape(1, -1)
    modelo_norm = joblib.load('modelo_ml/data_normaliza.sav')
    prever_norm = modelo_norm.transform(prever)
    modelo_ml_salvo = joblib.load('modelo_ml/melhor_modelo_ml.sav')
    resultado = modelo_ml_salvo.predict(prever_norm)
    return resultado[0]


@app.route("/result", methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        print(request)
        lista_formulario = request.form.to_dict()
        print(lista_formulario)
        lista_formulario = list(lista_formulario.values())
        lista_formulario = list(map(float, lista_formulario))

        resultado = previsao_diabetes(lista_formulario)

        print(lista_formulario)

        if int(resultado) == 1:
            previsao = 'Possui diabetes'
        else:
            previsao = 'Não Possui diabetes'

        return render_template('result.html', previsao=previsao)



if __name__ == "__main__":
    app.run(debug=True)
