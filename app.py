from flask import Flask, render_template, request, send_file, session, redirect, url_for
import pandas as pd
import joblib
import io
from prepare_utils import prepare_data

app = Flask(__name__)
app.secret_key = "secret_key_for_chat_history"

# Charger le modèle
model = joblib.load("final_lgbm_model.pkl")


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    # Initialiser l'historique du chatbot s'il n'existe pas
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        try:
            data = {
                'loan_amnt': float(request.form['loan_amnt']),
                'term': f"{int(request.form['term'])} months",
                'int_rate': float(request.form['int_rate']),
                'annual_inc': float(request.form['annual_inc']),
                'emp_length': request.form['emp_length'],
                'grade': request.form['grade'],
                'sub_grade': request.form['sub_grade'],
                'home_ownership': request.form['home_ownership'],
                'purpose': request.form['purpose'],
                'verification_status': request.form['verification_status'],
                'revol_util': float(request.form['revol_util']),
                'dti': float(request.form['dti']),
                'installment': float(request.form['installment']),
                'revol_bal': float(request.form['revol_bal']),
                'open_acc': int(request.form['open_acc']),
                'total_acc': int(request.form['total_acc']),
                'mort_acc': float(request.form['mort_acc']),
                'pub_rec_bankruptcies': float(request.form['pub_rec_bankruptcies']),
                'pub_rec': float(request.form['pub_rec']),
                'initial_list_status': request.form['initial_list_status'],
                'application_type': request.form['application_type'],
                'issue_d': request.form['issue_d'],
                'earliest_cr_line': request.form['earliest_cr_line']
            }

            df_input = pd.DataFrame([data])
            df_prepared = prepare_data(df_input)
            prediction = int(model.predict(df_prepared)[0])

        except Exception as e:
            return render_template("form.html", error_message=str(e), chat_history=session["chat_history"])

    return render_template("form.html", prediction=prediction, csv_summary=None, csv_download=None, chat_history=session["chat_history"])


@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "csv_file" not in request.files:
        return "Aucun fichier reçu."

    if "chat_history" not in session:
        session["chat_history"] = []

    file = request.files["csv_file"]
    try:
        df = pd.read_csv(file, sep=';')
        df_prepared = prepare_data(df)
        df["prediction"] = model.predict(df_prepared)

        total = len(df)
        risque = int((df["prediction"] == 1).sum())
        fiables = int((df["prediction"] == 0).sum())
        head_html = df[['loan_amnt', 'int_rate', 'grade', 'purpose', 'prediction']].head().to_html(index=False)

        output = io.StringIO()
        df.to_csv(output, index=False, sep=';')
        output.seek(0)
        csv_str = output.getvalue()

        return render_template(
            "form.html",
            prediction=None,
            csv_summary={"total": total, "risque": risque, "fiables": fiables, "head_html": head_html},
            csv_download=csv_str,
            chat_history=session["chat_history"]
        )

    except Exception as e:
        return render_template("form.html", error_message=str(e), chat_history=session["chat_history"])


@app.route("/download_csv", methods=["POST"])
def download_csv():
    csv_content = request.form.get("csv_content", "")
    return send_file(io.BytesIO(csv_content.encode()), mimetype="text/csv", as_attachment=True, download_name="predictions_output.csv")


@app.route("/chat", methods=["POST"])
def chat():
    question = request.form.get("question", "").strip().lower()

    faq = {
        # Variables explicatives
        "dti": "Le DTI est le ratio entre les dettes mensuelles et le revenu mensuel brut.",
        "intérêt": "Le taux d’intérêt est exprimé en pourcentage annuel.",
        "emp_length": "Ancienneté de l’emprunteur dans son emploi.",
        "grade": "Note de crédit allant de A (très bon) à G (risqué).",
        "sub_grade": "Sous-note précisant la granularité (ex: B3, C4).",
        "modèle": "Le modèle utilisé est LightGBM pour prédire le défaut de paiement.",
        "revol_util": "Pourcentage du crédit renouvelable utilisé par l'emprunteur.",
        "rgpd": "L'application respecte le RGPD : aucune donnée personnelle stockée.",
        "application_type": "Type de demande : individuelle ou conjointe.",
        "loan_amnt": "Montant demandé pour le prêt.",
        "term": "Durée du prêt : 36 ou 60 mois.",
        "installment": "Mensualité due chaque mois.",
        "home_ownership": "Statut de logement : propriétaire, locataire, hypothéqué.",
        "annual_inc": "Revenu annuel déclaré.",
        "verification_status": "Statut de vérification du revenu : vérifié, non vérifié, etc.",
        "issue_d": "Date d’émission du prêt.",
        "earliest_cr_line": "Date de la première ligne de crédit dans le dossier.",
        "open_acc": "Nombre de crédits actuellement ouverts.",
        "pub_rec": "Nombre de documents publics dérogatoires.",
        "revol_bal": "Solde dû sur le crédit renouvelable.",
        "total_acc": "Nombre total de comptes de crédit.",
        "initial_list_status": "Statut initial du prêt (f ou w).",
        "mort_acc": "Nombre de prêts immobiliers actifs.",
        "pub_rec_bankruptcies": "Nombre de faillites dans les documents publics.",
        "purpose": "Motif du prêt : dettes, voiture, travaux, etc.",
        "client à risque": "Un client est considéré comme à risque si la prédiction vaut 1 (défaut).",

        # Questions globales
        "fiabilité": "Le modèle a été validé avec des scores de précision satisfaisants et une validation croisée.",
        "variable la plus influente": "Les variables les plus influentes sont : DTI, taux d’intérêt et montant du prêt.",
        "pourquoi ce client est à risque": "Cela peut être dû à un taux élevé d’intérêt ou de DTI, un faible revenu ou un historique de crédit négatif.",
        "pourquoi ce client est fiable": "Probablement un bon revenu,un bon taux d’intérêt ou de DTI une faible utilisation de crédit et une bonne ancienneté.",
    }

    if "chat_history" not in session:
        session["chat_history"] = []

    # Chercher la réponse dans les mots-clés
    response = "Désolé, je ne comprends pas votre question. Essayez avec : 'DTI', 'modèle', 'client à risque'..."
    for keyword, answer in faq.items():
        if keyword in question:
            response = answer
            break

    # Ajouter à l’historique
    session["chat_history"].append({"question": question, "response": response})
    session.modified = True  # Pour forcer la mise à jour de la session

    return redirect(url_for("predict"))


@app.route("/reset_chat")
def reset_chat():
    session.pop("chat_history", None)
    return redirect(url_for("predict"))


if __name__ == "__main__":
    app.run(debug=True)
