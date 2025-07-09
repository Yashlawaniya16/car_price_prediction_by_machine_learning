import csv
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

app.secret_key = 'your_secret_key_here'

model = pickle.load(open('LinearRegression.pkl', 'rb'))
model_avg = pickle.load(open('model_avg.pkl', 'rb'))
model_min = pickle.load(open('model_min.pkl', 'rb'))
model_max = pickle.load(open('model_max.pkl', 'rb'))

car = pd.read_csv('Cleaned_Car_data.csv')


def retrain_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    df = pd.read_csv('Cleaned_Car_data.csv')

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df.dropna(subset=['name', 'company', 'year', 'kms_driven', 'fuel_type', 'Price'], inplace=True)

    if df.empty:
        print("Error: Dataset is empty after cleaning.")
        return

    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']

    cat_cols = ['name', 'company', 'fuel_type']
    ct = ColumnTransformer([
        ('encoder', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', ct),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    model_pipeline.fit(X, y)

    with open('LinearRegression.pkl', 'wb') as f:
        pickle.dump(model_pipeline, f)

    print("Model retrained and saved with RandomForestRegressor.")


@app.route('/retrain', methods=['GET'])
def retrain():
    try:
        retrain_model()
        global model, model_min, model_max
        model = pickle.load(open('LinearRegression.pkl', 'rb'))
        model_min = pickle.load(open('model_min.pkl', 'rb'))
        model_max = pickle.load(open('model_max.pkl', 'rb'))
        flash("All models retrained successfully!")
    except Exception as e:
        flash(f"Error during retraining: {str(e)}")
    return redirect(url_for('index'))


@app.route('/', methods=['GET', 'POST'])
def index():
    car = pd.read_csv('Cleaned_Car_data.csv')
    car['year'] = pd.to_numeric(car['year'], errors='coerce')
    car['Price'] = pd.to_numeric(car['Price'], errors='coerce')
    car['kms_driven'] = pd.to_numeric(car['kms_driven'], errors='coerce')
    car.dropna(subset=['year', 'Price', 'kms_driven'], inplace=True)

    companies = sorted(car['company'].dropna().unique())
    car_models = sorted(car['name'].dropna().unique())
    year = sorted(car['year'].dropna().astype(int).unique(), reverse=True)
    fuel_type = sorted(car['fuel_type'].dropna().unique())

    companies.insert(0, 'Select Company')

    return render_template('index.html',
                           companies=companies,
                           car_models=car_models,
                           years=year,
                           fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                data=[[car_model, company, year, driven, fuel_type]])

        # 🧠 Predict average price only
        predicted_avg = model_avg.predict(input_df)[0]

        # 🔢 Calculate min and max as 15% range
        min_price = predicted_avg * 0.85
        max_price = predicted_avg * 1.15

        # Round and return to template
        return render_template('result.html',
                               predicted_price=np.round(predicted_avg, 2),
                               min_price=np.round(min_price, 2),
                               max_price=np.round(max_price, 2),
                               car_model=car_model)

    except Exception as e:
        print("❌ Prediction Error:", e)
        return "Error: " + str(e)



@app.route('/add_car', methods=['GET', 'POST'])
def add_car():
    if request.method == 'POST':
        name = request.form['name']
        company = request.form['company']
        year = int(request.form['year'])
        kms = int(request.form['kms_driven'])
        fuel = request.form['fuel_type']
        price = float(request.form['price'])

        new_data = pd.DataFrame([[name, company, year, price, kms, fuel]],
                                columns=['name', 'company', 'year', 'price', 'kms_driven', 'fuel_type'])
        new_data.to_csv('Cleaned_Car_data.csv', mode='a', header=False, index=False)

        retrain_model()
        global model
        model = pickle.load(open('LinearRegression.pkl', 'rb'))

        flash("Car added and model retrained!")
        return redirect(url_for('index'))

    return render_template('add_car.html')


@app.route('/delete_car', methods=['GET', 'POST'])
def delete_car():
    if request.method == 'POST':
        model_to_delete = request.form['car_model'].strip().lower()
        df = pd.read_csv('Cleaned_Car_data.csv')
        original_len = len(df)
        df = df[df['name'].str.lower() != model_to_delete]
        df.to_csv('Cleaned_Car_data.csv', index=False)
        deleted = original_len - len(df)
        if deleted > 0:
            flash(f"Deleted {deleted} entries for model: {model_to_delete}")
        else:
            flash(f"No matching entries found for: {model_to_delete}")
        return redirect(url_for('index'))
    return render_template('delete_car.html')


@app.route('/feedback')
def feedback():
    feedbacks = []
    try:
        with open('feedback.csv', 'r', encoding='utf-8') as file:
            reader = list(csv.reader(file))
            for row in reversed(reader[1:]):
                if len(row) == 3:
                    feedbacks.append({
                        'name': row[0],
                        'rating': int(row[1]),
                        'message': row[2]
                    })
    except FileNotFoundError:
        feedbacks = []
    return render_template('feedback.html', feedbacks=feedbacks)


@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    name = request.form['name']
    rating = request.form['rating']
    feedback_text = request.form['feedback']

    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([name, rating, feedback_text])

    return redirect(url_for('index', success=1))


@app.route('/loan', methods=['GET', 'POST'])
def loan():
    if request.method == 'POST':
        try:
            car_price = float(request.form['car_price'])
            down_payment = float(request.form['down_payment'])
            interest_rate = float(request.form['interest_rate'])
            tenure_years = int(request.form['tenure'])

            loan_amount = car_price - down_payment
            monthly_rate = interest_rate / 12 / 100
            tenure_months = tenure_years * 12

            if loan_amount <= 0 or monthly_rate <= 0 or tenure_months <= 0:
                raise ValueError("Invalid input values")

            emi = (loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months) / \
                  ((1 + monthly_rate)**tenure_months - 1)

            total_payment = emi * tenure_months
            total_interest = total_payment - loan_amount

            return render_template('loan_result.html',
                                   emi=round(emi, 2),
                                   total_payment=round(total_payment, 2),
                                   total_interest=round(total_interest, 2),
                                   loan_amount=round(loan_amount, 2))
        except Exception as e:
            return f"Error: {e}"
    return render_template('loan_form.html')


if __name__ == '__main__':
    app.run(debug=True)
