from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from model import predict_health_risk
from scipy.integrate import odeint

app = Flask('__name__')

# Database configuration
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "health.db")}'
app.config['SECRET_KEY'] = 'iutfgpsofioiysdf9e98p9yufuopsdufloi'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(10), nullable=False)

class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    blood_group = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    blood_pressure = db.Column(db.String(50), nullable=False)
    gender = db.Column(db.String(10), nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Helper function to check if the user is authenticated
def is_authenticated():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        if user:
            return True
    return False

# Routes
@app.route('/')
def home_root():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_root():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']

        if not (username and email and password and role):
            return "Please fill all the fields."

        user = User(username=username, email=email, password=password, role=role)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for('login_root'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login_root():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email, password=password).first()

        if not user:
            return "Invalid credentials. Try again."

        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role

        if user.role == 'Admin':
            return redirect(url_for('admin_dashboard'))
        elif user.role == 'Doctor':
            return redirect(url_for('doctor_dashboard'))
        elif user.role == 'Patient':
            return redirect(url_for('patient_dashboard'))

    return render_template('login.html')

@app.route('/patient_dashboard', methods=['GET', 'POST'])
def patient_dashboard():
    if not is_authenticated() or session['role'] != 'Patient':
        return redirect(url_for('login_root'))

    if request.method == 'POST':
        # Save health data
        health_data = HealthData(
            patient_id=session['user_id'],
            age=request.form['age'],
            bmi=request.form['bmi'],
            blood_group=request.form['blood_group'],
            height=request.form['height'],
            cholesterol=request.form['cholesterol'],
            blood_pressure=request.form['blood_pressure'],
            gender=request.form['gender']
        )
        db.session.add(health_data)
        db.session.commit()
        flash("Health data submitted successfully!", "success")

    # Fetch messages
    messages = Message.query.filter_by(receiver_id=session['user_id']).all()
    return render_template('patient_dashboard.html', messages=messages)

@app.route('/doctor_dashboard', methods=['GET', 'POST'])
def doctor_dashboard():
    if not is_authenticated() or session['role'] != 'Doctor':
        return redirect(url_for('login_root'))

    if request.method == 'POST':
        # Send message to a patient
        receiver_id = request.form['patient_id']
        content = request.form['message']
        message = Message(sender_id=session['user_id'], receiver_id=receiver_id, content=content)
        db.session.add(message)
        db.session.commit()
        flash("Message sent to the patient!", "success")

    # Fetch patient data
    patients = User.query.filter_by(role='Patient').all()
    health_data = HealthData.query.all()
    
    # Fetch messages
    messages = Message.query.filter_by(receiver_id=session['user_id']).all()
    return render_template('doctor_dashboard.html', patients=patients, health_data=health_data, messages=messages)

@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    if not is_authenticated() or session['role'] != 'Admin':
        return redirect(url_for('login_root'))

    if request.method == 'POST':
        # Delete a user
        user_id = request.form['user_id']
        user = User.query.get(user_id)
        db.session.delete(user)
        db.session.commit()
        flash("User deleted successfully!", "success")

    users = User.query.order_by(db.case(
        (User.role == 'Admin', 1),
        (User.role == 'Doctor', 2),
        (User.role == 'Patient', 3),
        else_=4
    )).all()
    health_data = HealthData.query.all()
    return render_template('admin_dashboard.html', users=users, health_data=health_data)

@app.route('/send_message', methods=['POST'])
def send_message():
    if not is_authenticated():
        return redirect(url_for('login_root'))

    receiver_id = request.form['receiver_id']
    content = request.form['content']
    message = Message(sender_id=session['user_id'], receiver_id=receiver_id, content=content)
    db.session.add(message)
    db.session.commit()
    flash("Message sent successfully!", "success")
    return redirect(request.referrer)

@app.route('/delete_patient', methods=['POST'])
def delete_patient():
    patient_id = request.form.get('patient_id')
    # Delete the patient from the HealthData table (if it's tied to HealthData)
    patient = HealthData.query.filter_by(patient_id=patient_id).first()
    
    if patient:
        db.session.delete(patient)
        db.session.commit()
        return jsonify({"status": "success", "message": "Patient deleted successfully!"})
    
    return jsonify({"status": "error", "message": "Patient not found."})

@app.route('/delete_message', methods=['POST'])
def delete_message():
    message_id = request.form.get('message_id')
    # Delete the message from the database
    message = Message.query.filter_by(id=message_id).first()
    if message:
        db.session.delete(message)
        db.session.commit()
        return jsonify({"status": "success", "message": "Message deleted successfully!"})
    return jsonify({"status": "error", "message": "Message not found."})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home_root'))

@app.route('/simulate', methods=['POST'])
def simulate():
    # Get data from form
    age = int(request.form['age'])
    initial_infected = int(request.form['initial_infected'])
    contact_rate = float(request.form['contact_rate'])
    recovery_rate = float(request.form['recovery_rate'])
    disease_duration = int(request.form['disease_duration'])

    # Total population size
    population = 1000
    initial_susceptible = population - initial_infected

    # Define SIR model equations
    def sir_model(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / population
        dIdt = beta * S * I / population - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

    # Initial conditions: S0, I0, R0
    S0 = initial_susceptible
    I0 = initial_infected
    R0 = 0  # Initially no one is recovered

    # Time points (days)
    t = np.arange(0, disease_duration + 1, 1)  # Ensure whole numbers (integers)

    # Solve ODE
    sol = odeint(sir_model, [S0, I0, R0], t, args=(contact_rate, recovery_rate))

    # Prepare results
    susceptible = sol[:, 0].tolist()
    infected = sol[:, 1].tolist()
    recovered = sol[:, 2].tolist()

    # Generate description based on simulation results
    max_infected = max(infected)
    peak_day = t[np.argmax(infected)]  # Find the day with the peak infection
    end_day = t[-1]  # Last day of the simulation
    total_recovered = round(recovered[-1])  # Total recovered at the end of the simulation

    description = f"The disease peaked on day {peak_day} with {round(max_infected)} infected individuals. " \
                  f"After day {peak_day}, the number of infected people started to decrease as more individuals recovered. " \
                  f"By the end of the simulation (day {end_day}), {total_recovered} people had recovered. " \
                  f"The disease is expected to subside over time as the number of susceptible individuals decreases."

    return jsonify({
        "time": t.tolist(),
        "susceptible": susceptible,
        "infected": infected,
        "recovered": recovered,
        "description": description
    })



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=3030)
