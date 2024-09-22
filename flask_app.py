from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flights_data.db'
db = SQLAlchemy(app)

# Define the FlightData model
class FlightData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departure_airport = db.Column(db.String(100), nullable=False)
    arrival_airport = db.Column(db.String(100), nullable=False)
    departure_time = db.Column(db.String(100), nullable=False)
    arrival_time = db.Column(db.String(100), nullable=False)
    price = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<FlightData {self.departure_airport} to {self.arrival_airport}>'

# Create the database tables
with app.app_context():
    db.create_all()

# Create a new flight
@app.route('/flights', methods=['POST'])
def create_flight():
    data = request.get_json()
    new_flight = FlightData(
        departure_airport=data['departure_airport'],
        arrival_airport=data['arrival_airport'],
        departure_time=data['departure_time'],
        arrival_time=data['arrival_time'],
        price=data['price']
    )
    db.session.add(new_flight)
    db.session.commit()
    return jsonify({'message': 'Flight created successfully'}), 201

# Retrieve all flights
@app.route('/flights', methods=['GET'])
def get_flights():
    flights = FlightData.query.all()
    return render_template('flights.html', flights=flights)

# Retrieve a single flight by ID
@app.route('/flights/<int:id>', methods=['GET'])
def get_flight(id):
    flight = FlightData.query.get_or_404(id)
    return render_template('flight.html', flight=flight)

# Search for flights by departure and arrival airports
@app.route('/flights/search', methods=['GET'])
def search_flights():
    departure_airport = request.args.get('departure_airport')
    arrival_airport = request.args.get('arrival_airport')
    flights = FlightData.query.filter_by(departure_airport=departure_airport, arrival_airport=arrival_airport).all()
    return render_template('flights.html', flights=flights)

if __name__ == '__main__':
    app.run(debug=True)
