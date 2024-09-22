import os
import requests
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv

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

# Function to retrieve data from SerpApi
def retrieve_flight_data():
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("No API key found. Please set the API_KEY environment variable.")

    url = 'https://serpapi.com/search'
    params = {
        'engine': 'google_flights',
        'q': 'Flights from New York to Los Angeles',
        'api_key': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Process and save data to the database
    for flight in data.get('flights_results', []):
        flight_data = FlightData(
            departure_airport=flight['departure_airport'],
            arrival_airport=flight['arrival_airport'],
            departure_time=flight['departure_time'],
            arrival_time=flight['arrival_time'],
            price=flight['price']
        )
        db.session.add(flight_data)
    db.session.commit()

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        retrieve_flight_data()
