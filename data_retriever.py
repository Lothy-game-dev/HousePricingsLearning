import os
import requests
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv, set_key
import json
import datetime

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flights_data.db'
db = SQLAlchemy(app)

# Define the FlightData model
class FlightData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departure_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    arrival_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    flight_date = db.Column(db.String(20), nullable=False)
    departure_time = db.Column(db.String(20), nullable=False)
    arrival_time = db.Column(db.String(20), nullable=False)

class Airports(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(3), nullable=False)
    name = db.Column(db.String(100), nullable=False)

def retrieve_flight_data(departure_airport, end_airport, departure_date):
    url = 'https://api.aviationstack.com/v1/flights'

    params = {
        'access_key': os.getenv('AVIATION_API_KEY'),
        'dep_iata': departure_airport,
        'arr_iata': end_airport
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        flight_data = response.json()
        for flight in flight_data.get('data', []):
            date = flight['flight_date']
            if date != departure_date:
                continue
            flight_data_to_save = {
                'flight_date': flight['flight_date'],
                'departure_airport': db.session.query(Airports.id).filter_by(key=flight['departure']['iata']).first()[0],
                'arrival_airport': db.session.query(Airports.id).filter_by(key=flight['arrival']['iata']).first()[0],
                'departure_time': flight['departure']['estimated'],
                'arrival_time': flight['arrival']['estimated'],
            }
            # Save the flight data to the database
            new_flight = FlightData(
                departure_airport_id=flight_data_to_save['departure_airport'],
                arrival_airport_id=flight_data_to_save['arrival_airport'],
                flight_date = flight_data_to_save['flight_date'],
                departure_time=flight_data_to_save['departure_time'],
                arrival_time=flight_data_to_save['arrival_time'],
            )
            db.session.add(new_flight)
            db.session.commit()
            print('Add data to database success')
        return flight_data
    else:
        print("Error:", response.status_code, response.text)
        return None

def save_airports_to_db_aviation():
    aviation_data_saved = os.getenv('AVIATION_DATA_SAVED')
    aviation_api_key = os.getenv('AVIATION_API_KEY')

    if aviation_data_saved.lower() == 'false':
        url = f'http://api.aviationstack.com/v1/airlines?access_key={aviation_api_key}&limit=10000'
        response = requests.get(url)

        if response.status_code == 200:
            airlines_data = response.json().get('data', [])
            for airline in airlines_data:
                airline_key = airline.get('hub_code')
                airline_name = airline.get('airline_name')
                if airline_key and airline_name:
                    airport = Airports(key=airline_key, name=airline_name)
                    db.session.add(airport)
            db.session.commit()
            dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
            set_key(dotenv_path, 'AVIATION_DATA_SAVED', 'true')
            print("Airports data saved to the database.")
        else:
            print("Error:", response.status_code, response.text)
    else:
        print("Data from Aivation API already saved!")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        save_airports_to_db_aviation()
        retrieve_flight_data('SGN', 'CAN', '2024-09-22')
