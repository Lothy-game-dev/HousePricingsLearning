# Load environment variables from .env file
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

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

    def serialize(self):
        return {
            'id': self.id,
            'departure_airport': self.departure_airport_id,
            'arrival_airport': self.arrival_airport_id,
            'flight_date': self.flight_date,
            'departure_time': self.departure_time,
            'arrival_time': self.arrival_time
        }

class Airports(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(3), nullable=False)
    name = db.Column(db.String(100), nullable=False)

    def serialize(self):
        return {
            'id': self.id,
            'key': self.key,
            'name': self.name
        }
