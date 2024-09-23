# Load environment variables from .env file
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

load_dotenv()

# Initialize Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///flights_data.db'
db = SQLAlchemy(app)

class RouteData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departure_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    arrival_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    departure_date = db.Column(db.String(20), nullable=False)
    end_date = db.Column(db.String(20), nullable=False)
    # in USD
    total_price = db.Column(db.Float, nullable=False)
    # in minutes
    total_duration = db.Column(db.Float, nullable=False)

    def serialize(self):
        return {
            'id': self.id,
            'departure_airport': self.departure_airport_id,
            'arrival_airport': self.arrival_airport_id,
            'departure_date': self.departure_date,
            'end_date': self.end_date,
            'total_price': self.total_price,
            'total_duration': self.total_duration
        }

# Define the FlightData model
class FlightData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    departure_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    arrival_airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'), nullable=False)
    departure_time = db.Column(db.String(20), nullable=False)
    arrival_time = db.Column(db.String(20), nullable=False)
    airline_logo = db.Column(db.String(255), nullable=False)
    route_id = db.Column(db.Integer, db.ForeignKey('route_data.id'), nullable=False)

    def serialize(self):
        return {
            'id': self.id,
            'departure_airport': self.departure_airport_id,
            'arrival_airport': self.arrival_airport_id,
            'departure_time': self.departure_time,
            'arrival_time': self.arrival_time,
            'airline_logo': self.airline_logo,
            'route_id': self.route_id,
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
