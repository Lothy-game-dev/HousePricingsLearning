from flask import request, jsonify, render_template

from data_retriever import FlightData
from models import db, app, Airports


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
    departure_airport = Airports.query.filter_by(key=departure_airport).first()
    arrival_airport = Airports.query.filter_by(key=arrival_airport).first()
    flights = FlightData.query.filter_by(departure_airport_id=departure_airport.id, arrival_airport_id=arrival_airport.id).all()
    return jsonify({
        'departure_airport': departure_airport.serialize(),
        'arrival_airport': arrival_airport.serialize(),
        'flights': [flight.serialize() for flight in flights]
    })

if __name__ == '__main__':
    app.run(debug=True)
