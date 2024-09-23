from flask import request, jsonify, render_template

from models import db, app, Airports, FlightData, RouteData
from data_retriever import retrieve_flight_data

# Retrieve all flights
@app.route('/', methods=['GET'])
def index():
    airports = Airports.query.all()
    airports_list = [{'id': airport.id, 'name': airport.name} for airport in airports]
    return render_template('index.html', flights=None, airports=airports_list, data_chosen={})

# Retrieve a single flight by ID
@app.route('/flight/<int:id>', methods=['GET'])
def get_flight(id):
    flight = RouteData.query.get_or_404(id)
    all_other_flights = RouteData.query.filter(
        RouteData.departure_airport_id == flight.departure_airport_id,
        RouteData.arrival_airport_id == flight.arrival_airport_id,
    ).all()
    return render_template('detail.html', flight=flight, all_other_flights=all_other_flights)


# Search for flights by departure and arrival airports
@app.route('/flights/search', methods=['GET'])
def search_flights():
    # temp
    departure_airport_id = request.args.get('departure_airport')
    print(departure_airport_id)
    departure_airport = db.session.query(Airports.key).filter_by(id=departure_airport_id).first()[0]
    arrival_airport_id = request.args.get('arrival_airport')
    print(arrival_airport_id)
    arrival_airport = db.session.query(Airports.key).filter_by(id=arrival_airport_id).first()[0]
    departure_date = request.args.get('departure_date')
    arrival_date = request.args.get('arrival_date')
    data_chosen = {
        'departure_airport': departure_airport_id,
        'arrival_airport': arrival_airport_id,
        'departure_date': departure_date,
        'arrival_date': arrival_date
    }
    try:
        flights = retrieve_flight_data(departure_airport, arrival_airport, departure_date, arrival_date)
    except Exception as e:
        print(e)
        flights = []
    airports = Airports.query.all()
    airports_list = [{'id': airport.id, 'name': airport.name} for airport in airports]
    return render_template('index.html', flights=flights, airports=airports_list, data_chosen=data_chosen)

if __name__ == '__main__':
    app.run(debug=True)
