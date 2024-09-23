import os

import requests
from dotenv import set_key

from models import Airports, db, app, FlightData, RouteData


def retrieve_flight_data(departure_airport, end_airport, departure_date, end_date):
    # Check if there are existing routes in the database for the given parameters
    existing_routes = RouteData.query.filter_by(
        departure_airport_id=db.session.query(Airports.id).filter_by(key=departure_airport).first()[0],
        arrival_airport_id=db.session.query(Airports.id).filter_by(key=end_airport).first()[0],
        departure_date=departure_date,
        end_date=end_date,
    ).all()

    # If existing routes are found, return them
    if existing_routes:
        print(f'{len(existing_routes)} existing routes found')
        return [route.serialize() for route in existing_routes]

    # If no existing routes are found, fetch new data from the API
    url = 'https://serpapi.com/search'

    # Set up the parameters for the API request
    params = {
        'engine': 'google_flights',
        'departure_id': departure_airport,
        'arrival_id': end_airport,
        'outbound_date': departure_date,
        'return_date': end_date,
        'api_key': os.getenv("SERAPI_KEY"),
    }

    # Set up the headers for the API request
    headers = {
        'Authorization': f'Bearer {os.getenv("SERAPI_KEY")}'
    }

    # Make the API request
    response = requests.get(url, params=params, headers=headers)
    print(response.request.url)

    # If the request is successful, process the response data
    if response.status_code == 200:
        # Get the airport IDs from the database
        departure_airport_id = db.session.query(Airports.id).filter_by(key=departure_airport).first()[0]
        end_airport_id = db.session.query(Airports.id).filter_by(key=end_airport).first()[0]

        # Parse the response JSON
        flight_data = response.json()
        flight_return_data = []
        if flight_data.get('best_flights'):

            # Process the best flights
            for flight in flight_data.get('best_flights', []):
                flight_inner_data = flight['flights']
                route_data_to_save = {
                    'departure_airport_id': departure_airport_id,
                    'arrival_airport_id': end_airport_id,
                    'departure_date': departure_date,
                    'end_date': end_date,
                    'total_price': flight['price'],
                    'total_duration': flight['total_duration'],
                    'flights': [],
                }
                # Save the route data to the database
                new_route = RouteData(
                    departure_airport_id=route_data_to_save['departure_airport_id'],
                    arrival_airport_id=route_data_to_save['arrival_airport_id'],
                    departure_date=route_data_to_save['departure_date'],
                    end_date=route_data_to_save['end_date'],
                    total_price=route_data_to_save['total_price'],
                    total_duration=route_data_to_save['total_duration']
                )
                db.session.add(new_route)
                db.session.commit()
                route_data_to_save['id'] = new_route.id
                for flight_inner in flight_inner_data:
                    flight_data_to_save = {
                        'departure_airport': db.session.query(Airports.id).filter_by(key=flight_inner['departure_airport']['id']).first()[0],
                        'arrival_airport': db.session.query(Airports.id).filter_by(key=flight_inner['arrival_airport']['id']).first()[0],
                        'departure_time': flight_inner['departure_airport']['time'],
                        'arrival_time': flight_inner['arrival_airport']['time'],
                        'airline_logo': flight_inner['airline_logo'],
                    }
                    # Save the flight data to the database
                    new_flight = FlightData(
                        departure_airport_id=flight_data_to_save['departure_airport'],
                        arrival_airport_id=flight_data_to_save['arrival_airport'],
                        departure_time=flight_data_to_save['departure_time'],
                        arrival_time=flight_data_to_save['arrival_time'],
                        airline_logo=flight_data_to_save['airline_logo'],
                        route_id=route_data_to_save['id']
                    )
                    db.session.add(new_flight)
                    db.session.commit()
                    print('Add best flight data to database success')
                    route_data_to_save['flights'].append(new_flight.serialize())
                flight_return_data.append(route_data_to_save)

            # Process the other flights
            for flight in flight_data.get('other_flights', []):
                flight_inner_data = flight['flights']
                route_data_to_save = {
                    'departure_airport_id': departure_airport_id,
                    'arrival_airport_id': end_airport_id,
                    'departure_date': departure_date,
                    'end_date': end_date,
                    'total_price': flight['price'],
                    'total_duration': flight['total_duration'],
                    'flights': [],
                }
                # Save the route data to the database
                new_route = RouteData(
                    departure_airport_id=route_data_to_save['departure_airport_id'],
                    arrival_airport_id=route_data_to_save['arrival_airport_id'],
                    departure_date=route_data_to_save['departure_date'],
                    end_date=route_data_to_save['end_date'],
                    total_price=route_data_to_save['total_price'],
                    total_duration=route_data_to_save['total_duration']
                )
                db.session.add(new_route)
                db.session.commit()
                route_data_to_save['id'] = new_route.id
                for flight_inner in flight_inner_data:
                    flight_data_to_save = {
                        'departure_airport': db.session.query(Airports.id).filter_by(key=flight_inner['departure_airport']['id']).first()[0],
                        'arrival_airport': db.session.query(Airports.id).filter_by(key=flight_inner['arrival_airport']['id']).first()[0],
                        'departure_time': flight_inner['departure_airport']['time'],
                        'arrival_time': flight_inner['arrival_airport']['time'],
                        'airline_logo': flight_inner['airline_logo'],
                    }
                    # Save the flight data to the database
                    new_flight = FlightData(
                        departure_airport_id=flight_data_to_save['departure_airport'],
                        arrival_airport_id=flight_data_to_save['arrival_airport'],
                        departure_time=flight_data_to_save['departure_time'],
                        arrival_time=flight_data_to_save['arrival_time'],
                        airline_logo=flight_data_to_save['airline_logo'],
                        route_id=route_data_to_save['id']
                    )
                    db.session.add(new_flight)
                    db.session.commit()
                    print('Add other flight data to database success')
                    route_data_to_save['flights'].append(new_flight.serialize())
                flight_return_data.append(route_data_to_save)
            return flight_return_data
        else:
            print('No flight data found')
            return None
    else:
        print("Error:", response.status_code, response.text)
        return None

def save_airports_to_db_aviation():
    # Check if the aviation data has already been saved
    aviation_data_saved = os.getenv('AVIATION_DATA_SAVED')
    aviation_api_key = os.getenv('AVIATION_API_KEY')

    if aviation_data_saved.lower() == 'false':
        # Fetch the aviation data from the API
        url = f'http://api.aviationstack.com/v1/airlines?access_key={aviation_api_key}&limit=10000'
        response = requests.get(url)

        # If the request is successful, save the data to the database
        if response.status_code == 200:
            airlines_data = response.json().get('data', [])
            for airline in airlines_data:
                airline_key = airline.get('hub_code')
                airline_name = airline.get('airline_name')
                if airline_key and airline_name:
                    airport = Airports(key=airline_key, name=airline_name)
                    db.session.add(airport)
            db.session.commit()
            # Update the .env file to indicate that the data has been saved
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
        # testing only
        save_airports_to_db_aviation()
        retrieve_flight_data('SGN', 'CAN', '2024-09-23', '2024-09-25')
