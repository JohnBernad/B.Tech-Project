from flask import Flask, request, render_template
from playwright.sync_api import sync_playwright
from selectolax.lexbor import LexborHTMLParser
from datetime import datetime, timedelta
import time
import pymongo
import csv
import os
from tqdm import tqdm
from lstm import update_model, test_and_evaluate, predict_price
from keras.models import load_model
import joblib
from waitress import serve

app = Flask(__name__)
client = pymongo.MongoClient("mongodb://localhost:27017/")  # Connect to MongoDB
db = client["flight_data"]  # Select the database
collection = db["flights"]  # Select the collection

# Get today's date
today = datetime.now()
# Format today's date in 'DD-MM-YYYY' format
today_str = today.strftime('%d-%m-%Y')
# Subtract one day from today's date to get yesterday's date
yesterday = today - timedelta(days=1)
# Format yesterday's date in 'DD-MM-YYYY' format
yesterday_str = yesterday.strftime('%d-%m-%Y')
# Include today's date in the filename
tod = f'Dataset/flight_data_{today_str}.csv'
yest = f'Dataset/flight_data_{yesterday_str}.csv'


# Load model, scalers, and encoders outside the function
model = load_model('LSTM_Model/lstm_model.h5')
numerical_scaler = joblib.load("LSTM_Model/numerical_scaler.pkl")
target_scaler = joblib.load("LSTM_Model/target_scaler.pkl")
label_encoders = {}
for col in ['airline', 'source', 'destination']:
    label_encoders[col] = joblib.load(f"LSTM_Model/{col}_encoder.pkl")

print("Click Here: http://127.0.0.1:5000/")

@app.route('/powerbi')
def powerbi():
    return render_template('powerbi.html')
@app.route('/')
def index():
    start_time = time.time() * 1000  # Current time in milliseconds
    return render_template('Home.html', start_time=start_time)


def daily_scrape():
    # Define the destination codes
    destinations = ["TRV", "DEL", "BOM", "BLR", "HYD", "MAA", "CCU", "AMD", "GOX", "PNQ"]

    # Define the class_s as 0 for Economy
    class_s = 0

    # Define the from_place as COK
    from_place = "COK"

    # Generate dates from today till 4 months ahead with 5 days in terval
    dates = [(datetime.now() + timedelta(days=i * 5)).strftime('%Y-%m-%d') for i in range(int(30 * 4 / 5))]
    print(dates)

    # Use sync_playwright
    with sync_playwright() as playwright:
        # Iterate over the destinations and dates
        for to_place in tqdm(destinations, desc="Destinations"):
            for departure_date in tqdm(dates, desc="Dates", leave=False):
                # Retry mechanism
                retries = 3  # Number of retries
                while retries > 0:
                    try:
                        # Get the page and scrape the results
                        parser = get_page(playwright, from_place, to_place, departure_date, class_s)
                        daily_scrape_data = daily_scrape_google_flights(parser, from_place, to_place, departure_date)
                        # print(daily_scrape_data)
                        break  # Break out of the retry loop if successful
                    except Exception as e:
                        # Decrement retries
                        retries -= 1
                        if retries == 0:
                            # If no more retries left, print error and move to the next iteration
                            print(f"Failed to scrape for {from_place} -> {to_place}, {departure_date}: {e}")
                            print("Moving to the next iteration...")
                            break
                        else:
                            # Print error and retry
                            print(
                                f"An error occurred while scraping for {from_place} -> {to_place}, {departure_date}: {e}")
                            print(f"Retrying ({retries} retries left)...")


@app.route('/trigger_daily_scrape')
def trigger_daily_scrape():
    daily_scrape()
    return render_template('Home.html', message='Daily scrape process has been successfully completed.')


@app.route('/update_model')
def update_Model():
    update_model(yest)
    test_and_evaluate(tod)
    return render_template('Home.html', message='Model Updated')


@app.route('/search', methods=['POST'])
def search_flights():
    from_place = request.form['from_place']
    to_place = request.form['to_place']
    departure_date = request.form['departure_date']
    class_s = int(request.form['class'])

    if class_s == 0:
        classs = "Economy"
    elif class_s == 1:
        classs = "Premium Economy"
    elif class_s == 2:
        classs = "Business Class"
    elif class_s == 3:
        classs = "First Class"

    with sync_playwright() as playwright:
        parser = get_page(playwright, from_place, to_place, departure_date, class_s)
        google_flights_results = scrape_google_flights(parser, from_place, to_place, departure_date)

    # print(google_flights_results)
    # Render the template with the flight data
    return render_template('Home.html', flights_data=google_flights_results or {}, from_place=from_place,
                           to_place=to_place, departure_date=departure_date, classs=classs)


@app.route('/prediction_search', methods=['POST'])
def p_search_flights():
    p_airline = request.form['p_airline']
    p_from_place = request.form['p_from_place']
    p_to_place = request.form['p_to_place']
    p_duration = int(request.form['p_duration'])  # Convert to integer
    p_stop = int(request.form['p_stops'])  # Convert to integer
    # Assuming departure_date and booking_date are in the format 'YYYY-MM-DD'
    booking_date_str = request.form['p_Booking_date']
    departure_date_str = request.form['p_departure_date']

    # Convert the dates from string to datetime objects
    booking_date = datetime.strptime(booking_date_str, '%Y-%m-%d')
    departure_date = datetime.strptime(departure_date_str, '%Y-%m-%d')

    # Calculate the difference between the two dates
    date_diff = departure_date - booking_date
    days_difference = date_diff.days

    # Parse date information
    p_day_diff, p_day_of_week, p_day_of_month, p_month, p_year = extract_date_info(departure_date_str)

    input_data = [p_airline, p_duration, p_from_place, p_to_place, p_stop, days_difference, p_day_of_week, p_day_of_month, p_month]

    print(input_data)
    try:
        p_prediction = predict_price(input_data, model, numerical_scaler, target_scaler, label_encoders)
    except Exception as e:
        print(f"An error occurred while making a prediction: {e}")
        p_prediction = None

    print(p_prediction)

    return render_template('Home.html', p_prediction=p_prediction)


def get_page(playwright, from_place, to_place, departure_date, class_s):
    page = playwright.chromium.launch(headless=True).new_page()
    page.goto('https://www.google.com/travel/flights')

    # one way selection
    page.query_selector('.VfPpkd-TkwUic') and page.query_selector('.VfPpkd-TkwUic').click()

    time.sleep(.25)
    page.query_selector('.VfPpkd-rymPhb') and page.query_selector('.VfPpkd-rymPhb').click()

    # class selection
    page.query_selector('.KbCVkc .VfPpkd-TkwUic') and page.query_selector('.KbCVkc .VfPpkd-TkwUic').click()

    time.sleep(.25)
    if class_s == 0:
        # Economy
        Economy_xpath = "//div[@class='TQYpgc']/div/div/div[2]/ul/li[1]"
        Economy = page.query_selector('xpath=' + Economy_xpath) and page.query_selector(
            'xpath=' + Economy_xpath).click()

    elif class_s == 1:
        # Premium Economy
        Premium_Economy_xpath = "//div[@class='TQYpgc']/div/div/div[2]/ul/li[2]"  # Premium Economy
        Premium_Economy = page.query_selector('xpath=' + Premium_Economy_xpath) and page.query_selector(
            'xpath=' + Premium_Economy_xpath).click()

    elif class_s == 2:
        # Business Class
        Business_xpath = "//div[@class='TQYpgc']/div/div/div[2]/ul/li[3]"  # Business
        Business = page.query_selector('xpath=' + Business_xpath) and page.query_selector(
            'xpath=' + Business_xpath).click()

    elif class_s == 3:
        # First Class
        first_xpath = "//div[@class='TQYpgc']/div/div/div[2]/ul/li[4]"
        first = page.query_selector('xpath=' + first_xpath) and page.query_selector('xpath=' + first_xpath).click()

    # type "From"
    from_place_field = page.query_selector_all('.e5F5td')[0]
    from_place_field.click()
    time.sleep(.35)
    from_place_field.type(from_place)
    # time.sleep(1)
    page.keyboard.press('Enter')

    # type "To"
    to_place_field = page.query_selector_all('.e5F5td')[1]
    to_place_field.click()
    time.sleep(.35)
    to_place_field.type(to_place)
    time.sleep(.2)
    page.keyboard.press('Enter')

    # Type "Departure date"
    departure_date_f = page.query_selector('[aria-label="Departure"]')
    departure_date_f.click()
    time.sleep(.45)

    departure_date_field = page.query_selector('.X4feqd input[aria-label="Departure"]')
    departure_date_field.click()
    time.sleep(.45)
    departure_date_field.fill(departure_date)
    time.sleep(.25)
    departure_date_field.click()
    time.sleep(.45)
    departure_date_field.fill(departure_date)
    time.sleep(.45)
    page.keyboard.press('Enter')
    page.keyboard.press('Enter')

    # press "Explore"
    page.query_selector('.MXvFbd .VfPpkd-LgbsSe').click()
    time.sleep(2)

    # press "More flights"
    more = page.query_selector('.zISZ5c button')
    if more:
        more.click()
    time.sleep(3)

    parser = LexborHTMLParser(page.content())
    page.close()

    return parser


def scrape_google_flights(parser, from_place, to_place, departure_date):
    data = {}

    categories = parser.root.css('.zBTtmb')
    category_results = parser.root.css('.Rk10dc')

    # Parse date information
    day_diff, day_of_week, day_of_month, month, year = extract_date_info(departure_date)



    for category, category_result in zip(categories, category_results):
        category_data = []

        for result in category_result.css('.yR1fYc'):
            try:
                date = result.css('[jscontroller="cNtv4b"] span') if result.css(
                    '[jscontroller="cNtv4b"] span') else None
                departure_time = date[0].text()
                arrival_time = date[1].text()
                company = result.css_first(' .Ir0Voe .sSHqwe').text() if result.css_first(' .Ir0Voe .sSHqwe') else None
                duration = result.css_first(' .AdWm1c.gvkrdb').text() if result.css_first(' .AdWm1c.gvkrdb') else None
                stops = result.css_first(' .EfT7Ae .ogfYpf').text() if result.css_first(' .EfT7Ae .ogfYpf') else None
                emissions = result.css_first(' .V1iAHe .AdWm1c').text() if result.css_first(
                    ' .V1iAHe .AdWm1c') else None
                emission_comparison = result.css_first(' .N6PNV').text() if result.css_first('.N6PNV') else None
                price = result.css_first(' .U3gSDe .FpEdX span').text() if result.css_first(
                    '.U3gSDe .FpEdX span') else None
                # price_type = result.css_first(' .U3gSDe .N872Rd').text() if result.css_first('.U3gSDe .N872Rd') else None
                price_type = "One Way"

                converted_price = convert_price_to_int(price)
                if converted_price is not None:  # Only insert if converted price is not None

                    n_airline = standardize_airline_name(company)
                    n_duration = convert_duration_to_minutes(duration)
                    n_stops = convert_stops_to_int(stops)
                    n_source = from_place
                    n_destination = to_place
                    n_day_diff = day_diff
                    n_day_of_week = day_of_week
                    n_day_of_month = day_of_month
                    n_month = month

                    input_data = [n_airline, n_duration, n_source, n_destination, n_stops, n_day_diff, n_day_of_week, n_day_of_month, n_month]
                    try:
                        prediction = predict_price(input_data, model, numerical_scaler, target_scaler, label_encoders)
                    except Exception as e:
                        print(f"An error occurred while making a prediction: {e}")
                        prediction = None
                else:
                    prediction = None
                flight_data = {
                    'departure_time': departure_time,
                    'arrival_time': arrival_time,
                    'company': standardize_airline_name(company),
                    'duration': duration,
                    'stops': stops,
                    'emissions': emissions,
                    'emission_comparison': emission_comparison,
                    'price': price,
                    'actual_price': converted_price,
                    'price_type': price_type,
                    'prediction': prediction
                }

                airports = result.css_first('.Ak5kof .sSHqwe')
                service = result.css_first('.hRBhge')

                if service:
                    flight_data['service'] = service.text()

                else:
                    flight_data['departure_airport'] = airports.first_child.css_first(
                        'span:nth-child(1) .eoY5cb').text()
                    flight_data['arrival_airport'] = airports.last_child.css_first('span:nth-child(1) .eoY5cb').text()
                category_data.append(flight_data)
            except Exception as e:
                print(f"An error occurred while parsing a flight: {e}")

        data[category.text().lower().replace(' ', '_')] = category_data

    return data


def daily_scrape_google_flights(parser, from_place, to_place, departure_date):
    data = []

    categories = parser.root.css('.zBTtmb')
    category_results = parser.root.css('.Rk10dc')

    # Parse date information
    day_diff, day_of_week, day_of_month, month, year = extract_date_info(departure_date)

    for category, category_result in zip(categories, category_results):
        for result in category_result.css('.yR1fYc'):
            try:
                company = result.css_first(' .Ir0Voe .sSHqwe').text() if result.css_first(' .Ir0Voe .sSHqwe') else None
                duration = result.css_first(' .AdWm1c.gvkrdb').text() if result.css_first(' .AdWm1c.gvkrdb') else None
                stops = result.css_first(' .EfT7Ae .ogfYpf').text() if result.css_first(' .EfT7Ae .ogfYpf') else None
                price = result.css_first(' .U3gSDe .FpEdX span').text() if result.css_first(
                    '.U3gSDe .FpEdX span') else None

                converted_price = convert_price_to_int(price)
                if converted_price is not None:  # Only insert if converted price is not None
                    flight_data = {
                        'airline': standardize_airline_name(company),
                        'duration': convert_duration_to_minutes(duration),
                        'stops': convert_stops_to_int(stops),
                        'price': converted_price,
                        # Add other fields
                        'source': from_place,
                        'destination': to_place,
                        'day_diff': day_diff,
                        'day_of_week': day_of_week,
                        'day_of_month': day_of_month,
                        'month': month,
                        'year': year
                    }

                    # Append data to the list
                    data.append(flight_data)
            except Exception as e:
                print(f"An error occurred while parsing a flight: {e}")

    # Write data to CSV file
    write_to_csv(data)

    # Insert data into MongoDB
    for flight_data in data:
        collection.insert_one(flight_data)

    return data


def write_to_csv(data):
    fields = ['airline', 'duration', 'source', 'destination', 'stops', 'day_diff', 'day_of_week', 'day_of_month',
              'month', 'price']

    # Get today's date in 'DD-MM-YYYY' format
    today_csv = datetime.now().strftime('%d-%m-%Y')

    # Include today's date in the filename
    filename = f'Dataset/flight_data_{today_csv}.csv'

    # Create 'Dataset' directory if it doesn't exist
    if not os.path.exists('Dataset'):
        os.makedirs('Dataset')

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # Write header only if file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        for flight in data:
            # Write only the fields in 'fields' to the CSV
            writer.writerow({field: flight[field] for field in fields})


# preprocessing Functions


def standardize_airline_name(airline):
    # Mapping of airline names to their standard short names
    airline_mapping = {
        "Air Arabia": "Air Arabia",
        "AirAsia": "AirAsia",
        "Air India": "Air India",
        "Air India Express": "Air India Express",
        "Air New Zealand": "Air New Zealand",
        "Akasa Air": "Akasa Air",
        "Asiana": "Asiana",
        "Alliance Air": "Alliance Air",
        "Batik Air": "Batik Air",
        "EgyptAir": "EgyptAir",
        "Emirates": "Emirates",
        "Etihad": "Etihad",
        "flydubai": "flydubai",
        "Gulf Air": "Gulf Air",
        "IndiGo": "IndiGo",
        "Jazeera": "Jazeera",
        "Kuwait Airways": "Kuwait Airways",
        "Lufthansa": "Lufthansa",
        "Maldivian": "Maldivian",
        "Malaysia Airlines": "Malaysia Airlines",
        "Oman Air": "Oman Air",
        "Qatar Airways": "Qatar Airways",
        "Saudia": "Saudia",
        "Singapore Airlines": "Singapore Airlines",
        "SpiceJet": "SpiceJet",
        "SriLankan": "SriLankan",
        "THAI": "THAI",
        "Thai AirAsia": "Thai AirAsia",
        "Vistara": "Vistara"
    }

    # Iterate through the mapping and check if any key (airline name) is present in the given airline string
    for key in airline_mapping.keys():
        if key in airline:
            return airline_mapping[key]

    # Return None if the airline is not found in the mapping
    return airline


def convert_duration_to_minutes(duration):
    parts = duration.split()
    minutes = 0
    for i in range(0, len(parts), 2):
        if 'hr' in parts[i + 1]:
            minutes += int(parts[i]) * 60
        elif 'min' in parts[i + 1]:
            minutes += int(parts[i])
    return minutes


def convert_stops_to_int(stops):
    if stops == "Nonstop":
        return 0
    else:
        return int(stops[0])


def convert_price_to_int(price):
    if price == "Price unavailable":
        return None
    else:
        return int(price.replace('₹', '').replace(',', ''))


def extract_date_info(date_str):
    # Parse the date string
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Calculate the difference in days from today
    day_diff = (date - datetime.now()).days + 1

    # Get the day of the week (0=Monday, 6=Sunday)
    day_of_week = date.weekday()

    # Get the day of the month
    day_of_month = date.day

    # Get the month
    month = date.month

    # Get the year
    year = date.year

    return day_diff, day_of_week, day_of_month, month, year


if __name__ == '__main__':
    serve(app,port = 5000)


