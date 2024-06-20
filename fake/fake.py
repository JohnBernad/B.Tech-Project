import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Define the parameters
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
num_rows = 50000

# Generate dates between start and end date
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Function to generate holiday variations
def generate_holiday_variation(date):
    holidays = {
        datetime(2024, 1, 1): 50,
        datetime(2024, 2, 14): 40,
        datetime(2024, 4, 1): 30,
        datetime(2024, 5, 27): 20,
        datetime(2024, 7, 4): 30,
        datetime(2024, 9, 2): 20,
        datetime(2024, 10, 31): 40,
        datetime(2024, 11, 28): 50,
        datetime(2024, 12, 25): 60
    }
    for holiday_date, adjustment in holidays.items():
        if date == holiday_date:
            return adjustment
    return 0

# Function to generate seasonal variation
def generate_seasonal_variation(month):
    seasonal_adjustments = {
        1: -10,
        2: -5,
        3: 0,
        4: 5,
        5: 10,
        6: 15,
        7: 20,
        8: 15,
        9: 10,
        10: 5,
        11: 0,
        12: -5
    }
    return seasonal_adjustments[month]

# Generate dataset
data = []
for _ in range(num_rows):
    date = random.choice(date_range)
    duration = random.randint(250, 450)
    source = random.choice(["New York", "Chicago", "Los Angeles", "San Francisco", "Miami"])
    destinations = ["New York", "Chicago", "Los Angeles", "San Francisco", "Miami"]
    destinations.remove(source)
    destination = random.choice(destinations)
    stops = random.randint(0, 2)
    day_diff = random.randint(10, 60)
    day_of_week = date.weekday()
    day_of_month = date.day
    month = date.month
    base_price = 300 + generate_seasonal_variation(month) + generate_holiday_variation(date)
    price = int(base_price + (random.random() * 100))
    airline = random.choice(["Delta", "United", "American Airlines"])
    data.append([airline, duration, source, destination, stops, day_diff, day_of_week, day_of_month, month, price])

# Create DataFrame
df = pd.DataFrame(data, columns=["airline", "duration", "source", "destination", "stops", "day_diff", "day_of_week", "day_of_month", "month", "price"])

# Write DataFrame to CSV
df.to_csv("airline_prices_with_variation.csv", index=False)
