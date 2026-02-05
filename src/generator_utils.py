import random
import numpy as np
from faker import Faker

fake = Faker()

def get_time_of_day():
    # Returns a random hour (0-23)
    return random.randint(0, 23)

def get_traffic_density(hour, road_type):
    # Rule: Rush hours (8-10 AM, 5-7 PM) have high traffic
    base_traffic = random.randint(20, 50)
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        base_traffic += 40
    # Rule: Highways have higher baseline traffic
    if road_type == 'Highway':
        base_traffic += 10
    return min(base_traffic, 100)  # Cap at 100

def get_weather():
    # Weighted probabilities: Clear is common, Snow is rare
    return np.random.choice(['Clear', 'Rain', 'Fog', 'Snow'], p=[0.6, 0.25, 0.1, 0.05])

def determine_actual_severity(hour, weather, speed, road_type):
    # This is the "Physics Engine" determining if an accident is Major (1) or Minor (0)
    risk_score = 0
    
    # Rule 1: Night Time (10 PM - 4 AM) is dangerous
    if hour >= 22 or hour <= 4:
        risk_score += 30
    
    # Rule 2: Bad Weather
    if weather in ['Rain', 'Fog', 'Snow']:
        risk_score += 25
    
    # Rule 3: High Speed
    if speed > 80:
        risk_score += 30
    
    # Rule 4: Highway speeds are fatal
    if road_type == 'Highway':
        risk_score += 10
    
    # Threshold: If risk > 70, it's a Major Accident
    # (Updated from 50 to 70 to make major accidents rarer/harder to predict)
    final_risk = risk_score + random.randint(-10, 10)
    return 1 if final_risk > 70 else 0

def generate_description(severity, weather, vehicle):
    # Generates a text description. 
    # If Severity is High (1), we use scary words like "Rollover", "Head-on".
    actions_minor = ["scratched bumper", "cracked taillight", "fender bender", "brushed side"]
    actions_major = ["rolled over", "head-on collision", "hit pedestrian", "caught fire", "totaled"]
    causes = ["slippery road", "brake failure", "distracted driving", "speeding"]
    
    if severity == 1:
        action = random.choice(actions_major)
    else:
        action = random.choice(actions_minor)
    
    cause = random.choice(causes)
    return f"{vehicle} {action} due to {cause} during {weather} conditions."
