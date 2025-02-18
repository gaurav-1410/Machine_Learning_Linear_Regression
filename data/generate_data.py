import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
num_samples = 100  

# Generate synthetic features
size_sqft = np.random.randint(800, 3500, num_samples)  # House size in square feet
bedrooms = np.random.randint(1, 5, num_samples)  # Number of bedrooms
bathrooms = np.random.randint(1, 4, num_samples)  # Number of bathrooms
distance_from_city = np.random.uniform(1, 20, num_samples)  # Distance from city center in km
house_age = np.random.randint(0, 50, num_samples)  # House age in years

# Generate target variable (House price in lakhs)
# Assuming house price depends on all these factors
price_lakh = (
    (size_sqft * 0.5) + 
    (bedrooms * 10) + 
    (bathrooms * 8) - 
    (distance_from_city * 2) - 
    (house_age * 1.5) +
    np.random.normal(0, 20, num_samples)  # Adding some noise
)

# Create DataFrame
df = pd.DataFrame({
    'Size_sqft': size_sqft,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Distance_from_city_km': distance_from_city,
    'House_Age_years': house_age,
    'Price_lakh': price_lakh
})

# Save to CSV
df.to_csv("data/dataset.csv", index=False)

print("Dataset generated successfully and saved as 'data/dataset.csv'.")
