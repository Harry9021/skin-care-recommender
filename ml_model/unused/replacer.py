import pandas as pd

# Load the CSV file
df = pd.read_csv('skin-care-recommender/ml_model/cleaned_result.csv')

# Create a dictionary for concern mappings
concern_mapping = {
    "Anti-Pollution": 0,
    "Tan Removal": 1,
    "Dryness": 2,
    "Deep Nourishment": 3,
    "Blackheads and Whiteheads": 4,
    "Oil Control": 5,
    "Fine Lines and Wrinkles": 6,
    "Uneven Skin Tone": 7,
    "Dark Spots": 8,
    "Dark Circles": 9,
    "Skin Tightening": 10,
    "Under Eye Concern": 11,
    "Skin Inflammation": 12,
    "General Care": 13,
    "Redness": 14,
    "Skin Sagging": 15,
    "Lightening": 16,
    "Sun Protection": 17,
    "Pigmentation": 18,
    "Blackheads Removal": 19,
    "Oily Skin": 20,
    "Anti-Ageing": 21,
    "Hydration": 22,
    "Dull Skin": 23,
    "Uneven Texture": 24,
    "Irregular Textures": 25,
    "Pore Minimizing and Blurring": 26,
    "Excess Oil": 27,
    "Daily Use": 28,
    "Dullness": 29,
    "Anti Acne Scarring": 30,
    "Softening and Smoothening": 31,
    "Acne or Blemishes": 32,
    "Pore Care": 33
}

# Create a dictionary for skin type mappings
skin_type_mapping = {
    "All": 0,
    "Normal": 1,
    "Dry": 2,
    "Oily": 3,
    "Combination": 4,
    "Sensitive": 5
}

# Replace the concerns in each concern column
df['concern'] = df['concern'].map(concern_mapping)
df['concern 2'] = df['concern 2'].map(concern_mapping)
df['concern 3'] = df['concern 3'].map(concern_mapping)

# Replace the skin types in the 'skin type' column
df['skin type'] = df['skin type'].map(skin_type_mapping)

# Check the result (this will show the first few rows of your DataFrame with replaced concerns and skin types)
print(df.head())

# Optionally, save the DataFrame to a new CSV file
df.to_csv('skin-care-recommender/ml_model/to_be_use_dataset.csv', index=False)
