import pandas as pd


# Save the cleaned data to a new CSV file
df = pd.read_csv('skin-care-recommender/ml_model/cleaned_result.csv')
df.to_csv('skin-care-recommender/ml_model/to_be_use_dataset.csv', index=False)
data = df

# Extract unique values from 'skin type' and 'concern'
# unique_skin_types = data_cleaned['skin type'].unique().tolist()
# unique_concerns = data['concern'].unique().tolist()
unique_concerns_1 = data['concern'].dropna().unique()
unique_concerns_2 = data['concern 2'].dropna().unique()
unique_concerns_3 = data['concern 3'].dropna().unique()

# Combine all unique concerns into one set to remove duplicates
all_unique_concerns = set(unique_concerns_1) | set(unique_concerns_2) | set(unique_concerns_3)

# Convert to a list if needed
all_unique_concerns_list = list(all_unique_concerns)


# Print or save the combined unique concerns

# Print the unique skin types and concerns
# print("Unique Skin Types:", unique_skin_types)
# print("Unique Concerns:", unique_concerns)
# print("All unique concerns:", all_unique_concerns_list)
z = 0
for i in all_unique_concerns_list:
    print(i + " = " + str(z))
    z += 1