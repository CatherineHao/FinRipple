import pandas as pd
import json


def update_json_with_csv(json_filename: str, csv_filename: str, output_filename: str):
    """
    Updates the JSON data with publication datetime and news body fields from the corresponding CSV file.

    Args:
        json_filename (str): The filename of the JSON file to be updated.
        csv_filename (str): The filename of the CSV file containing additional data.
        output_filename (str): The filename where the updated JSON data will be saved.

    Raises:
        ValueError: If the number of rows in the JSON and CSV files do not match.
    """
    # Load JSON data
    with open(json_filename, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    # Load CSV data
    csv_data = pd.read_csv(csv_filename)
    print(csv_data.columns)

    # Check if JSON and CSV have the same number of rows
    if len(json_data) != len(csv_data):
        raise ValueError("The number of rows in the JSON and CSV files do not match. Please check the data.")

    updated_json_data = []
    count = 0

    # Iterate through the JSON data and update
    for index, item in enumerate(json_data):
        try:
            # Try to parse the 'response' field in the JSON
            response_data = json.loads(item['response'])

            # Get 'publication_datetime' and 'body' fields from the CSV file for the current row
            publication_datetime = str(csv_data.iloc[index]['date'])  # Convert to string
            news_body = str(csv_data.iloc[index]['body'])  # Convert to string

            # Update the JSON item with the data from the CSV file
            item['date'] = publication_datetime
            item['news_body'] = news_body

            # Convert the updated response data back to a JSON string
            item['response'] = json.dumps(response_data, indent=4)
            updated_json_data.append(item)

        except json.JSONDecodeError as e:
            # Print error message and skip the current JSON item if parsing fails
            print(f"JSONDecodeError at index {index}: {e}")
            count += 1
            continue

    # Save the updated JSON data to a new file
    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(updated_json_data, json_file, indent=4)

    print(f"Total errors: {count}")


# Usage example:
data_str = '2021-12'
json_filename = f'{data_str}_results.json'
csv_filename = f'news/{data_str}.csv'
output_filename = f'{data_str}_updated.json'

update_json_with_csv(json_filename, csv_filename, output_filename)
