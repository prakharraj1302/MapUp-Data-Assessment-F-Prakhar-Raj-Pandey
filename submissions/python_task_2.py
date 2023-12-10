import pandas as pd
# additional imports
import numpy as np
import datetime

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    unique_ids = sorted(set(df['id_start']).union(set(df['id_end'])))
    id_to_index = {id_: index for index, id_ in enumerate(unique_ids)}

    # Initialize the distance matrix with infinities
    num_ids = len(unique_ids)
    distance_matrix = np.full((num_ids, num_ids), np.inf)

    # Fill in the known distances and set the diagonal to 0
    for _, row in df.iterrows():
        i, j = id_to_index[row['id_start']], id_to_index[row['id_end']]
        distance_matrix[i, j] = distance_matrix[j, i] = row['distance']
    np.fill_diagonal(distance_matrix, 0)

    # Floyd-Warshall algorithm
    for k in range(num_ids):
        for i in range(num_ids):
            for j in range(num_ids):
                if distance_matrix[i, k] + distance_matrix[k, j] < distance_matrix[i, j]:
                    distance_matrix[i, j] = distance_matrix[i, k] + distance_matrix[k, j]

    # Convert the matrix back to a DataFrame
    distance_matrix_df = pd.DataFrame(distance_matrix, index=unique_ids, columns=unique_ids)

    return distance_matrix_df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    data = []

    # Iterate over each row and column in the distance matrix
    for i, row_id in enumerate(df.index):
        for j, col_id in enumerate(df.columns):
            # Skip the diagonal elements (where start and end IDs are the same)
            if i != j:
                data.append([row_id, col_id, df.iloc[i, j]])
    
    # Create a DataFrame from the list
    unrolled_df = pd.DataFrame(data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    ref_df = df[df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    avg_distance_ref = ref_df['distance'].mean()

    # Calculate the 10% threshold
    lower_threshold = avg_distance_ref * 0.9
    upper_threshold = avg_distance_ref * 1.1

    # Prepare a list to store the results
    result = []

    # Iterate over each id_start in the DataFrame
    for id_start in df['id_start'].unique():
        if id_start != reference_id:
            # Calculate the average distance for the current id_start
            avg_distance_current = df[df['id_start'] == id_start]['distance'].mean()

            # Check if the average distance is within the threshold
            if lower_threshold <= avg_distance_current <= upper_threshold:
                result.append({'id_start': id_start, 'average_distance': avg_distance_current})

    # Create a DataFrame from the result list
    result_df = pd.DataFrame(result)

    return result_df



def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    # df = df[['id_start','id_end','moto','car','rv','bus','truck']]

    return df


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    time_ranges = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),  # Early hours
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)), # Day hours
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59)) # Evening hours
    ]

    # Days of the week
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a new DataFrame to store results
    results = []

    # Adjusting the logic to create a 24-hour span for each day of the week
    for i, (start_day, end_day) in enumerate(zip(days, days[1:] + days[:1])):
        for start_time, end_time in time_ranges:
            # Apply discount factor based on time and day
            if start_day in ['Saturday', 'Sunday']:
                discount_factor = 0.7
            elif start_time in [time_ranges[0][0], time_ranges[2][0]]:
                discount_factor = 0.8
            else:
                discount_factor = 1.2
            
            for _, row in df.iterrows():
                new_row = {
                    "id_start": row['id_start'],
                    "id_end": row['id_end'],
                    "distance": row['distance'],
                    "start_day": start_day,
                    "start_time": start_time,
                    "end_day": end_day if start_time != time_ranges[2][0] else start_day,
                    "end_time": end_time
                }
                # Apply the discount factor to vehicle rates
                for vehicle in ["moto", "car", "rv", "bus", "truck"]:
                    new_row[vehicle] = round(row[vehicle] * discount_factor, 2)
                
                results.append(new_row)

    return pd.DataFrame(results)
