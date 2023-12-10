import pandas as pd
# additional imports
from datetime import datetime, timedelta


def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Pivot 'df': rows='id_1', columns='id_2', values='car'.
    df = df.pivot(index='id_1', columns='id_2', values='car')

    # Replace NaNs in 'df' with 0.
    df = df.replace(float('nan'), 0)

    # Return modified DataFrame.
    return df

def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    def setType(val) -> str:
        # Categorize 'val' into 'low', 'medium', or 'high'.
        if(val <= 15):
            return 'low'
        elif(val > 15 and val <= 25):
            return 'medium'
        else:
            return 'high'

    # Apply 'setType' to 'car' column, creating 'car_type' column.
    df['car_type'] = df['car'].apply(setType)

    # Count occurrences of each 'car_type' and convert to dictionary.
    type_count = df['car_type'].value_counts().to_dict()

    # Sort 'type_count' dictionary by key.
    sorted_type_count = dict(sorted(type_count.items()))

    # Return sorted type count dictionary.
    return sorted_type_count



def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    mean_bus = df['bus'].mean()

    # Find indices where 'bus' values are greater than twice the mean
    indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    sorted_indexes = sorted(indexes)

    return sorted_indexes


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    avg_truck_per_route = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    filtered_routes = avg_truck_per_route[avg_truck_per_route > 7].index.tolist()

    # Sort the list of routes
    sorted_routes = sorted(filtered_routes)

    return sorted_routes


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    def tranformFx(val) -> int:
        # Adjust 'val': scale down if > 20, scale up if <= 20, and round to 1 decimal.
        if(val > 20):
            return round((val * 0.75), 1)
        elif(val <= 20):
            return round((val * 1.25), 1)

    # Apply 'tranformFx' to each element in 'matrix'.
    matrix = matrix.applymap(tranformFx)

    
    return matrix


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    weekday_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 
                   'Friday': 5, 'Saturday': 6, 'Sunday': 7}

    # Mapping startDay and endDay from string to numeric
    df['startDay'] = df['startDay'].map(weekday_map)
    df['endDay'] = df['endDay'].map(weekday_map)

    def check_time_coverage(group):
        # Initialize coverage for each day of the week
        day_coverage = {i: False for i in range(1, 8)}
        total_coverage = timedelta(0)

        for _, row in group.iterrows():
            start_day, end_day = row['startDay'], row['endDay']
            start_time = datetime.strptime(row['startTime'], '%H:%M:%S')
            end_time = datetime.strptime(row['endTime'], '%H:%M:%S')

            # Marking covered days and calculating total coverage
            for day in range(start_day, end_day + 1):
                day_coverage[day % 7 or 7] = True
                if start_day == end_day:
                    total_coverage += end_time - start_time
                else:
                    total_coverage += timedelta(hours=24) - (start_time - end_time)

        # Return True if coverage is complete
        return all(day_coverage.values()) and total_coverage >= timedelta(days=7)

    # Applying check for each (id, id_2) unique pair
    return df.groupby(['id', 'id_2']).apply(check_time_coverage)
