import os
import csv
from typing import Optional
from datetime import datetime


def log_iteration(iteration, initial_condition, querytime, function_value, iter_time, regret, log_file_path="optimization_log.csv", auxilary_info: Optional[dict] = {}):
    """Log the details of the current iteration, including initial log file setup."""
    
    # Check if the log file exists, create it with an empty list if it doesn't
    dir_name = os.path.dirname(log_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(log_file_path):
        # If not, create it and write the header
        with open(log_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for key, value in auxilary_info.items():
                writer.writerow([key, value])
            writer.writerow(["----------------------------------------------------------------------"])
            writer.writerow(["time", "iter", "initial_cond", "queried_time", "function_value", "opt_iter_time", "regret"])

    # Create the log entry
    log_entry = [
        datetime.now().isoformat(),
        iteration,
        initial_condition.tolist(), 
        querytime,
        function_value.tolist(),
        float(iter_time),
        float(regret)
    ]

    # Append the new log entry to the log file
    with open(log_file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(log_entry)