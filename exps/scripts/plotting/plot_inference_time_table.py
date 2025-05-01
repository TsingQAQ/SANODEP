import re
import numpy as np

def calculate_average_and_std_time_per_iteration(filename):
    times = []
    with open(filename, 'r') as file:
        for line in file:
            match_s_it = re.search(r'(\d+\.\d+)s/it', line)
            match_it_s = re.search(r'(\d+\.\d+)it/s', line)
            if match_s_it:
                times.append(float(match_s_it.group(1)))
            elif match_it_s:
                times.append(1 / float(match_it_s.group(1)))

    if times:
        return np.mean(times), np.std(times)
    else:
        return None, None

filename = 'slurm20-115351.out'
average_time_per_iteration, std_time_per_iteration = calculate_average_and_std_time_per_iteration(filename)

if average_time_per_iteration is not None:
    print(f'The average time per iteration is {average_time_per_iteration} seconds.')
    print(f'The standard deviation of time per iteration is {std_time_per_iteration} seconds.')
else:
    print('No s/it or it/s values found in the file.')