import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib

import plotly.graph_objects as go
from multiprocessing import Pool

# Constants
constants_df = pd.read_csv('constants_test21.csv')

YMIN      = constants_df["min_y"].values[0]
YMAX      = constants_df["max_y"].values[0]
YRANGE    = constants_df["y_range"].values[0]
ZMIDPOINT = constants_df["z_midpoint"].values[0]

NUM_REPS = 10

# Y Planes of intersection
Y_BOT = 0.0710

YMIN_TOP = 0.073599 # SOMEWHAT BELOW THIS
YMAX_TOP = 0.0743 # SOMEWHAT ABOVE THIS

YTOL_TOP = YRANGE/32
YTOL_BOT = YRANGE/16

YTOP_MIN = YMIN_TOP - YTOL_TOP
YTOP_MAX = YMAX_TOP + YTOL_TOP
YBOT_MIN = Y_BOT - YTOL_BOT
YBOT_MAX = Y_BOT + YTOL_BOT

def check_number_in_range(number, a, b):
    minimum = min(a, b)
    maximum = max(a, b)
    return minimum <= number <= maximum

def get_zplane_crossing_index(particle_trace, zval, ymin, ymax):
    trace_copy = particle_trace.copy()
    trace_copy['crossed'] = trace_copy.apply(lambda row: abs(row['z']) > zval and check_number_in_range(row['y'], YMIN, YMAX), axis=1)
    # print(trace_copy)
    
    index_crossed = trace_copy[trace_copy['crossed']].index[0]
    return index_crossed

def dissect_trace(particle_trace, index_crossed_top, index_crossed_bot):
    
    trace_copy = particle_trace.copy()
    
    before_top_crossing = trace_copy.iloc[:index_crossed_top]
    after_top_crossing  = trace_copy.iloc[index_crossed_top:]
    middle_part         = trace_copy.iloc[index_crossed_top:index_crossed_bot]
    after_bot_crossing  = trace_copy.iloc[index_crossed_bot:]
    before_bot_crossing = trace_copy.iloc[index_crossed_bot:]
    
    return before_top_crossing, after_top_crossing, middle_part, after_bot_crossing, before_bot_crossing

# Now let's attempt stretching 
def stretch_trace(particle_trace, num_reps, index_crossed_top, index_crossed_bot):
    before_top_crossing, after_top_crossing, middle_part, after_bot_crossing, before_bot_crossing = dissect_trace(particle_trace, index_crossed_top, index_crossed_bot)
    
    # Calculate the height of the repeating part and adjust the y values of the after_bot_crossing
    height_repeating_part = max(middle_part['y']) - min(middle_part['y'])
    after_bot_crossing['y'] = after_bot_crossing['y'] - height_repeating_part*(num_reps-1)
    
    # Calculate the x-value difference between the first point after_top_crossing and the first point after_bot_crossing
    x_diff = after_bot_crossing['x'].iloc[0] - after_top_crossing['x'].iloc[0]
    after_bot_crossing['x'] = after_bot_crossing['x'] + x_diff*(num_reps-1)

    middle_parts_dfs = []
    for n in range(0, num_reps):
        middle_part_copy = middle_part.copy()
        middle_part_copy['y'] = middle_part_copy['y'] - height_repeating_part*n
        middle_part_copy['x'] = middle_part_copy['x'] + x_diff*n
        middle_parts_dfs.append(middle_part_copy)

    middle_parts_concat = pd.concat(middle_parts_dfs)

    extrapolated_particle_trace = pd.concat([before_top_crossing, middle_parts_concat, after_bot_crossing])
    extrapolated_particle_trace.reset_index(drop=True, inplace=True)
    
    tan_theta = x_diff*num_reps/(extrapolated_particle_trace['y'].max() - extrapolated_particle_trace['y'].min())
    theta_rad = np.radians(np.arctan(tan_theta))

    # Rotation matrix for x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_rad), -np.sin(theta_rad)],
        [0, np.sin(theta_rad), np.cos(theta_rad)]
    ])

    # Apply rotation
    rotated_points = extrapolated_particle_trace[['x', 'y', 'z']].values @ Rx.T  # Matrix multiplication
    
    rotated_trace = extrapolated_particle_trace.copy()
    rotated_trace[['x', 'y', 'z']] = rotated_points
    
    return rotated_trace     
    # return extrapolated_particle_trace


    
def process_particle_trace(particle_trace_path):
    print(f"Stretching {particle_trace_path}")
    try:
        particle_trace = pd.read_csv(particle_trace_path)

        index_crossed_top = get_zplane_crossing_index(particle_trace, ZMIDPOINT, YTOP_MIN, YTOP_MAX)
        index_crossed_bot = get_zplane_crossing_index(particle_trace, ZMIDPOINT, YBOT_MIN, YBOT_MAX)

        stretched_trace = stretch_trace(particle_trace, NUM_REPS, index_crossed_top, index_crossed_bot)

        new_path = pathlib.Path('stretched_traces') / (particle_trace_path.stem + '_stretched.csv')
        stretched_trace.to_csv(new_path, index=False)
        
    except Exception as e:
        print(f"Error processing {particle_trace_path}: {e}")



if __name__ == '__main__':
    
    # Get all particle traces
    particle_paths = [path for path in pathlib.Path('.').glob('particle_*.csv') if not str(path).endswith('frozen.csv')]

    # Create a pool of workers
    with Pool(28) as pool:
        pool.map(process_particle_trace, particle_paths[::10])
