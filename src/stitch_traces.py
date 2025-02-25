import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev

from tqdm import tqdm
import multiprocessing

import data, figures


constantsDF = pd.read_csv(data.DIR_PATH / "constants_test21.csv")
ending_box_coords = pd.read_csv(data.DIR_PATH / "ending_box_coords.txt")
starting_box_coords = pd.read_csv(data.DIR_PATH / "starting_box_coords.txt")

YMIN = ending_box_coords['y'].min()
YMAX = starting_box_coords['y'].max()
ZMIDPOINT = constantsDF["z_midpoint"].values[0]


THRESHOLD_TOP = - data.PLATE_THICKNESS / 500
SIGMA_TOP = 4

THRESHOLD_BOT = - data.PLATE_THICKNESS / 500
SIGMA_BOT = 2
NUM_REPS = 10

NUM_INTERPOLATION_PTS = 50
NUM_DATA_PTS_TO_USE = 48

def get_data_around_edge(pTrace, cutoff=10, num_pts=1000, which="first"):
    
    pTrace_percent = 100 * pTrace['y'].diff().abs() / pTrace['y'].diff().abs().max()

    # Get a 1000 points around when 10 percent is exceeded for the first/last time
    pTrace_percent_10 = pTrace_percent[pTrace_percent > cutoff]
    if which == "last":
        pTrace_percent_10_index = pTrace_percent_10.index[-1]
    else:
        pTrace_percent_10_index = pTrace_percent_10.index[0]

    return pTrace.iloc[pTrace_percent_10_index - num_pts//2 : pTrace_percent_10_index + num_pts//2]

def find_dissection_pt(pTrace_around, sigma, threshold, particle_path=None):
    
    # Convert index and y-values to NumPy arrays
    x = pTrace_around.index.to_numpy()
    y = pTrace_around["y"].to_numpy()

    # Ensure x is strictly increasing
    if np.any(np.diff(x) <= 0):
        raise ValueError("x values must be strictly increasing.")

    # Smooth data to reduce noise
    y_smooth = gaussian_filter1d(y, sigma=sigma)

    # Compute first derivative
    try:
        dy = np.gradient(y_smooth, x)
    except IndexError:
        print(f"Could not compute gradient for {particle_path}.")
        print(f"y_smooth: {y_smooth}")
        print(f"x: {x}")
        return
    
    # Find the first index where dy drops below a threshold
    try:
        drop_start_idx = np.where(dy < threshold)[0][0]  # First significant drop
        
    except IndexError:
        print(f"Could not find a drop below {threshold} for {particle_path}.")
        return
    
    # I'd like this point from the dataframe
    drop_start_point = pTrace_around.iloc[drop_start_idx]

    return drop_start_point

def check_number_in_range(number, a, b):
    minimum = min(a, b)
    maximum = max(a, b)
    return minimum <= number <= maximum



def get_zplane_crossing_index_recursive(particle_trace, zval, ymin, ymax, which='first'):
    tol = constantsDF["z_range"].values[0] / 20
    
    trace_copy = particle_trace.copy()
    trace_copy['crossed'] = trace_copy.apply(
        lambda row: check_number_in_range(row['z'], zval - tol, zval + tol) and check_number_in_range(row['y'], ymin, ymax), 
        axis=1
    )
    
    index_crossed = trace_copy[trace_copy['crossed']].index

    if not index_crossed.empty:
        return index_crossed[0] if which == 'first' else index_crossed[-1]
    
    # Recursive case: Reduce ymin and try again
    new_ymin = ymin - data.PLATE_THICKNESS / 8
    return get_zplane_crossing_index_recursive(particle_trace, zval, new_ymin, ymax, which)

def get_multiple_zplane_crossings(particle_trace, zval, ymin, ymax, num_crossings, which='first'):
    crossings = []
    
    for _ in range(num_crossings):
        # Find crossing index
        crossing_index = get_zplane_crossing_index_recursive(particle_trace, zval, ymin, ymax, which)
        crossings.append(crossing_index)
        
        # Update ymin and ymax for the next crossing
        new_ymax = particle_trace.iloc[crossing_index]['y'] - data.PLATE_THICKNESS / 16
        ymin = new_ymax - data.PLATE_THICKNESS / 16
        ymax = new_ymax
    
    return crossings



def interpolate_middle_part(newTopPiece, newBottomPiece, num_interpolation_pts=50, num_data_pts_to_use=100, top_fraction=3/4, bot_fraction=1/4):
    
    # * Divide the top and bottom pieces into head and tail
    topPieceData = newTopPiece.tail(num_data_pts_to_use//2)
    botPieceData = newBottomPiece.head(num_data_pts_to_use//2)
    
    # * Take only a portion of the head and tail
    tailPortion = topPieceData.head(int(top_fraction*num_data_pts_to_use))
    headPortion = botPieceData.tail(int(bot_fraction*num_data_pts_to_use))
    
    dfToInterp = pd.concat([tailPortion, headPortion])
    
    x = dfToInterp['x'].values
    y = dfToInterp['y'].values
    z = dfToInterp['z'].values

    # * Perform the spline interpolation
    tck, u = splprep([x, y, z], s=0)
    new_points = splev(np.linspace(0, 1, num_interpolation_pts), tck)

    # Create a DataFrame for the interpolated points
    interpDF = pd.DataFrame({
        'x': new_points[0],
        'y': new_points[1],
        'z': new_points[2]
    })

    newTopPieceTrimmed = newTopPiece.iloc[:-num_data_pts_to_use//2 + 1]
    newBottomPieceTrimmed = newBottomPiece.iloc[num_data_pts_to_use//2 - 1:]
    interpPiece = pd.concat([newTopPieceTrimmed, interpDF, newBottomPieceTrimmed])
    
    return interpPiece

def stitch_trace(pTrace):
    
    # Skip the first 100 pts to account for those tracks that start at the far edge
    pTraceTrim = pTrace.iloc[100:]

    crossings = get_multiple_zplane_crossings(pTraceTrim, ZMIDPOINT, YMAX - data.PLATE_THICKNESS / 16, YMAX, num_crossings=6)

    crossing_indices = { i+1: int(crossings[i]) for i in range(len(crossings)) }

    pTraceTrim_around_top = pTraceTrim.iloc[crossing_indices[1]:crossing_indices[2]]
    pTraceTrim_around_bot = pTraceTrim.iloc[crossing_indices[5]:crossing_indices[6]]

    # drop_start_point_top = find_dissection_pt(pTraceTrim_around_top, sigma=SIGMA_TOP, threshold=THRESHOLD_TOP)
    drop_start_point_top = pTrace[pTrace['y'] < YMAX - data.PLATE_THICKNESS].iloc[0]
    # drop_start_point_bot = pTrace[pTrace['y'] < YMAX - data.PLATE_THICKNESS - data.HEIGHT_ONE_REP].iloc[0]
    drop_start_point_bot = find_dissection_pt(pTraceTrim_around_bot, sigma=SIGMA_BOT, threshold=THRESHOLD_BOT)
    
    drop_start_index_top = pTrace[pTrace['y'] == drop_start_point_top['y']].index[0]
    drop_start_index_bot = pTrace[pTrace['y'] == drop_start_point_bot['y']].index[0]
    
    
    
    # # * After finding the drop_start_index_top, there's one minor adjustment to be made. For the height difference between the starting point and the edge of the first plate
    # yi = pTrace['y'].max()
    # delta_h = YMAX - yi
    
    
    # # Get the starting y-value
    # y_start = pTrace.iloc[drop_start_index_top]['y']
    # target_y = y_start - delta_h  # The target y-value

    # # Find the first index where y <= target_y after drop_start_index_top
    # next_index = (pTrace.iloc[drop_start_index_top+1:]
    #             .index[pTrace.iloc[drop_start_index_top+1:]['y'] <= target_y]
    #             .min())

    # # Get the corresponding point
    # drop_next_pt = pTrace.loc[next_index] if not np.isnan(next_index) else None
    # # Get the index of that new point
    # drop_start_index_top = next_index
    
    topPart = pTrace.iloc[:drop_start_index_top]
    midPart = pTrace.iloc[drop_start_index_top:drop_start_index_bot]
    botPart = pTrace.iloc[drop_start_index_bot:]

    # Create a dictionary of copies
    midPartCopies = {i: midPart.copy() for i in range(0, NUM_REPS )}
    
    # Calculate the height
    max_y = midPartCopies[0]['y'].max()
    min_y = midPartCopies[0]['y'].min()
    height = max_y - min_y

    # Shift each copy by i * height
    for i in range(1, NUM_REPS):
        midPartCopies[i]['y'] -= (i) * height
        
    last_y = midPartCopies[2].tail(1)['y'].values[0]
    first_y = midPartCopies[3].head(1)['y'].values[0]
    y_diff = abs(first_y - last_y)

    # # Shift each copy by i * height
    for i in range(1, NUM_REPS):
        midPartCopies[i]['y'] += (i) * y_diff

    botPartShifted = botPart.copy()
    botPartShifted['y'] -= (NUM_REPS - 1) * (height-y_diff)

    # * Interpolate the missing parts
    interpPiece = interpolate_middle_part(midPartCopies[0], midPartCopies[1], num_interpolation_pts=NUM_INTERPOLATION_PTS, num_data_pts_to_use=NUM_DATA_PTS_TO_USE)

    # Now let's repeat that for all the copies
    for i in range(2, NUM_REPS):
        interpPiece = interpolate_middle_part(interpPiece, midPartCopies[i], num_interpolation_pts=NUM_INTERPOLATION_PTS, num_data_pts_to_use=NUM_DATA_PTS_TO_USE)

    # The final DF is just the top + interpolated + bottom
    finalDF = pd.concat([topPart, interpPiece, botPartShifted])
    
    return finalDF, topPart, midPartCopies, botPartShifted

def stitch_trace_wrapper(file_path):
    pTrace = pd.read_csv(file_path)
    try:
        finalDF, topPart, midPartCopies, botPartShifted = stitch_trace(pTrace)
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return
    
    finalDF.to_csv(data.DIR_PATH / "particle_data_stitched" / file_path.name, index=False)
    
    # Save the figure
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot top part
    ax.scatter(topPart['x'], topPart['y'], topPart['z'], c='black', s=0.5, label='Top Part')

    # Plot mid part copies
    for i in range(0, NUM_REPS):
        ax.plot(midPartCopies[i]['x'], midPartCopies[i]['y'], midPartCopies[i]['z'], color='red', lw=1, alpha=0.95 )

    # Plot bottom part
    ax.scatter(botPartShifted['x'], botPartShifted['y'], botPartShifted['z'], c='green', s=0.05, label='Bot Part')

    # Plot final interpolated part
    ax.scatter(finalDF['x'], finalDF['y'], finalDF['z'], c='C0', s=0.5, label='Final DF', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend()
    ax.grid(False)
    # init view
    ax.view_init(elev=10, azim=20)


    # Zoom in
    ax.set_xlim(constantsDF['min_x'].values[0], constantsDF['max_x'].values[0] - constantsDF['x_range'].values[0] / 2)
    ax.set_ylim(YMAX - 6*data.PLATE_THICKNESS, YMAX- 2*data.PLATE_THICKNESS)
    ax.set_zlim(constantsDF['min_z'].values[0], constantsDF['max_z'].values[0])
    
    plt.savefig(figures.DIR_PATH / "check_stitches" / file_path.name.replace(".csv", ".png"))
    plt.close()
    
# Let's use multiprocessing to speed things up

def main():
    files = list(data.DIR_PATH.glob("particle_data_trimmed/*.csv"))
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm(pool.imap(stitch_trace_wrapper, files), total=len(files)))
        
if __name__ == "__main__":
    main()
             