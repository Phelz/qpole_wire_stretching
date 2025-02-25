import pathlib
import pandas as pd


DIR_PATH = pathlib.Path(__file__).parent

NUM_REPS = 10
PLATE_THICKNESS = 1.266206 / 1_000 # meters
# HEIGHT_ONE_REP = 0.00258985992366699003 # meters
# HEIGHT_ONE_REP = (2.66220000 + 0.02 )/ 1_000  # Plates + 0.02m gap connector 
HEIGHT_ONE_REP = (2.65559999993917  + 0.12700599999999995 )/ 1_000  # Plates + 0.02m gap connector 

# HEIGHT_BOT_PART = (1.2700100 + 0.02) / 1_000 # meters
HEIGHT_BOT_PART = (1.2700100 + 0.12700599999999995) / 1_000 # meters

TOTAL_GEOMETRY_HEIGHT = 30.55324399999871 / 1_000 # meters 

endBoxCoords = pd.read_csv(DIR_PATH / 'ending_box_coords.txt')
startBoxCoords = pd.read_csv(DIR_PATH / 'starting_box_coords.txt')

YMIN = endBoxCoords['y'].min()
YMAX = startBoxCoords['y'].max()

XMIN = endBoxCoords['x'].min()
XMAX = startBoxCoords['x'].max()

ZMIN = endBoxCoords['z'].min()
ZMAX = startBoxCoords['z'].max()


constantsDF = pd.read_csv(DIR_PATH / 'constants_test21.csv')
ZMIDPOINT = constantsDF["z_midpoint"].values[0]

