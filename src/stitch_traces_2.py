import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import splprep, splev

from tqdm import tqdm
import multiprocessing

import data, figures


def dissect_trace(trace, drop_start_index_top, drop_start_index_bot):
    topPart = trace.iloc[:drop_start_index_top]
    midPart = trace.iloc[drop_start_index_top:drop_start_index_bot]
    botPart = trace.iloc[drop_start_index_bot:]
    return topPart, midPart, botPart

def interpoolate_bet_parts(topPiece, botPiece, num_interpolation_pts=300, num_data_pts_to_use=30, top_fraction=1/2, bot_fraction=1/2):
    
    # * Divide the top and bottom pieces into head and tail
    topPieceData = topPiece.tail(int(top_fraction*num_data_pts_to_use))
    botPieceData = botPiece.head(int(bot_fraction*num_data_pts_to_use))
    
    dfToInterp = pd.concat([topPieceData, botPieceData])
    x, y, z = dfToInterp['x'].values, dfToInterp['y'].values, dfToInterp['z'].values
    

    # * Perform the spline interpolation
    tck, u = splprep([x, y, z], s=0)
    new_points = splev(np.linspace(0, 1, num_interpolation_pts), tck)

    interpDF = pd.DataFrame({
        'x': new_points[0], 'y': new_points[1], 'z': new_points[2]
    })
    
    topPieceTrimmed = topPiece.head(len(topPiece)-int(top_fraction*num_data_pts_to_use))
    botPieceTrimmed = botPiece.tail(len(botPiece)-int(bot_fraction*num_data_pts_to_use))

    return interpDF, topPieceTrimmed, botPieceTrimmed



def process_particle_trace(trace_path):
    
    pTrace = pd.read_csv(trace_path)

    # * 1st Dissection
    drop_start_index_top = pTrace[pTrace['y'] < data.YMAX - data.PLATE_THICKNESS].iloc[0].name # Leaves the top plate
    drop_start_index_bot = pTrace[pTrace['y'] < data.YMAX - data.PLATE_THICKNESS - data.HEIGHT_ONE_REP].iloc[0].name # Leaves the top plate and the first rep

    topPart, midPart, botPart = dissect_trace(pTrace, drop_start_index_top, drop_start_index_bot)

    # * Interpolate the data connecting the pieces to have a finer dissection
    interpTop, topPieceTrimmed, _ = interpoolate_bet_parts(topPart, midPart, num_interpolation_pts=300, num_data_pts_to_use=30, top_fraction=1/2, bot_fraction=1/2)
    interpBot, _, botPieceTrimmed = interpoolate_bet_parts(midPart, botPart, num_interpolation_pts=300, num_data_pts_to_use=30, top_fraction=1/2, bot_fraction=1/2)
    midPartTrimmed = midPart.iloc[int(30*1/2):int(-30*1/2)] # Trim the middle part

    # Reset index
    for df in [topPieceTrimmed, interpTop, midPartTrimmed, interpBot, botPieceTrimmed]:
        df.reset_index(drop=True, inplace=True)

    wholeTrace = pd.concat([topPieceTrimmed, interpTop, midPartTrimmed, interpBot, botPieceTrimmed]).reset_index()

    # * 2nd Dissection (with finer data)
    drop_start_index_top = wholeTrace[wholeTrace['y'] < data.YMAX - data.PLATE_THICKNESS].iloc[0].name
    drop_start_index_bot = wholeTrace[wholeTrace['y'] < data.YMAX - data.PLATE_THICKNESS - data.HEIGHT_ONE_REP].iloc[0].name

    newTopPart, newMidPart, newBotPart = dissect_trace(wholeTrace, drop_start_index_top, drop_start_index_bot)

    # * Create copies of the middle piece
    midPartCopies = {i: newMidPart.copy() for i in range(data.NUM_REPS)}
    # Shift the middle pieces in y  
    for i in range(data.NUM_REPS):
        midPartCopies[i]['y'] -= i*data.HEIGHT_ONE_REP
            
    # Shift the bot piece
    newBotPart['y'] -= (data.NUM_REPS-1)*data.HEIGHT_ONE_REP

    interpMidPieces = {}
    for i in range(data.NUM_REPS-1):
        connectorPiece, upPiece, downPiece = interpoolate_bet_parts(midPartCopies[i].iloc[:-10], midPartCopies[i+1].iloc[10:], num_interpolation_pts=150, num_data_pts_to_use=100, top_fraction=1/4, bot_fraction=3/4)        
        
        connectorPieceUP = connectorPiece[connectorPiece['y'] > data.YMAX - data.PLATE_THICKNESS - (i+1)*data.HEIGHT_ONE_REP]
        connectorPieceDOWN = connectorPiece[connectorPiece['y'] < data.YMAX - data.PLATE_THICKNESS - (i+1)*data.HEIGHT_ONE_REP]
        
        topPiece = pd.concat([upPiece, connectorPieceUP]).reset_index(drop=True)
        bottomPiece = pd.concat([connectorPieceDOWN, downPiece]).reset_index(drop=True)
        
        interpMidPieces[i] = topPiece
        midPartCopies[i+1] = bottomPiece
        

    finalTrace = newTopPart.copy()
    for i in range(data.NUM_REPS-1):
        finalTrace = pd.concat([finalTrace, interpMidPieces[i]])
    finalTrace = pd.concat([finalTrace, midPartCopies[data.NUM_REPS-1], newBotPart]).reset_index()




    
def main():
    files = list(data.DIR_PATH.glob("particle_data_trimmed/*.csv"))
    
    # Suppress warnings from pandas
    pd.options.mode.chained_assignment = None  # default='warn'
    
    with multiprocessing.Pool(processes=8) as pool:
        list(tqdm(pool.imap(process_particle_trace, files), total=len(files)))
        
if __name__ == "__main__":
    main()