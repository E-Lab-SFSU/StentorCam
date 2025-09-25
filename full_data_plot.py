import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import numpy as np
import math
from collections import defaultdict

'''
STENTORCAM FULL DATA ANALYSIS CODE

Takes in csv file of coordinates from Fiji and calculates speeds/behaviors and plots the averages

Example command line argument: python3 /path_to_code.py -i path_to_data.csv -o output.png -color orange -wave 605

Output should contain a png plot visualizing speed/behavoir data from the input file

Functions included:

    parseInput(infile)
    compute_speeds(track_coords)
    plot_avg_velocity(panel, *speed_dicts)
    plot_sliding_avg_veloctiy(panel, speed_dict, window_size)

'''
# Attempt to apply custom plotting style
try:
    plt.style.use('BME163') #BME163 can be substituted with any matplotlib custom style
except:
    print("Warning: 'BME163' style not found. Using default style.")

# ----------------------- #
# Argument Parsing
# ----------------------- #
parser = argparse.ArgumentParser(description='Generate avg velocity visualization over all tracks')
parser.add_argument('-i', '--input', required=True) #input csv file, ex: -i data.csv
parser.add_argument('-o', '--output', default='avg_velocity.png') #output file name, ex: -o output.png
parser.add_argument('-color', '--color', type=str, default="gray") #color of laser in experiment for plotting purposes, ex: -color orange 
parser.add_argument('-wave', '--wavelength', type=str, default='no input') #wavelength of color used, ex: -wave 600
args = parser.parse_args()

#creates variables from command line arguments 
inFile = args.input
outFile = args.output
inputColor = str(args.color)
inputWave = str(args.wavelength)
if inputColor == 'full-spectrum': #for plotting purposes full-spectrum is adjusted to purple, no actual data will be adjusted
    inputColor = 'purple'
    



# ----------------------- #
# Set up Figure & Panels
# ----------------------- #
figureHeight = 5.8
figureWidth = 6.2+2
fig = plt.figure(figsize=(figureWidth, figureHeight))

legend = plt.axes([(5+2)/figureWidth, 1.4/figureHeight, 0.4/figureWidth, 3/figureHeight], frameon=True)
avg_velocity = plt.axes([(0.7)/figureWidth, 0.5/figureHeight, (4+2)/figureWidth, 1.8/figureHeight], frameon=True)
sliding_avg_velocity = plt.axes([(0.7)/figureWidth, 3.5/figureHeight, (4+2)/figureWidth, 1.8/figureHeight], frameon=True)

# ----------------------- #
# Utility Functions
# ----------------------- #
def parseInput(infile):
    tracks = defaultdict(dict)  # track_id -> {frame: (x,y)}  stores necessary variables in nested dict
    with open(infile) as f:
        for _ in range(4):
            next(f)  # skip header lines may need to adjust depending on csv headers
        for line in f:
            splitLine = line.strip().split(',')
            frame = float(splitLine[3])
            x, y = float(splitLine[1]), float(splitLine[2])
            track_id = int(splitLine[4])
            tracks[track_id][frame] = (x, y)
    return tracks

def compute_speeds(track_coords):
    speeds = {}     #compiles speeds for each frame into a dict that can be sorted by frame number
    frames = sorted(track_coords)   #sorts coords to ensure coordinates are in chronological order
    for i in range(1, len(frames)):     #iterates frame by frame
        x1, y1 = track_coords[frames[i-1]]   #takes two frames at a time to calculate distance between the cell from frame to frame 
        x2, y2 = track_coords[frames[i]]
        dist = math.hypot(x2 - x1, y2 - y1)
        speed = dist / (frames[i] - frames[i-1]) * 30 / 56.25    
                                                ''' speed adjusted to mm/s based on given frame rate & scaling ***this must be adjusted for using reference, see below
                                                    (30 referes to frames per second / 56.25 refers to pixels/mm or the number of pixels covered by a 1mm reference)
                                                    pixels/mm can be obtained by measuring on Fiji the number of pixels that span a 1mm reference slide using the exact zoom setup used for recordings'''
        speeds[frames[i]] = speed
    return speeds

def plot_avg_velocity(panel, *speed_dicts):
    # Aggregate all speeds per frame
    speed_data = defaultdict(list)
    for d in speed_dicts:
        for frame, val in d.items():
            speed_data[frame].append(val)
    
    # Compute mean and std for each frame
    frames = sorted(speed_data)
    means = []
    stds = []
    for f in frames:
        values = speed_data[f]
        means.append(np.mean(values)) #use numpy to calculate means and stds
        stds.append(np.std(values))
    
    # Plot average line
    panel.plot(frames, means, color='black', linewidth=0.8, label='Mean Velocity')
    
    # Plot ±1 std deviation lines
    upper = [m + s for m, s in zip(means, stds)]
    lower = [m - s for m, s in zip(means, stds)]
    panel.plot(frames, upper, color='none', linewidth=0.3, label='+1 SD')
    panel.plot(frames, lower, color='none', linewidth=0.3, label='-1 SD')

    # Fill between std deviation lines
    panel.fill_between(frames, lower, upper, color='black', alpha=0.2)

    return dict(zip(frames, means))  # for use in sliding avg

def plot_sliding_avg_velocity(panel, speed_dict, window_size):
    #sort speeds by frame number
    frames = sorted(speed_dict)
    speeds = [speed_dict[f] for f in frames]
    half = window_size // 2

    smoothed_means = []
    smoothed_stds = []

    #calculate sliding window averages
    for i in range(len(speeds)):
        window = speeds[max(0, i - half):min(len(speeds), i + half + 1)]
        smoothed_means.append(np.mean(window))
        smoothed_stds.append(np.std(window))

    # Plot the smoothed average line
    panel.plot(frames, smoothed_means, color='black', linewidth=0.8)

    # Compute and plot ±1 SD lines
    upper = [m + s for m, s in zip(smoothed_means, smoothed_stds)]
    lower = [m - s for m, s in zip(smoothed_means, smoothed_stds)]
    panel.plot(frames, upper, color='none', linewidth=0.3)
    panel.plot(frames, lower, color='none', linewidth=0.3)
    panel.fill_between(frames, lower, upper, color='black', alpha=0.2)


# ----------------------- #
# Plot Legend Panel
# ----------------------- #
legend.set_xlim(0, 1)
legend.set_ylim(0, 1)
legend.set_yticks([0, 0.16666, 0.333333, 0.5, 0.6666666, 0.833333, 1])
legend.set_yticklabels(['90s', 'LED Off', '60s', 'LED On', '30s', 'LED Off', '0s'], fontsize=10)
legend.yaxis.tick_right()
legend.set_xticks([])
legend.tick_params(length=7, width=1)

for rect, color in zip(
    [patches.Rectangle((0, y), 1, h, color=c, alpha=0.3)
     for y, h, c in [(0, 0.34, 'none'), (0.34, 0.33, inputColor), (0.67, 0.33, 'none')]],
    ['none', 'inputColor', 'none']):
    legend.add_patch(rect)

# ----------------------- #
# Process Input & Compute Speeds
# ----------------------- #
tracks = parseInput(inFile)

#process data using compute function
num_tracks = len(tracks)
all_speeds = []
for track_id, coords in tracks.items():
    speeds = compute_speeds(coords)
    if speeds:
        all_speeds.append(speeds)

if not all_speeds:
    exit("No speed data available from input.")

# ----------------------- #
# Plot average velocity and sliding average velocity
# ----------------------- #
avg_speeds = plot_avg_velocity(avg_velocity, *all_speeds)
plot_sliding_avg_velocity(sliding_avg_velocity, avg_speeds, window_size=10)

# Axis limits and labels
for p in [avg_velocity, sliding_avg_velocity]:
    p.set_xlim(0, 2700)
    p.set_ylim(0, 1.25)

    # Major ticks every 15 seconds = 450 frames
    major_ticks = np.arange(0, 2701, 450)
    major_labels = [str(int(t / 30)) for t in major_ticks]

    # Minor ticks every 5 seconds = 150 frames
    minor_ticks = np.arange(0, 2701, 150)
    minor_labels = [str(int(t / 30)) for t in minor_ticks]

    # Set both major and minor ticks
    p.set_xticks(major_ticks)
    p.set_xticks(minor_ticks, minor=True)

    # Set both major and minor labels manually
    p.set_xticklabels(major_labels)
    p.set_xticklabels(minor_labels, minor=True)

    # Customize tick appearance
    p.tick_params(axis='x', which='major', length=7, width=1.2, labelsize=9)
    p.tick_params(axis='x', which='minor', length=4, width=0.8, labelsize=7)

    p.set_xlabel('Time (seconds)')
    p.set_ylabel('Speed (millimeters per second)')

    # Gridlines for both levels
    p.grid(True, which='major', axis='both', linestyle='--', linewidth=0.3)
    p.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.2)

    # Background shading
    p.axvspan(0, 900, color='none', alpha=0.2)
    p.axvspan(900, 1800, color=inputColor, alpha=0.3, zorder=0)
    p.axvspan(1800, 2700, color='none', alpha=0.2)

        # Add track count annotation
    p.text(0.98, 0.95, f'averaged over {num_tracks} tracks',
           ha='right', va='top', transform=p.transAxes,
           fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', lw=0.5))




if inputColor == 'purple':
    inputColor = 'full spectrum'

# ----------------------- #
# Titles
# ----------------------- #
fig.text((2.7+(+2/2))/figureWidth, 0.948276, "Smoothed Stentor Average Velocity (" + inputColor + " " + inputWave + "nm)", fontsize=14, ha='center', va='center')
fig.text((2.7+(+2/2))/figureWidth, 0.431, "Average Stentor Velocity (" + inputColor + " " + inputWave + "nm)", fontsize=14, ha='center', va='center')

# ----------------------- #
# Save Plot
# ----------------------- #
plt.savefig(outFile, dpi=600)
