import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import numpy as np
import math
from collections import defaultdict

'''
STENTORCAM ISOLATED DATA ANALYSIS CODE

Takes in csv file of coordinates from Fiji and visualizes individual trails as well as behaviors

Example command line argument: python3 /path_to_code.py -i path_to_data.csv -s1 4 -s2 7 -s3 12 etc. -color green -o output.png
                                                                            s1/s2/etc refers to a specific track id/ids in the csv file
                                                                            see argument parsing for further details


Output should contain a png plot visualizing each track id seperately with speed/behavior analysis

Functions included:

    parseInput(infile)
    plot_track(panel, track_coords)
    plot_velocity(panel, track_dict)
    plot_avg_velocity(panel, *speed_dicts) same as fulldataplot
    plot_sliding_avg_veloctiy(panel, speed_dict, window_size) same as fulldataplot
    plot_circle/plot_ref (stricly plotting purposes)

'''

# Attempt to apply custom plotting style
try:
    plt.style.use('BME163')
except:
    print("Warning: 'BME163' style not found. Using default style.")

# ----------------------- #
# Argument Parsing
# ----------------------- #
parser = argparse.ArgumentParser(description='Generate track visualization')
parser.add_argument('-i', '--input', required=True)
parser.add_argument('-o', '--output', default='track.png')
parser.add_argument('-s1', '--stent1', type=str) #track id for specific track id/ids csv
parser.add_argument('-s2', '--stent2', type=str) # //
parser.add_argument('-s3', '--stent3', type=str) # // 
parser.add_argument('-s4', '--stent4', type=str) # //
parser.add_argument('-s5', '--stent5', type=str) # //
parser.add_argument('-color', '--color', type=str, default='gray')
args = parser.parse_args()

# creates variables for command line input
inFile = args.input
outFile = args.output
t1_ids = list(map(int, args.stent1.split(','))) if args.stent1 else [] #can collect multiple track ids for one output trail (useful for stitching together tracks                            
t2_ids = list(map(int, args.stent2.split(','))) if args.stent2 else [] #that may have been separated during image processing)
t3_ids = list(map(int, args.stent3.split(','))) if args.stent3 else []
t4_ids = list(map(int, args.stent4.split(','))) if args.stent4 else []
t5_ids = list(map(int, args.stent5.split(','))) if args.stent5 else []
inputColor = str(args.color)

if inputColor == 'full-spectrum': #for plotting purposes full-spectrum is adjusted to purple, no actual data will be adjusted
    inputColor = 'purple'


if not any([t1_ids, t2_ids, t3_ids, t4_ids, t5_ids]):
    exit("No track IDs specified for any stentor.")

# ----------------------- #
# Set up Figure & Panels
# ----------------------- #
figureHeight = 9.5
figureWidth = 28
fig = plt.figure(figsize=(figureWidth, figureHeight))

# Top row
scale1 = plt.axes([(0.75)/figureWidth, 6/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
scale2 = plt.axes([(4.75)/figureWidth, 6/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
scale3 = plt.axes([(8.75)/figureWidth, 6/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
scale4 = plt.axes([(12.75)/figureWidth, 6/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
scale5 = plt.axes([(16.75)/figureWidth, 6/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)

# Middle row
expand1 = plt.axes([(0.75)/figureWidth, 2.5/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
expand2 = plt.axes([(4.75)/figureWidth, 2.5/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
expand3 = plt.axes([(8.75)/figureWidth, 2.5/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
expand4 = plt.axes([(12.75)/figureWidth, 2.5/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)
expand5 = plt.axes([(16.75)/figureWidth, 2.5/figureHeight, 3/figureWidth, 3/figureHeight], frameon=True)

# Bottom row (velocity plots)
velocity1 = plt.axes([(0.75)/figureWidth, 0.5/figureHeight, 3/figureWidth, 1.5/figureHeight], frameon=True)
velocity2 = plt.axes([(4.75)/figureWidth, 0.5/figureHeight, 3/figureWidth, 1.5/figureHeight], frameon=True)
velocity3 = plt.axes([(8.75)/figureWidth, 0.5/figureHeight, 3/figureWidth, 1.5/figureHeight], frameon=True)
velocity4 = plt.axes([(12.75)/figureWidth, 0.5/figureHeight, 3/figureWidth, 1.5/figureHeight], frameon=True)
velocity5 = plt.axes([(16.75)/figureWidth, 0.5/figureHeight, 3/figureWidth, 1.5/figureHeight], frameon=True)

# Side panels
legend = plt.axes([(20.5)/figureWidth, 2.25/figureHeight, 0.75/figureWidth, 3.5/figureHeight], frameon=True)
avg_velocity = plt.axes([(22.75)/figureWidth, 1.5/figureHeight, 4/figureWidth, 2.5/figureHeight], frameon=True)
sliding_avg_velocity = plt.axes([(22.75)/figureWidth, 5.5/figureHeight, 4/figureWidth, 2.5/figureHeight], frameon=True)

# ----------------------- #
# Utility Functions
# ----------------------- #
def parseInput(infile, ids1, ids2, ids3, ids4, ids5):
    #similar in function to fulldataplot see parseInput in fulldataplot code
    track1, track2, track3, track4, track5 = {}, {}, {}, {}, {}
    with open(infile) as f:
        for _ in range(4):
            next(f)
        for line in f:
            splitLine = line.strip().split(',')
            frame = float(splitLine[3])
            x, y = float(splitLine[1]), float(splitLine[2])
            track_id = int(splitLine[4])
            if track_id in ids1: track1[frame] = (x, y)
            if track_id in ids2: track2[frame] = (x, y)
            if track_id in ids3: track3[frame] = (x, y)
            if track_id in ids4: track4[frame] = (x, y)
            if track_id in ids5: track5[frame] = (x, y)
    return track1, track2, track3, track4, track5

def plot_track(panel, track_coords):
    #plots trails individually
    frames = sorted(track_coords)
    x_vals = [track_coords[f][0] for f in frames]
    y_vals = [track_coords[f][1] for f in frames]
    colors = ['black', inputColor, 'black']
    segments = [(x_vals[:900], y_vals[:900]), (x_vals[900:1800], y_vals[900:1800]), (x_vals[1800:], y_vals[1800:])]
    for seg, color in zip(segments, colors):
        panel.plot(*seg, color=color)
    
    panel.plot(x_vals[0], y_vals[0], 'o', color='green', markersize=5)
    panel.plot(x_vals[-1], y_vals[-1], 'o', color='red', markersize=5)

def plot_circle(panel):
    #reference circle
    theta = np.linspace(0, 2 * np.pi, 200)
    x = 320 + 254 * np.cos(theta)
    y = 256 + (254 * 512/640) * np.sin(theta)
    panel.plot(x, y, color='black')

def plot_ref(panel):
    #reference 1mm
    panel.axhline(y=475, color='black', xmin=0.7945125, xmax=0.9)
    panel.text(537.25, 455, "1mm", fontsize=10, ha='center', va='center')

def plot_velocity(panel, track_dict):
    #see compute_speed function in fulldataplot 
    speeds, frames = {}, sorted(track_dict)
    for i in range(1, len(frames)):
        x1, y1 = track_dict[frames[i-1]]
        x2, y2 = track_dict[frames[i]]
        dist = math.hypot(x2 - x1, y2 - y1)
        speed = dist / (frames[i] - frames[i-1]) * 30 / 56.25
        speeds[frames[i]] = speed
    panel.plot(list(speeds), list(speeds.values()), color='black', linewidth=0.5)
    return speeds

def plot_avg_velocity(panel, *speed_dicts):
    #plots avg speeds for all tracks listed in input
    avg_speed, count = defaultdict(float), defaultdict(int)
    for d in speed_dicts:
        for frame, val in d.items():
            avg_speed[frame] += val
            count[frame] += 1
    averaged = {f: avg_speed[f]/count[f] for f in avg_speed}
    frames = sorted(averaged)
    panel.plot(frames, [averaged[f] for f in frames], color='black', linewidth=0.5)
    return averaged

def plot_sliding_avg_velocity(panel, speed_dict, window_size):
    #plots sliding avg smoothed speeds 
    frames = sorted(speed_dict)
    speeds = [speed_dict[f] for f in frames]
    half = window_size // 2
    smoothed = [np.mean(speeds[max(0,i-half):min(len(speeds), i+half+1)]) for i in range(len(speeds))]
    panel.plot(frames, smoothed, color='black', linewidth=1)

# ----------------------- #
# Process Input & Plot
# ----------------------- #
track1, track2, track3, track4, track5 = parseInput(inFile, t1_ids, t2_ids, t3_ids, t4_ids, t5_ids)
track1 = dict(sorted(track1.items()))
track2 = dict(sorted(track2.items()))
track3 = dict(sorted(track3.items()))
track4 = dict(sorted(track4.items()))
track5 = dict(sorted(track5.items()))

for p, t in zip([scale1, scale2, scale3, scale4, scale5], [track1, track2, track3, track4, track5]):
    plot_track(p, t)
    plot_circle(p)
    plot_ref(p)

for p, t in zip([expand1, expand2, expand3, expand4, expand5], [track1, track2, track3, track4, track5]):
    plot_track(p, t)
    p.invert_yaxis()

speed1 = plot_velocity(velocity1, track1)
speed2 = plot_velocity(velocity2, track2)
speed3 = plot_velocity(velocity3, track3)
speed4 = plot_velocity(velocity4, track4)
speed5 = plot_velocity(velocity5, track5)

avg_speeds = plot_avg_velocity(avg_velocity, speed1, speed2, speed3, speed4, speed5)
plot_sliding_avg_velocity(sliding_avg_velocity, avg_speeds, window_size=20)

for panel in [velocity1, velocity2, velocity3]:
    for x in [900, 1800]:
        panel.axvline(x=x, color='gray', linestyle=(0, (2, 4)), linewidth=0.5)

# ----------------------- #
# Final Annotations & Axes
# ----------------------- #
# Labels
titles = ["Stentor 1", "Stentor 2", "Stentor 3", "Stentor 4", "Stentor 5"]
for i, title in enumerate(titles):
    fig.text((2.25 + 4*i) / figureWidth, 0.97, title, fontsize=20, ha='center', va='center')

fig.text(0.5/figureWidth, 0.421052, "Expanded", fontsize=20, ha='center', va='center', rotation=90)
fig.text(0.5/figureWidth, 0.789473, "Scale", fontsize=20, ha='center', va='center', rotation=90)

# Axis Limits
for p in [scale1, scale2, scale3, scale4, scale5]:
    p.set_xlim(0, 640)
    p.set_ylim(512, 0)

for p in [velocity1, velocity2, velocity3, velocity4, velocity5]:
    p.set_xlim(0, 2700)
    p.set_ylim(0, 1.5)

avg_velocity.set_xlim(0, 2700)
avg_velocity.set_ylim(0, 1.25)
sliding_avg_velocity.set_xlim(0, 2700)
sliding_avg_velocity.set_ylim(0, 1.25)

# Time Bar
legend.set_xlim(0, 1)
legend.set_ylim(0, 1)
legend.set_yticks([0, 0.16666, 0.333333, 0.5, 0.6666666, 0.833333, 1])
legend.set_yticklabels(['90s', 'LED Off', '60s', 'LED On', '30s', 'LED Off', '0s'], fontsize=10)
legend.yaxis.tick_right()
legend.set_xticks([])
legend.tick_params(length=7, width=1)

for rect, color in zip([patches.Rectangle((0, y), 1, h, color=c, alpha=0.6)
                        for y, h, c in [(0, 0.015, 'red'), (0.34, 0.33, inputColor), (0.985, 0.015, 'green')]],
                        ['none', inputColor, 'none']):
    legend.add_patch(rect)

# Format Plots
for p in [velocity1, velocity2, velocity3, velocity4, velocity5, avg_velocity, sliding_avg_velocity]:
    p.set_xticks([0, 450, 900, 1350, 1800, 2250, 2700])
    p.set_xticklabels(['0', '15', '30', '45', '60', '75', '90'])
    p.set_xlabel('Time (seconds)')
    p.set_ylabel('Speed (millimeters per second)')
    p.axvspan(0, 900, color='none',alpha=0.2)
    p.axvspan(900, 1800, color=inputColor, alpha=0.2)
    p.axvspan(1800, 2700, color='none',alpha=0.2)

for p in [scale1, scale2, scale3, expand1, expand2, expand3, scale4, scale5, expand4, expand5]:
    p.ticklabel_format(style='plain', axis='x')
    p.set_xticklabels([])
    p.set_yticklabels([])
    p.set_xticks([])
    p.set_yticks([])

# ----------------------- #
# Save Plot
# ----------------------- #
plt.savefig(outFile, dpi=600)
