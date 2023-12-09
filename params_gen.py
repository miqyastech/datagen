# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Output Directory
OUTPUT_DIR = './data/output/'

# A Random Seed for Reproducibility
seed = 191
np.random.seed(seed)

# Meshes to be generated
NUM_MESHES = 75000

# Setting up Beta values from normal distribution with mean=0 and standard deviation=1.5
betas = np.random.normal(0, 1.5, (NUM_MESHES, 300))

# Permutations of Left and Right Shoulder Positions
left_shoulder_z = np.linspace(start=-0.7, stop=-1.1, num=11)
right_shoulder_z = np.linspace(start=0.67, stop=1.07, num=11)

# Setting up Pose Parameters n=72, we are assuming a fixed pose
num_pose_params = 72

pose = np.zeros(num_pose_params)

pose[54] = 0.15
# Right Elbow 'x'
pose[57] = 0.15
# Left Collar [x,y,z]
pose[39:42] = [-0.15, 0, -0.4]
# Right Collar [x,y,z]
pose[42:45] = [-0.15, 0, 0.4]
# Neck 'x'
pose[45] = 0.30

# Parameters List [Beta + Pose]
params_list = []

# Generating Parameters
for ix in tqdm(range(NUM_MESHES)):
    # filename should contain fix number of digits
    fname = 'mesh_' + str(ix).zfill(5) + '.obj'

    beta = betas[ix]
    idx = np.random.choice(np.arange(len(left_shoulder_z)), 1, replace=False)
    lsz = left_shoulder_z[idx][0]
    rsz = right_shoulder_z[idx][0]

    pose[50] = lsz
    pose[53] = rsz

    params_row = [fname] + list(beta) + list(pose)
    params_list.append(params_row)

# Preparation for generating a csv file for train/test purpose
beta_columns_header = [
                          'B1_Height', 'B2_Weight', 'B3_Inseam_and_Reach', 'B4_Hip_Height',
                          'B5_Neck_Lenght', 'B6_Subtle_Weight', 'B7_Torso_vs_Lims',
                          'B8_Chest_Height', 'B9_Fitness', 'B10_Vertical_Weight'
                      ] + ['B' + str(i) for i in range(11, 301)]
pose_columns_header = ['P' + str(i) for i in range(0, 72)]
body_joints = [
    'Pelvis', 'Left_Hip', 'Right_Hip', 'LowBack', 'Left_Knee', 'Right_Knee', 'Middle_Back', 'Left_Ankle', 'Right_Ankle',
    'Top_Back', 'Left_Foot', 'Right_Foot', 'Neck', 'Left_Collar', 'Right_Collar', 'Head(tilt)', 'Left_Shoulder',
    'Right_Shoulder', 'Left_Elbow', 'Right_Elbow', 'Left_Wrist', 'Right_Wrist', 'Left_Hand', 'Right_Hand'
]
j = 0
for i, v in enumerate(pose_columns_header):
    pose_columns_header[i] = v + '_' + body_joints[j]
    if (i + 1) % 3 == 0:
        j += 1

header = ['synth_name'] + beta_columns_header + pose_columns_header

params_df = pd.DataFrame(data=params_list, columns=header)
params_df.to_csv(os.path.join(OUTPUT_DIR, 'params.csv'), index=False)
