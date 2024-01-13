import os
import numpy as np
import pandas as pd
import trimesh as tm
from smpl.smpl_webuser.serialization import load_model


INPUT_DIR = './data/input/'
OUTPUT_DIR = './data/output/meshes'

# Read the csv file
df = pd.read_csv(INPUT_DIR + 'params.csv')

# Model Choice - 1 = Male / 2 = Female
MODEL_FILENAME: str
MODEL_CHOICE = 2

if MODEL_CHOICE == 1:
    MODEL_FILENAME = 'basicModel_m_lbs_10_207_0_v1.1.0.pkl'
elif MODEL_CHOICE == 2:
    MODEL_FILENAME = 'basicModel_f_lbs_10_207_0_v1.1.0.pkl'

model = load_model(os.path.join('./smpl/models', MODEL_FILENAME))

volumes = []
heights = []

# We now traverse through each row of the dataframe and generate the meshes
for ix, row in df.iterrows():
    # Extracting the parameters from the row
    fname = row['synth_name']
    beta = row[1:301].values
    pose = row[301:].values

    # Generating the mesh
    model.betas[:] = beta
    model.pose[:] = pose

    mesh_dict = {
        'vertices': model.r,
        'faces': model.f
    }
    
    min_y = np.min(model.r[:, 1])
    max_y = np.max(model.r[:, 1])
    
    height = round(abs(max_y - min_y), 4)
    heights.append(height)
    
    mesh = tm.load(mesh_dict, process=False, maintain_order=True)
    
    volumes.append(mesh.volume)
    
    # Saving the mesh
    mesh.export(os.path.join(OUTPUT_DIR, fname))
    
    
# Saving the volumes and heights to a csv file
vh_df = pd.DataFrame({
    'synth_name': df['synth_name'],
    'volume': volumes,
    'height': heights
})

vh_df.to_csv(os.path.join(OUTPUT_DIR, 'volumes_heights.csv'), index=False)