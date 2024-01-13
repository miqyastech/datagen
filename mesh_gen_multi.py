import os
import time
import numpy as np
import pandas as pd
import trimesh as tm
import multiprocessing
from tqdm import tqdm
from smpl.smpl_webuser.serialization import load_model


def mesh_generation(inp_slice: pd.DataFrame, slice_num, model_fname, output_folder='./output'):
    model = load_model(os.path.join("./smpl/models", model_fname))

    volumes = []
    heights = []

    # We now traverse through each row of the dataframe and generate the meshes
    for ix, row in df.iterrows():
        # Extracting the parameters from the row
        fname = row["synth_name"]
        beta = row[1:301].values
        pose = row[301:].values

        # Generating the mesh
        model.betas[:] = beta
        model.pose[:] = pose

        mesh_dict = {"vertices": model.r, "faces": model.f}

        min_y = np.min(model.r[:, 1])
        max_y = np.max(model.r[:, 1])

        height = round(abs(max_y - min_y), 4)
        heights.append(height)

        mesh = tm.load(mesh_dict, process=False, maintain_order=True)

        volumes.append(mesh.volume)

        # Saving the mesh
        mesh.export(os.path.join(OUTPUT_DIR, fname))


    # Saving the volumes and heights to a csv file
    vh_df = pd.DataFrame(
        {"synth_name": df["synth_name"], "volume": volumes, "height": heights}
    )

    _csv_name = f'chunk_{str(slice_num)}_volumes_heights.csv'
    vh_df.to_csv(os.path.join(OUTPUT_DIR, _csv_name), index=False)
    
    print('LOG: SAVED CSV FOR CHUNK -- ', slice_num)


def assign_workers_for_mesh_generation(inp_df, model_fname, output_folder='./output/', num_workers=8):

    chunks = np.array_split(inp_df, num_workers)

    starttime = time.time()
    processes = []
    for i in range(0, num_workers):
        p = multiprocessing.Process(target=mesh_generation,
                                    args=(chunks[i], i, model_fname, output_folder,))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))

if __name__ == "__main__":

    INPUT_DIR = "./data/input/"
    OUTPUT_DIR = "./data/output/meshes"

    # Read the csv file
    df = pd.read_csv(INPUT_DIR + "params.csv")

    # Model Choice - 1 = Male / 2 = Female
    MODEL_FILENAME: str
    MODEL_CHOICE = 2

    if MODEL_CHOICE == 1:
        MODEL_FILENAME = "basicModel_m_lbs_10_207_0_v1.1.0.pkl"
    elif MODEL_CHOICE == 2:
        MODEL_FILENAME = "basicModel_f_lbs_10_207_0_v1.1.0.pkl"

    assign_workers_for_mesh_generation(df, MODEL_FILENAME, OUTPUT_DIR, num_workers=8)
