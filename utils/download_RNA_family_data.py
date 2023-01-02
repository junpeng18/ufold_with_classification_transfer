import numpy as np
import pandas as pd
import os
import wget
import tqdm
from utils.path_utils import base_dir


def download_family_data(dir_out=None, max_nfam_per_architecture=10, seed=1234):
    # Load resource list
    resources = pd.read_table(
        os.path.join(base_dir, 'data/Rfam_fasta_resources.txt'), header=None, usecols=[1], names=['fasta_file']
    )
    resources = list(resources['fasta_file'])

    # Fetch sequence URLs
    index0 = pd.read_excel(
        os.path.join(base_dir, 'data/RNArchitecture.xlsx'), sheet_name='Database Family List', engine='openpyxl'
    )
    index = pd.DataFrame({
        'RFAM_ID': index0['RFAM#'],
        'architecture': index0['Architecture']
    })
    index['filename'] = [str(id) + '.fa.gz' for id in list(index['RFAM_ID'])]
    index = index.query('filename in @resources')
    fasta_path = 'https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta_files/'
    index['url'] = [os.path.join(fasta_path, name) for name in list(index['filename'])]

    # Limit data size
    np.random.seed(seed)
    archs = index['architecture'].unique()  # 19 architectures
    ids_to_download = []
    for arch in archs:
        ids_arch = index.query('architecture == @arch')['RFAM_ID'].unique()
        ids_arch = np.random.choice(
            ids_arch, min(max_nfam_per_architecture, len(ids_arch)), replace=False
        )
        ids_to_download.extend(ids_arch)
    index = index.query('RFAM_ID in @ids_to_download')

    # Download
    if dir_out is not None:
        for id, url in tqdm.tqdm(zip(index['RFAM_ID'], index['url']), total=index.shape[0]):
            wget.download(url, os.path.join(dir_out, str(id) + '.fa.gz'))

    return index


if __name__ == "__main__":
    index = download_family_data(os.path.join(base_dir, 'data/family_data'))
    print(index.iloc[0, ])
