import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io
from tqdm import tqdm
from src.dataLoaders.DataUtils import DataUtils


# Initialize input params, specify the band, instrument, segment information
BAND = ['middle', 'symphonic']
SEGMENT = 2
YEAR = ['2016', '2017', '2018']
INSTRUMENTS = [
    'Alto Saxophone',
    'Bb Clarinet',
    'Flute'
]

# define paths to FBA dataset and FBA annotations
PATH_FBA_ANNO = os.path.realpath('../../../MIG-FbaData')
PATH_FBA_AUDIO = os.path.realpath('../../../FBA2013Data')

count = 0
for band in tqdm(BAND):
    for instr in tqdm(INSTRUMENTS):
        utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, band, instr)
        for year in tqdm(YEAR):
            student_ids = utils.scan_student_ids(year)
            segment_info = utils.get_segment_info(year, SEGMENT, student_ids)
            for idx in range(len(segment_info)):
                seg = segment_info[idx]
                if seg is not None:
                    data_folder = utils.get_anno_folder_path(year)
                    sid = student_ids[idx]
                    pyin_file_path = os.path.join(
                        data_folder, str(sid), str(sid) + '_pyin_pitchtrack.txt'
                    )
                    if not os.path.exists(pyin_file_path):
                        count += 1
                        utils.compute_and_save_pitch_contour(year, sid, pyin_file_path)
print(count)
