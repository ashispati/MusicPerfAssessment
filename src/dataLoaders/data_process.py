import os
import sys
from collections import defaultdict
import dill
import numpy as np
import scipy.io
from tqdm import tqdm
from src.dataLoaders.DataUtils import DataUtils


# Initialize input params, specify the band, instrument, segment information
BAND = 'middle'
SEGMENT = 2
YEAR = ['2013', '2014', '2015']

# define paths to FBA dataset and FBA annotations
PATH_FBA_ANNO = os.path.realpath('../../../MIG-FbaData')
PATH_FBA_AUDIO = os.path.realpath('../../../FBA2013Data')

# create data holder
perf_assessment_data = []
req_audio = False
# instantiate the data utils object for different instruments and create the data
INSTRUMENT = 'Alto Saxophone'
utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, BAND, INSTRUMENT)
for year in tqdm(YEAR):
    perf_assessment_data += utils.create_data(year, SEGMENT, include_audio=req_audio)

INSTRUMENT = 'Bb Clarinet'
utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, BAND, INSTRUMENT)
for year in tqdm(YEAR):
    perf_assessment_data += utils.create_data(year, SEGMENT, include_audio=req_audio)

INSTRUMENT = 'Flute'
utils = DataUtils(PATH_FBA_ANNO, PATH_FBA_AUDIO, BAND, INSTRUMENT)
for year in tqdm(YEAR):
    perf_assessment_data += utils.create_data(year, SEGMENT, include_audio=req_audio)

print(len(perf_assessment_data))

file_name = BAND + '_' + str(SEGMENT) + '_data'
with open('../dat/' + file_name + '.dill', 'wb') as f:
    dill.dump(perf_assessment_data, f)
