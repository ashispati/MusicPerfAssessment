import json
import dill
from collections import defaultdict
import numpy as np
import sys
import sys
import os
from DataUtils import DataUtils

# Initialize input params, specify the band, intrument, segment information
BAND = 'middle'
INSTRUMENT = 'Alto Saxophone'
SEGMENT = 2
YEAR = ['2013', '2014', '2015']

## define paths to FBA dataset and FBA annotations
PATH_FBA_DATA = '/Users/Ashis/Documents/Github/FBA2013data/'
PATH_FBA_ANNO = '/Users/Ashis/Documents/Github/FBA2013/'

utils = DataUtils()

## scan student ids based on the input params
student_ids = {}
for year in YEAR:
    student_ids[year] = utils.scan_student_ids(PATH_FBA_ANNO, BAND, INSTRUMENT, year)

## extract segment information from the annotations
segment_data = {}
for year in YEAR:
    segment_data[year] = utils.get_segment_info(PATH_FBA_ANNO, student_ids[year], BAND, year, SEGMENT)

## extract pYin pitch contours for all files 
pitch_contour_data = {}
for year in YEAR:
    pitch_contour_data[year] = utils.get_pyin_pitch_contour(PATH_FBA_DATA, student_ids[year], BAND, year, segment_data[year])

## extract ground truth ratings for all files



## put everything together and save as .dll file



