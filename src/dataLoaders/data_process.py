import os
import sys
import dill
from collections import defaultdict
import numpy as np
from DataUtils import DataUtils

# Initialize input params, specify the band, intrument, segment information
BAND = 'middle'
INSTRUMENT = 'Alto Saxophone'
SEGMENT = 2
YEAR = ['2013', '2014', '2015']

# define paths to FBA dataset and FBA annotations
PATH_FBA_ANNO = '/Users/Som/GitHub/FBA2013/'

utils = DataUtils(PATH_FBA_ANNO, BAND, INSTRUMENT)

# scan student ids based on the input params
student_ids = {}
for year in YEAR:
    student_ids[year] = utils.scan_student_ids(year)

# extract pYin pitch contours for all files
pitch_contour_data = {}
for year in YEAR:
    pitch_contour_data[year] = utils.get_pitch_contours_segment(
        year, SEGMENT, student_ids[year])

# extract ground truth ratings for all files
ground_truth = {}
for year in YEAR:
    ground_truth[year] = utils.get_perf_rating_segment(
        year, SEGMENT, student_ids[year])


# put everything together and save as .dll file
perf_assessment_data = []
for year in YEAR:
    for student_idx in range(len(student_ids[year])):
        assessment_data = {}
        assessment_data['year'] = year
        assessment_data['band'] = BAND
        assessment_data['instrumemt'] = INSTRUMENT
        assessment_data['student_id'] = student_ids[year][student_idx]
        assessment_data['segment'] = SEGMENT
        assessment_data['pitch_contour'] = pitch_contour_data[year][student_idx]
        assessment_data['ratings'] = ground_truth[year][student_idx]
        perf_assessment_data.append(assessment_data)

file_name = BAND + '_' + INSTRUMENT[:4] + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    with open('../dat/' + file_name + '.dill', 'wb') as f:
        dill.dump(perf_assessment_data, f)
else:
    with open('../dat/' + file_name + '_3.dill', 'wb') as f:
        dill.dump(perf_assessment_data, f)
