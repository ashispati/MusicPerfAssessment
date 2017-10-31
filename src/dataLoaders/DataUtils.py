import sys
import os
import json
import dill
from collections import defaultdict
import numpy as np
import pandas as pd
from pandas import ExcelFile

class DataUtils(object):
    """
    Class containing helper functions to read the music performance data from the FBA folder
    """

    def __init__(self, path_to_annotations, band, instrument):
        """
        Initialized the data utils class
        Arg:
                path_to_annotations:	string, full path to the folder containing the FBA annotations
                band:					string, which band type
                instrument:				string, which instrument
        """
        self.path_to_annotations = path_to_annotations
        self.band = band
        self.instrument = instrument

    def get_excel_file_path(self, year):
        """
        Returns the excel file name containing the student performance details
        Arg:
                year:	string, which year
        """
        if self.band == 'middle':
            file_name = 'Middle School'
        elif self.band == 'concert':
            file_name = 'Concert Band Scores'
        elif self.band == 'symphonic':
            file_name = 'Symphonic Band Scores'
        else:
            raise ValueError("Invalid value for 'band'")
        xls_path = 'FBA' + year + '/'
        xls_file_path = self.path_to_annotations + xls_path + file_name + '.xlsx'
        return xls_file_path

    def get_anno_folder_path(self, year):
        """
        Returns the full path to the root folder containing all the FBA segment files and
		assessments
        Arg:
                year:	string, which year
        """
        if self.band == 'middle':
            folder_name = 'middleschool'
        elif self.band == 'concert':
            folder_name = 'concertband'
        elif self.band == 'symphonic':
            folder_name = 'symphonicband'
        else:
            raise ValueError("Invalid value for 'band'")
        annotations_folder = self.path_to_annotations + 'FBA' + year + '/' + folder_name
        if year == '2013':
            annotations_folder += 'scores/'
        else:
            annotations_folder += '/'
        return annotations_folder
    
    def scan_student_ids(self, year):
        """
        Returns the student ids for the provide inputs as a list
        Args:
                year:	string, which year
        """
        # get the excel file path
        file_path = self.get_excel_file_path(year)

        # read the excel file
        xldata = pd.read_excel(file_path)
        instrument_data = xldata[xldata.columns[0]]
        # find the index where the student ids start for the input instrument
        start_idx = 0
        while instrument_data[start_idx] != self.instrument:
            start_idx += 1
        # iterate and the store the student ids
        student_ids = []
        while isinstance(instrument_data[start_idx + 1], int):
            student_ids.append(instrument_data[start_idx + 1])
            start_idx += 1
        return student_ids

    def get_segment_info(self, year, segment, student_ids=[]):
        """
        Returns the segment info for the provide inputs as a list of tuples (start_time, end_time)
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within this
								 function
        """
        annotations_folder = self.get_anno_folder_path(year)
        segment_data = []
        if student_ids == []:
            student_ids = self.scan_student_ids(year)
        for student_id in student_ids:
            segment_file_path = annotations_folder + \
                str(student_id) + '/' + str(student_id) + '_segment.txt'
            file_info = [line.rstrip('\n')
                         for line in open(segment_file_path, 'r')]
            segment_info = file_info[segment]
            if sys.version_info[0] < 3:
                to_floats = map(float, segment_info.split('\t'))
            else:
                to_floats = list(map(float, segment_info.split('\t')))
            # convert to tuple and append
            segment_data.append((to_floats[0], to_floats[0] + to_floats[1]))
        return segment_data

    def get_pitch_contours_segment(self, year, segment, student_ids=[]):
        """
        Returns the pitch contours for the provide inputs as a list of np arrays
                assumes pyin pitch contours have already been computed and stored as text files
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within this
								 function
        """
        data_folder = self.get_anno_folder_path(year)
        if student_ids == []:
            student_ids = self.scan_student_ids(year)
        segment_info = self.get_segment_info(year, segment, student_ids)
        pitch_contour_data = []
        idx = 0
        for student_id in student_ids:
            pyin_file_path = data_folder + \
                str(student_id) + '/' + str(student_id) + \
                '_pyin_pitchtrack.txt'
            lines = [line.rstrip('\n') for line in open(pyin_file_path, 'r')]
            pitch_contour = []
            start_time, end_time = segment_info[idx]
            idx = idx + 1
            for x in lines:
                if sys.version_info[0] < 3:
                    to_floats = map(float, x.split(','))
                else:
                    to_floats = list(map(float, x.split(',')))
                timestamp = to_floats[0]

                if timestamp < start_time:
                    continue
                else:
                    if timestamp > end_time:
                        break
                    else:
                        pitch = to_floats[1]
                        pitch_contour.append(to_floats[1])
            pitch_contour = np.asarray(pitch_contour)
            pitch_contour_data.append(pitch_contour)

        return pitch_contour_data

    def get_perf_rating_segment(self, year, segment, student_ids=[]):
        """
        Returns the performane ratings given by human judges for the input segment as a list of
		 tuples
        Args:
                year:			string, which year
                segment:		string, which segment
                student_ids:	list, containing the student ids., if empty we compute it within
								 this function
        """
        annotations_folder = self.get_anno_folder_path(year)
        perf_ratings = []
        if student_ids == []:
            student_ids = self.scan_student_ids(year)

        for student_id in student_ids:
            ratings_file_path = annotations_folder + \
                str(student_id) + '/' + str(student_id) + '_assessments.txt'
            file_info = [line.rstrip('\n')
                         for line in open(ratings_file_path, 'r')]
            segment_ratings = file_info[segment]
            if sys.version_info[0] < 3:
                to_floats = map(float, segment_ratings.split('\t'))
            else:
                to_floats = list(map(float, segment_ratings.split('\t')))
            # convert to tuple and append
            perf_ratings.append(
                (to_floats[2], to_floats[3], to_floats[4], to_floats[5]))

        return perf_ratings
