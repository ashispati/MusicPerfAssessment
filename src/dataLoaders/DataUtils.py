import json
import dill
from collections import defaultdict
import numpy as np
import sys
import os
import pandas as pd 
from pandas import ExcelFile

class DataUtils():

	@staticmethod
	def scan_student_ids(path_to_annotations, band, instrument, year):
		"""
		Returns the student ids for the provide inputs as a list 
		Args:
			path_to_annotations:	string, full path to the folder containing the FBA annotations
			band:					string, which band type
			instrument:				string, which instrument type
			year:					string, which year 
		"""
		if band == 'middle':
			file_name = 'Middle School'
		elif band == 'concert':
			file_name = 'Concert Band Scores'
		elif band == 'symphonic':
			file_name = 'Symphonic Band Scores'
		else:
			raise ValueError("Invalid value for 'band'")

		xls_path = 'FBA' + year + '/'
		file_path = path_to_annotations + xls_path + file_name + '.xlsx'
	
		# read the excel file
		xldata = pd.read_excel(file_path) 
		instrument_data = xldata[xldata.columns[0]]
		start_idx = 0
		while instrument_data[start_idx] != instrument:
			start_idx += 1

		student_ids = []
		while isinstance(instrument_data[start_idx + 1], (int, long)):
			student_ids.append(instrument_data[start_idx + 1]) 
			start_idx += 1
		return student_ids

	@staticmethod
	def get_segment_info(path_to_annotations, student_ids, band, year, segment):
		"""
		Returns the segment info for the provide inputs as a list of tuples (start_time, end_time)
		Args:
			path_to_annotations:	string, full path to the folder containing the FBA annotations
			student_ids:			list, containing the student ids 
			band:					string, which band type
			year:					string, which year 
			segment:				int, which segment
		"""
		# create the path to search path
		if band == 'middle':
			folder_name = 'middleschool'
		elif band == 'concert':
			folder_name = 'concertband'
		elif band == 'symphonic':
			folder_name = 'symphonicband'
		else:
			raise ValueError("Invalid value for 'band'")
		search_folder = path_to_annotations + 'FBA' + year + '/' + folder_name
		if year == '2013':
			search_folder += 'scores/'
		else:
			search_folder += '/'

		# iterate through the individual files ans store values
		segment_data = []
		for student_id in student_ids:
			segment_file_path = search_folder + str(student_id) + '/' + str(student_id) + '_segment.txt'
			file_info = [line.rstrip('\n') for line in open(segment_file_path, 'r')]
			segment_info = file_info[segment]
			if sys.version_info[0] < 3:
				to_floats = map(float, segment_info.split('\t'))
			else:
				to_floats = list(map(float, segment_info.split('\t')))
			# convert to tuple and append
			segment_data.append((to_floats[0], to_floats[0] +  to_floats[1]))
		return segment_data

	@staticmethod
	def get_pyin_pitch_contour(path_to_data, student_ids, band, year, segment_info):
		"""
		Returns the pitch contours for the provide inputs as a list of np arrays
			assumes pyin pitch contours have already been computed and stored as text files
		Args:
			path_to_data:			string, full path to the folder containing the FBA data
			student_ids:			list, containing the student ids 
			band:					string, which band type
			year:					string, which year 
			segment_info:			list of tuples, same length as student_ids, containing starting and ending time of the segment
		"""
		# create the path to search path
		if year == '2013':
			year_folder = '2013-2014'
		elif year == '2014':
			year_folder = '2014-2015'
		elif year == '2015':
			year_folder = '2015-2016'
		else:
			raise ValueError("Invalid value for 'year'")
		if band == 'middle':
			folder_name = 'middleschool'
		elif band == 'concert':
			folder_name = 'concertband'
		elif band == 'symphonic':
			folder_name = 'symphonicband'
		else:
			raise ValueError("Invalid value for 'band'")
		search_folder = path_to_data + year_folder + '/' + folder_name
		if year == '2013':
			search_folder += 'scores/'
		else:
			search_folder += '/'

		pitch_contour_data = []
		idx = 0
		for student_id in student_ids:
			pyin_file_path = search_folder + str(student_id) + '/' + str(student_id) + '_pyin_pitchtrack.txt'
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
