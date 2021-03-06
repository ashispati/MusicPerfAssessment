import os
import subprocess
import sys
import numpy as np
import pandas as pd
import librosa

# define bad students ids 
# for which recording is bad or segment annotation doesn't exist
bad_ids = {}
bad_ids['middle'] = [29429, 32951, 42996, 43261, 44627, 56948, 39299, 39421, 41333, 42462, 43811, 44319, 61218, 29266, 33163]
bad_ids['symphonic'] = [33026, 33476, 35301, 41602, 52950, 53083, 46038, 33368, 42341, 51598, 56778, 56925, 30430, 55642, 60935]

SEGMENT_CRITERIA_DICT = {
    1: (
        'lyrical_etude',
        [
            'musicality_tempo_style',
            'note_accuracy',
            'rhythmic_accuracy',
            'tone_quality'
        ]
    ),
    2: (
        'technical_etude',
[
            'musicality_tempo_style',
            'note_accuracy',
            'rhythmic_accuracy',
            'tone_quality'
        ]
    )
    # TODO: add others
}


class DataUtils(object):
    """
    Class containing helper functions to read the music performance data from the FBA folder
    """

    def __init__(self, path_to_annotations, path_to_audio, band, instrument):
        """
        Initialized the data utils class
        Arg:
            path_to_annotations:    string, full path to the folder containing the FBA annotations
            path_to_audio:          string, full path to the folder containing the FBA audio
            band:                   string, which band type
            instrument:             string, which instrument
        """
        self.path_to_annotations = path_to_annotations
        self.path_to_audio = path_to_audio
        if band == 'middle':
            self.band = band + 'school'
        else:
            self.band = band + 'band'
        self.instrument = instrument
        self.bad_ids = bad_ids[band]
        self.assessments_file = 'normalized_student_scores.csv'
        self.path_to_pyin_n3 = os.path.realpath('pYin/pyin.n3')
        self.path_to_sonic_annotator = os.path.realpath('pYin/exec/')

    def get_excel_file_path(self, year):
        """
        Returns the excel file name containing the student performance details
        Arg:
            year:   string, which year
        """
        folder_name = 'FBA' + year
        file_name = 'excel' + self.band + '.xlsx'
        xls_file_path = os.path.join(self.path_to_annotations, folder_name, self.band, file_name)
        return xls_file_path

    def get_audio_folder_path(self, year):
        """
        Returns the full path to the root folder containing all the FBA audio files
        Arg:
            year:   string, which year
        """
        folder_name_year = year + '-' + str(int(year)+1)
        band_folder = self.band
        if year == '2013':
            band_folder += 'scores'
        path_to_audio_folder = os.path.join(self.path_to_audio, folder_name_year, band_folder)
        return path_to_audio_folder

    def get_anno_folder_path(self, year):
        """
        Returns the full path to the root folder containing all the FBA segment files and assessments
        Arg:
            year:   string, which year
        """
        folder_name = 'FBA' + year
        path_to_anno_folder = os.path.join(self.path_to_annotations, folder_name, self.band, 'assessments')
        return path_to_anno_folder
    
    def scan_student_ids(self, year):
        """
        Returns the student ids for the provide inputs as a list
        Args:
            year:   string, which year
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
        # remove bad student ids
        for i in range(len(self.bad_ids)):
            if self.bad_ids[i] in student_ids: 
                student_ids.remove(self.bad_ids[i])
        return student_ids

    def get_segment_info(self, year, segment, student_ids=None):
        """
        Returns the segment info for the provide inputs as a list of tuples (start_time, end_time)
        Args:
            year:           string, which year
            segment:        string, which segment
            student_ids:    list, containing the student ids., if empty we compute it within this function
        """
        annotations_folder = self.get_anno_folder_path(year)
        segment_data = []
        if student_ids is None:
            student_ids = self.scan_student_ids(year)
        for student_id in student_ids:
            segment_file_path = os.path.join(annotations_folder, str(student_id), str(student_id) + '_segment.txt')
            if os.path.exists(segment_file_path):
                file_info = [
                    line.rstrip('\n') for line in open(segment_file_path, 'r')
                ]
                segment_info = file_info[segment]
                try:
                    to_floats = list(map(float, segment_info.split('\t')))
                except:
                    to_floats = list(map(float, segment_info.split(' ')))
                # convert to tuple and append
                segment_data.append((to_floats[0], to_floats[0] + to_floats[1]))
            else:
                segment_data.append(None)
                #print(segment_file_path)
        return segment_data

    def get_pitch_contours_segment(self, year, segment_info, student_ids=None):
        """
        Returns the pitch contours for the provide inputs as a list of np arrays
                assumes pyin pitch contours have already been computed and stored as text files
        Args:
            year:           string, which year
            segment_info:       string, which segment
            student_ids:    list, containing the student ids., if empty we compute it within this function
        """
        data_folder = self.get_anno_folder_path(year)
        if student_ids is None:
            student_ids = self.scan_student_ids(year)
        pitch_contour_data = []

        for idx, student_id in enumerate(student_ids):
            pyin_file_path = os.path.join(data_folder, str(student_id), str(student_id) + '_pyin_pitchtrack.txt')
            if not os.path.exists(pyin_file_path) and segment_info[idx] is not None:
                self.compute_and_save_pitch_contour(year, student_id, save_path=pyin_file_path)
            lines = [line.rstrip('\n') for line in open(pyin_file_path, 'r')]
            pitch_contour = []
            start_time, end_time = segment_info[idx]
            if segment_info[idx] is not None:
                for x in lines:
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
            else:
                pitch_contour_data.append(None)

        return pitch_contour_data

    def _create_sonic_annotator_command(self, path_to_audio_file):
        output_dir = os.path.dirname(self.path_to_pyin_n3)
        command_list = [
            os.path.join(self.path_to_sonic_annotator, 'sonic-annotator'),
            '-t',
            self.path_to_pyin_n3,
            path_to_audio_file,
            '-w',
            'csv',
            '--csv-force',
            '--csv-basedir',
            output_dir + '/'
        ]
        command = " ".join(command_list)
        path_to_output_file = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(path_to_audio_file))[0] + '_vamp_pyin_pyin_smoothedpitchtrack.csv'
        )
        return command, path_to_output_file

    def run_sonic_annotator(self, year, student_id, save_path):
        # run sonic annotator to compute pYin contour as a .csv file
        path_to_audio_file = self.get_audio_file_path(year, [student_id])[0]
        command, path_to_temp_file = self._create_sonic_annotator_command(path_to_audio_file)
        a = os.system(command)
        # Need to check if the process complerted correctly
        # Sonic annotator sometimes does not write files correctly
        # If not then we try again !!!
        if a != 0:
            self.run_sonic_annotator(year, student_id, save_path)
        return path_to_temp_file

    def compute_and_save_pitch_contour(self, year, student_id, save_path):
        # run sonic annotator to compute pYin contour as a .csv file
        path_to_temp_file = self.run_sonic_annotator(year, student_id, save_path)

        # read temp file created by sonic annotator
        pyin_data = pd.DataFrame.to_numpy(pd.read_csv(path_to_temp_file))
        pyin_f0 = pyin_data[:, 1]
        pyin_ts = np.round(pyin_data[:, 0], 3)

        # generate timestamps and readjust pitch contour to include unvoiced blocks also
        path_to_audio_file = self.get_audio_file_path(year, [student_id])[0]
        y, sr = librosa.load(path_to_audio_file, sr=44100, mono=True)
        window_size = 1024
        hop = 256
        num_blocks = np.ceil(y.shape[0] / hop)
        time_stamps = np.round(np.arange(0, num_blocks) * hop / sr, 3)
        pyin_f0_rev = np.zeros_like(time_stamps)
        a, _, c = np.intersect1d(pyin_ts, time_stamps, return_indices=True)
        pyin_f0_rev[c] = pyin_f0[: min(c.size, pyin_f0.size)]

        # save pitch contour
        output_lines = [str(time_stamps[i]) + ',' + str(pyin_f0_rev[i]) + '\n' for i in range(time_stamps.size)]
        with open(save_path, 'w') as outfile:
            outfile.writelines(output_lines)

        # delete temp file
        os.remove(path_to_temp_file)

        return pyin_f0

    def get_audio_file_path(self, year, student_ids=None):
        """
        Returns the audio paths for the provide inputs as a list of strings
        Args:
            year:           string, which year
            student_ids:    list, containing the student ids., if empty we compute it within this function
        """
        data_folder = self.get_audio_folder_path(year)
        if student_ids is None:
            student_ids = self.scan_student_ids(year)
        audio_file_paths = []
        for student_id in student_ids:
            audio_file_path = os.path.join(
                data_folder, str(student_id), str(student_id) + '.mp3'
            )
            #print(audio_file_path)
            if os.path.exists(audio_file_path):
                audio_file_paths.append(audio_file_path)
            else:
                audio_file_paths.append(None)
        return audio_file_paths

    def get_perf_rating_segment(self, year, segment, student_ids=None):
        """
        Returns the performane ratings given by human judges for the input segment as a list of tuples
        Args:
            year:           string, which year
            segment:        string, which segment
            student_ids:    list, containing the student ids., if empty we compute it within this function
        """
        # get student_ids if needed
        if student_ids is None:
            student_ids = self.scan_student_ids(year)

        # read assessment csv file
        path_to_assessments_file = os.path.join(self.path_to_annotations, self.assessments_file)
        assessments_data = pd.read_csv(path_to_assessments_file)

        # extract assessment ratings
        perf_ratings = []
        seg, criteria_list = SEGMENT_CRITERIA_DICT[segment]
        for student_id in student_ids:
            assessment = assessments_data.loc[
                (assessments_data['student_id'] == int(student_id)) &
                (assessments_data['year'] == int(year))
            ]
            # student_id and year combination must be unique
            assert assessment.shape[0] == 1
            ratings = []
            for criteria in criteria_list:
                assessment_str = seg + '_' + criteria
                ratings.append(assessment[assessment_str].values[0])
            perf_ratings.append(tuple(ratings))

        return perf_ratings
    
    def create_data(self, year, segment, include_audio=False):
        """
        Creates the data representation for a particular year
        Args:
            year:           string, which year
            segment:        string, which segment
            include_audio:  bool, include the audio files in the data if True
        """
        perf_assessment_data = []
        student_ids = self.scan_student_ids(year)
        segment_info = self.get_segment_info(year, segment, student_ids)
        pitch_contour_data = self.get_pitch_contours_segment(year, segment_info, student_ids)
        ground_truth = self.get_perf_rating_segment(year, segment, student_ids)
        counter_audio = 0
        counter_seg = 0
        for student_idx in range(len(student_ids)):
            assessment_data = {}
            assessment_data['year'] = year
            assessment_data['band'] = self.band
            assessment_data['instrumemt'] = self.instrument
            assessment_data['student_id'] = student_ids[student_idx]
            assessment_data['segment'] = segment
            if not include_audio:
                if pitch_contour_data[student_idx] is None:
                    continue
                assessment_data['pitch_contour'] = pitch_contour_data[student_idx]
            else:
                audio_file_paths = self.get_audio_file_path(year, student_ids)
                if segment_info[student_idx] is None:
                    counter_seg += 1
                    # print(f'Missing Seg Info for: {self.band}, {year}, {self.instrument}, {student_ids[student_idx]}')
                    continue
                elif audio_file_paths[student_idx] is None:
                    counter_audio += 1
                    # print(f'Missing Audio file for: {self.band}, {self.year}, {self.instrument}, {student_idx}')
                    continue
                # print(student_idx)
                y, sr = librosa.load(
                    audio_file_paths[student_idx],
                    offset=segment_info[student_idx][0],
                    duration=segment_info[student_idx][1] - segment_info[student_idx][0]
                )
                assessment_data['audio'] = (y, sr)
            assessment_data['ratings'] = ground_truth[student_idx]
            assessment_data['class_ratings'] = [round(x * 10) for x in ground_truth[student_idx]]
            perf_assessment_data.append(assessment_data)
            # print(counter_seg, counter_audio)
        return perf_assessment_data
