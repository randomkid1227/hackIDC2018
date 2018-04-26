import pandas as pd
import datetime, math
import os
from Constants import Constants
from sklearn.model_selection import train_test_split


# TODO: How to remove  first useless lines of file
class DatasetFactory(object):
    def __init__(self, file_array):
        # PreProcessedData array
        self.data = []
        for file_entry in file_array:
            self.data.append(PreProcessData(file_entry["path"],
                                            file_entry["with_vector"],
                                            file_entry["time_to_remove_from_start"],
                                            file_entry["time_to_remove_from_end"]))

class PreProcessData(object):
    def __init__(self, path, with_vector, time_from="0.0", time_to="0.0"):
        self.path = path
        self.df = self.parse_file(with_vector, time_from, time_to)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.prepare_data()

    '''
    Removes attributes index and timestamp and returns a data set
    Takes boolean vector - True - Keep vector attributes (x,y,z)
                           False - Remove attributes [x,y,z]
    Returns a data set of the object's path without
    '''
    def parse_file(self, vector=True, time_from="0.0", time_to="0.0"):
        df = pd.read_csv(self.path, delimiter=",", engine="python")
        # df.columns = ['index', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6',
        #               'channel7', 'channel8', 'x', 'y', 'z', 'timestamp']
        # TODO: Add the following once ready, 'temp', 'class']
        self.filter_df_by_time(df, time_from, time_to)

        df.drop(['index', 'timestamp'], axis='columns', inplace=True)
        if not vector:
            df.drop(['x', 'y', 'z'], axis='columns', inplace=True)
        return df

    # def set_class(self, df):
    #     self.add_attribute()

    '''
        Filters a data frame to some delta of time (from,to)
    '''
    def filter_df_by_time(self, df, time_from="0.0", time_to="0.0"):
        def parse_time_to_datetime(time_string):
            return datetime.datetime.strptime(time_string, Constants.TIMESTAMP_FMT)

        def get_delta(time_start, time_to_remove_from_start, time_end, time_to_remove_from_end):
            t_from = datetime.datetime.strptime(time_to_remove_from_start, "%S.%f")
            t_to = datetime.datetime.strptime(time_to_remove_from_end, "%S.%f")
            s_time = time_start + datetime.timedelta(seconds=t_from.second, microseconds=t_from.microsecond)
            e_time = time_end - datetime.timedelta(seconds=t_to.second, microseconds=t_to.microsecond)
            return s_time, e_time

        start_time = parse_time_to_datetime(df['timestamp'][0])
        end_time = parse_time_to_datetime(df['timestamp'][len(df['timestamp']) - 1])

        delta_start, delta_end = get_delta(start_time, time_from, end_time, time_to)

        invalid_indices = []
        timestamps = df['timestamp']
        for i in range(len(timestamps)):
            current = parse_time_to_datetime(timestamps[i])
            if not (delta_start < current < delta_end):
                invalid_indices.append(i)

        df.drop(invalid_indices, axis='index', inplace=True)


    '''
    Allows to add an attribute to the data frame, most likely not needed.
    '''
    def add_attribute(self, df, att_name, att_val):
        df[att_name] = pd.Series(att_val, index=df.index)
        self.__set_df(df)

    def __set_df(self, data_frame):
        self.df = data_frame

    def __set_value(self, row, column, value):
        if row is None:
            self.df[column] = pd.Series(value, index=self.df.index)
        else:
            self.df.loc[row, column] = value

    '''
    Sets object's data frames according to file
    '''
    def prepare_data(self):
        features = self.df.values[:, :8]
        target = self.df.values[:, 8]
        # for row in features:
        #     print(features)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target,
                                                                                test_size=0.2,
                                                                                stratify=target,
                                                                                shuffle=True,
                                                                                random_state=10)

    def standardize(self):
        num_features = ['channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6',
                        'channel7', 'channel8']
        scaled_features = {}
        for each in num_features:
            mean, std = self.df[each].mean(), self.df[each].std()
            scaled_features[each] = [mean, std]
            self.df.loc[:, each] = (self.df[each] - mean) / std

    '''
    Adds a column with string value to csv
    '''


class ClassifyCsv(object):
    # TODO: Make sure to save files with different name.
    def __init__(self, path, class_value):
        df = pd.read_csv(path, delimiter=", ", engine="python", skiprows=6)
        df.columns = ['index', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'x', 'y', 'z', 'timestamp']
        df['class'] = pd.Series(class_value, index=df.index)
        df.to_csv("{}_with_class.csv".format(path.split(".")[0]), index=False)


def do_classify(foldername):
    for dirpath, dirnames, filenames in os.walk(foldername):
        for filename in filenames:
            if "_with_class.csv" in filename:
                continue
            ClassifyCsv("{}\\{}".format(dirpath, filename), "1" if "-k" in filename else "0")

do_classify("RawData")
