import pandas as pd
from sklearn.model_selection import train_test_split


# TODO: How to remove  first useless lines of file
class DatasetFactory(object):

    def __init__(self, file_array):
        # PreProcessedData array
        self.data = []
        for path in file_array:
            self.data.append(PreProcessData(path, False))


class PreProcessData(object):

    def __init__(self, path, with_vector):
        self.path = path
        self.df = self.parse_file(with_vector)
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
    def parse_file(self, vector=True):
        df = pd.read_csv(self.path, delimiter=",", engine="python", header=None)
        # df.columns = ['index', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6',
        #               'channel7', 'channel8', 'x', 'y', 'z', 'timestamp']
        # TODO: Add the following once ready, 'temp', 'class']

        df.drop([0, 12], axis=1, inplace=True)
        if not vector:
            df.drop([11,  10, 9], axis=1, inplace=True)
        return df

    # def set_class(self, df):
    #     self.add_attribute()

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
        self.df = self.df.Series(index=[0,1,2,3,4,5,6,7]).convert_objects(convert_numeric=True)
        print(self.df)
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

    def __init__(self, path, class_value):
        df = pd.read_csv(path, delimiter=", ", engine="python", header=None)
        df.columns = ['index', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6',
                      'channel7', 'channel8', 'x', 'y', 'z', 'timestamp']
        df['class'] = pd.Series(class_value, index=df.index)
        df.to_csv(path, index=False)
ClassifyCsv('ice.csv', 'True')
ClassifyCsv('no_ice.csv', 'False')
