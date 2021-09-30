import pandas as pd

class Data:
    def __init__(self,filename):
        self.file_name = filename
    def read_file(self):
        df_meta = pd.read_csv(self.file_name)
        return df_meta
    def create_feature(self,df_meta):
        #split COVID STATUS column to get labels in column 'split'
        df_meta['split'] = df_meta['COVID_STATUS'].str.split('_').str.get(0)
        #Generate a dict to re-categorize the split column
        cat_dict = {'healthy':0,'no':0,'resp':0,'recovered':0,'positive':1}
        #map cat_dict to split column 
        df_meta.loc[:,'split'] =  df_meta.loc[:,'split'].map(cat_dict)
        #df_meta2 = df_meta.dropna(subset=['split'])
        df_meta2 = df_meta.dropna()
        df_meta2.loc[:,'split'] = df_meta2.loc[:,'split'].astype('int32')
        return df_meta2
    def create_df(self):
        df_meta = self.read_file()
        df_meta2 = self.create_feature(df_meta)
        return df_meta2
    