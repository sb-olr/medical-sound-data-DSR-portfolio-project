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
        df_meta = df_meta.dropna()
        df_meta.loc[:,'split'] = df_meta.loc[:,'split'].astype('int32')

        return df_meta

    def create_df(self):
        df_meta = self.read_file()
        df_meta = self.create_feature(df_meta)

        # Separate Positive and Negative cases
        df_meta_positives = df_meta[df_meta['split'] == 1]
        df_meta_negatives = df_meta[df_meta['split'] == 0]

        # Create Training set - with only negative cases and Test set - with mix of positive and negative cases
        train_set = df_meta_negatives[len(df_meta_positives):]
        test_set = pd.concat([df_meta_negatives[:len(df_meta_positives)], df_meta_positives])


        return train_set, test_set
    