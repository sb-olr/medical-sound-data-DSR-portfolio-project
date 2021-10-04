from data_prepration import Data
from train import Train

#names_input = ['counting-normal','counting-fast','breathing-deep','breathing-shallow','cough-heavy','cough-shallow','vowel-a','vowel-e','vowel-o']



# Config
file_name = (
    "data/Corona-Hack-Respiratory-Sound-Metadata.csv"
)
base_path = "data/CoronaHack-Respiratory-Sound-Dataset"
names = ['vowel-e']
name_labels = "split"
image_target_height = 28
image_target_width = 28
model_name = "vae"   #"autoencoder" 
latent_dim = 2
batch_size = 64
epochs = 200
adam_learning_rate = 0.0001
# read meta data
data_obj = Data(filename=file_name)
train_df, test_df = data_obj.create_df()
# train
train_obj = Train(
    base_path = base_path,
    df = train_df,
    names = names,
    name_labels = name_labels,
    image_target_height = image_target_height,
    image_target_width = image_target_width,
)

history = train_obj.train(
     model_name = model_name,
     latent_dim = latent_dim,
     learning_rate = adam_learning_rate,
     batch_size = batch_size,
     epochs = epochs,
)


test_obj = Train(
    base_path = base_path,
    df = test_df,
    names = names,
    name_labels = name_labels,
    image_target_height = image_target_height,
    image_target_width = image_target_width,
)

