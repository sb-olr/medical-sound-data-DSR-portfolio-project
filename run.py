from data_prepration import Data
from train import Train

#names_input = ['counting-normal','counting-fast','breathing-deep','breathing-shallow','cough-heavy','cough-shallow','vowel-a','vowel-e','vowel-o']

# Config
file_name = (
    "./CoronaHack-Respiratory-Sound-Dataset/Corona-Hack-Respiratory-Sound-Metadata.csv"
)
base_path = "./CoronaHack-Respiratory-Sound-Dataset"
names = ["counting-fast"]
name_labels = "split"
image_target_height = 28
image_target_width = 28
model_name = "vae"  # "autoencoder"
latent_dim = 2
batch_size = 64
epochs = 200
adam_learning_rate = 0.0001
# read meta data
data_obj = Data(filename=file_name)
df_meta = data_obj.create_df()
# train
train_obj = Train(
    base_path=base_path,
    df=df_meta,
    names=names,
    name_labels=name_labels,
    image_target_height=image_target_height,
    image_target_width=image_target_width,
)

history = train_obj.train(
    model_name=model_name,
    latent_dim=latent_dim,
    learning_rate=adam_learning_rate,
    batch_size=batch_size,
    epochs=epochs,
)
