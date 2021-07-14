#%%
import os

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import shutil
import glob

import matplotlib.image as mpimg
from PIL import Image

#we have large image so turnoff limits
Image.MAX_IMAGE_PIXELS = None

%matplotlib inline

#%%
from tensorflow import keras
#%%
raster_path = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/T36UXV_20200406T083559_TCI_10m.jp2'
with rasterio.open(raster_path, "r", driver="JP2OpenJPEG") as src:
    raster_img = src.read()
    raster_meta = src.meta
    
print(raster_meta)

#%%
raster_img = reshape_as_image(raster_img)
plt.figure(figsize=(15,15))
plt.imshow(raster_img)


# cv2.imwrite('/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Image_for_split/Image_for_split.png',cv2.cvtColor(raster_img, cv2.COLOR_BGR2RGB))


#%%  Reading train labels with GeoPandas
train_df = gpd.read_file("/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/masks/Masks_T36UXV_20190427.shp")
print(len(train_df))
# print(train_df.head())
new_id = np.arange(len(train_df))
#column if was = none, so i changed it to 0-936 

train_df['id'] = new_id
print(train_df.head())


#%% Trying to cut fields from Raster  and onverting GeoDataframe to raster CRS

src = rasterio.open(raster_path, 'r')
failed = []
for num, row in train_df.iterrows():
    try:
        masked_image, out_transform = rasterio.mask.mask(src, [mapping(row['geometry'])], crop=True, nodata=0)
    except:
        failed.append(num)
print("Rasterio failed to mask {} files".format(len(failed)))

#%% Checking coordinates of polygon using Polygon methods
print(train_df['geometry'][0].exterior.coords.xy)


#%%

train_df = gpd.read_file("/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/masks/Masks_T36UXV_20190427.shp")

new_id = np.arange(len(train_df))
#column if was = none, so i changed it to 0-936 

train_df['id'] = new_id
print(train_df.head())
# let's remove rows without geometry
train_df = train_df[train_df.geometry.notnull()]

# assigning crs
train_df.crs = {'init' :'epsg:4224'}   #top left corner - Poltava

#transforming polygons to the raster crs
train_df = train_df.to_crs({'init' : raster_meta['crs']['init']})



#%%Cutting fields from Raster

src = rasterio.open(raster_path, 'r', driver="JP2OpenJPEG")
outfolder = "/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/My_out/"
os.makedirs(outfolder, exist_ok=True)
failed = []

for num, row in train_df.iterrows():
    try:
        
        masked_image, out_transform = rasterio.mask.mask(src, [mapping(row['geometry'])], crop=True, nodata=0)
        img_image = reshape_as_image(masked_image)
        img_path = os.path.join(outfolder, str(row['id']) + '.png')
        img_image = cv2.cvtColor(img_image, cv2.COLOR_RGB2BGR)
        # plt.figure(figsize=(15,15))
        # plt.imshow(img_image)
        # print(img_path)
        cv2.imwrite(img_path,img_image)
    except Exception as e:
#         print(e)
        failed.append(num)
print("Rasterio failed to mask {} files".format(len(failed)))

#%% Prepearing binary mask


# rasterize works with polygons that are in image coordinate system

def poly_from_utm(polygon, transform):
    poly_pts = []
    
    # make a polygon from multipolygon
    poly = cascaded_union(polygon)
    for i in np.array(poly.exterior.coords):
        
        # transfrom polygon to image crs, using raster meta
        poly_pts.append(~transform * tuple(i))
        
    # make a shapely Polygon object
    new_poly = Polygon(poly_pts)
    return new_poly

# creating binary mask for field/not_filed segmentation.

poly_shp = []
im_size = (src.meta['height'], src.meta['width'])
for num, row in train_df.iterrows():
    if row['geometry'].geom_type == 'Polygon':
        poly = poly_from_utm(row['geometry'], src.meta['transform'])
        poly_shp.append(poly)
    else:
        for p in row['geometry']:
            poly = poly_from_utm(p, src.meta['transform'])
            poly_shp.append(poly)

mask = rasterize(shapes=poly_shp,
                 out_shape=im_size)

# plotting the mask

plt.figure(figsize=(15,15))
plt.imshow(mask)

bin_mask_meta = src.meta.copy()



bin_mask_meta.update({'count': 1})
print(bin_mask_meta)
# os.makedirs('Mask_for_split')
with rasterio.open("/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Mask_for_split/Mask_for_split.jp2", 'w', **bin_mask_meta) as dst:
    dst.write(mask * 255, 1)
    
#DataSet  preparing


#%%
#try to correcty safe binary mask in png
jp2_filename = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Mask_for_split/Mask_for_split.jp2'
with rasterio.open(jp2_filename) as infile:
    # print(f"\nnew profile: {print.pformat(infile.profile)}\n")
    profile=infile.profile
    #
    # change the driver name from GTiff to PNG
    #
    profile['driver']='PNG'
    #
    rast=infile.read()
    with rasterio.open('/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Mask_for_split/Mask_for_split.png', 'w', **profile) as dst:
        dst.write(rast)

#%%


#%%


#%%
#image splitter (with set size)
#creating a new directory and recursively deleting the contents of an existing one:
# creating a new directory and recursively deleting the contents of an existing one:
def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

#The crop function that goes over the original image are adjusted to the original image limit and contain the original pixels:
def crop(input_file, height, width):
    img = Image.open(input_file)
    img_width, img_height = img.size
    for i in range(img_height//height):
        for j in range(img_width//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield img.crop(box)
#function for splitting images and masks into smaller parts
def split(inp_img_dir, inp_msk_dir, out_dir, height, width, 
          start_num):
    image_dir = os.path.join(out_dir, 'images')
    mask_dir = os.path.join(out_dir, 'masks')
    dir_create(out_dir)
    dir_create(image_dir)
    dir_create(mask_dir)
    img_list = [f for f in
                os.listdir(inp_img_dir)
                if os.path.isfile(os.path.join(inp_img_dir, f))]
    file_num = 0
    for infile in img_list:
        infile_path = os.path.join(inp_img_dir, infile)
        for k, piece in enumerate(crop(infile_path,
                                       height, width), start_num):
            img = Image.new('RGB', (height, width), 255)
            img.paste(piece)
            img_path = os.path.join(image_dir, 
                                    infile.split('.')[0]+ '_'
                                    + str(k).zfill(5) + '.png')
            img.save(img_path)
        infile_path = os.path.join(inp_msk_dir,
                                   infile.split('.')[0] + '.png')
        for k, piece in enumerate(crop(infile_path,
                                       height, width), start_num):
            msk = Image.new('L', (height, width), 255)
            msk.paste(piece)
            msk_path = os.path.join(mask_dir,
                                    infile.split('.')[0] + '_'
                                    + str(k).zfill(5) + '.png')
            msk.save(msk_path)
        file_num += 1
        sys.stdout.write("\rFile %s was processed." % file_num)
        sys.stdout.flush()
#set the necessary variables:
inp_img_dir = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Image_for_split'
inp_msk_dir = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Mask_for_split'
out_dir = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Splitted'
height = 256
width = 256
start_num = 1




split(inp_img_dir, inp_msk_dir, out_dir, height, width, start_num)
#%%
import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

#data loading

def load_data(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "annotations/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
     path = path.decode()
     x = cv2.imread(path, cv2.IMREAD_COLOR)
     x = cv2.resize(x, (256, 256))
     x = x/255.0
     return x

def read_mask(path):
     path = path.decode()
     x = cv2.imread(path, 0)
     x = cv2.resize(x, (256, 256))
     x = x/255.0
     x = np.expand_dims(x, axis=-1)
     return x
 

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float64, tf.float64])
    x.set_shape([256, 256, 3])
    y.set_shape([256, 256, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    return dataset
#%%

#%%
path = '/media/mximus/2CC479EDC479B9A2/Python/Quantum Test/Test Tesk_Internship/Quantum internship/Splitted/'
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(path)



train_dataset = tf_dataset(train_x, train_y, batch=4)
print(train_dataset)
valid_dataset = tf_dataset(valid_x, valid_y, batch=4)
# dataset_train = {"train": train_dataset, "val": valid_dataset}
# print(dataset_train['train'])
#%%

#%%
from glob import glob

import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output

# -- Keras Functional API -- #
# -- UNet Implementation -- #
# Everything here is from tensorflow.keras.layers
# I imported tensorflow.keras.layers * to make it easier to read
dropout_rate = 0.5
input_size = (256, 256, 3)


initializer = 'he_normal'


# -- Encoder -- #
# Block encoder 1
inputs = Input(shape=input_size)
conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# Block encoder 2
max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# Block  encoder 3
max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

# Block  encoder 4
max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# -- Encoder -- #

# ----------- #
maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# ----------- #

# -- Decoder -- #
# Block decoder 1
up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# Block decoder 2
up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# Block decoder 3
up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# Block decoder 4
up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# -- Dencoder -- #

output = Conv2D(2, 1, activation = 'softmax')(conv_dec_4)

model = tf.keras.Model(inputs = inputs, outputs = output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


EPOCHS = 15

STEPS_PER_EPOCH = 1200 // 4
VALIDATION_STEPS = 500 // 4

# On GPU

model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=valid_dataset)