import pandas as pd
from PIL import Image
import numpy as np


def load_houses_data(path: str = './Houses-dataset/Houses Dataset/HouseInfo.txt') -> pd.DataFrame:
    houses_info = pd.read_csv(path, sep=r'\s+', header=None)
    houses_info.columns = ['n_bedrooms', 'n_bathrooms', 'area', 'zipcode', 'price']
    houses_info['img'] = houses_info.index + 1
    return houses_info


def min_max(series: pd.Series):
    return (series - series.min()) / (series.max() - series.min())


def preprocess_numerical_data(df: pd.DataFrame):
    df['n_bedrooms'] = min_max(df['n_bedrooms'])
    df['n_bathrooms'] = min_max(df['n_bathrooms'])
    df['area'] = min_max(df['area'])
    df['price'] = min_max(df['price'])
    return df[['n_bedrooms', 'n_bathrooms', 'area', 'price']]


def load_images(df: pd.DataFrame, path: str) -> np.ndarray:
    output_images = []
    for img_no in df['img']:
        input_images = []
        output_image = np.ones((64, 64, 3), dtype=float)
        for angle in ['bathroom', 'bedroom', 'frontal', 'kitchen']:
            img = Image.open(f'{path}/{img_no}_{angle}.jpg')
            img = img.resize((32, 32))
            img = np.array(img)
            input_images.append(img)
        output_image[:32, :32] = input_images[0]
        output_image[:32, 32:] = input_images[1]
        output_image[32:, 32:] = input_images[2]
        output_image[32:, :32] = input_images[3]
        output_images.append(output_image)
    return np.stack(output_images)
