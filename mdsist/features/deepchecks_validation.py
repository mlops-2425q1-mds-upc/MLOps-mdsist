from mdsist.config import JPG_IMAGES_DIR
from mdsist.config import PROCESSED_DATA_DIR
import pandas as pd
import os
from PIL import Image
import numpy as np
import base64
import io

def save_images_from_parquet(parquet_file, output_dir):
    df = pd.read_parquet(parquet_file)
    
    # Create directory to store
    os.makedirs(output_dir, exist_ok=True)  
    
    for index, row in df.iterrows():
        image_data = row['image']  # Image data
        label = row['label']  # Label

        # Decode and process image
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))  # Open the image from bytes

            # Generate subdirectory
            label_dir = os.path.join(output_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            
            # Save image in .jpg format
            image_path = os.path.join(label_dir, f'{index}.jpg')
            image.save(image_path)

save_images_from_parquet(PROCESSED_DATA_DIR / "test.parquet", JPG_IMAGES_DIR / 'test')
save_images_from_parquet(PROCESSED_DATA_DIR / "train.parquet", JPG_IMAGES_DIR / 'train')
