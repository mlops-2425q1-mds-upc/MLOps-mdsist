"""
File that parses parquet files to jpg images
"""

import pandas as pd
import os
from PIL import Image
import io

class ParquetJPGParser:
    def __init__(self, parquet_file, output_dir):
        """
        Initializes the class with the parquet file and output directory.
        
        Args:
            parquet_file: Path to the parquet file containing image data.
            output_dir: Directory where images will be saved.
        """
        self.parquet_file = str(parquet_file)
        self.output_dir = str(output_dir)
    
    def save_images(self):
        """
        Reads the parquet file and saves the images into the specified output directory.
        Images are saved in subdirectories corresponding to their labels.
        """
        df = pd.read_parquet(self.parquet_file)
        
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)  
        
        # Iterate through the dataframe rows and process images
        for index, row in df.iterrows():
            image_data = row['image']  # Image data
            label = row['label']  # Label

            # Decode and process image if image data is in bytes
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image_bytes = image_data['bytes']
                image = Image.open(io.BytesIO(image_bytes))  # Open the image from bytes

                # Create label subdirectory
                label_dir = os.path.join(self.output_dir, str(label))
                os.makedirs(label_dir, exist_ok=True)
                
                # Save image in .jpg format
                image_path = os.path.join(label_dir, f'{index}.jpg')
                image.save(image_path)
