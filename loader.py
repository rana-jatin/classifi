import os
import shutil
from PIL import Image


def df_load(img,label,num):
  """Creates a folder with the given label and saves the provided PIL Image object into it.

  Args:
    img: A PIL Image object.
    label: The label to use for the folder name.
  """
  try:
    os.makedirs(label, exist_ok=True)  # Create the folder if it doesn't exist

    # Construct the full path for saving the image
    img_path = os.path.join(label, "image"+(str(num))+".png")  # You can change the filename if needed
    num = num + 1
    # Save the PIL Image object
    img.save(img_path) 

    print(f"Image saved to '{label}' folder successfully.")
  except Exception as e:
    print(f"An error occurred: {e}")

# Example usage:
