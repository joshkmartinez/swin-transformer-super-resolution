from PIL import Image
import os

def downsample_images(input_folder, output_folder, size=(256, 256)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image in the input folder
    for img_name in os.listdir(input_folder):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            with Image.open(img_path) as img:
                # Resize using bicubic interpolation
                img_resized = img.resize(size, Image.BICUBIC)
                
                # Save the resized image in the output folder
                img_resized.save(os.path.join(output_folder, img_name))

# Replace 'your_folder_path' with the path to your folder with 1024x1024 images
input_folder = '00000'
output_folder = os.path.join(input_folder, 'LR')

downsample_images(input_folder, output_folder)
