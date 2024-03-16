import os
from PIL import Image

def separate_and_mirror(image_path):
    # Open the image
    image = Image.open(image_path)
    width, height = image.size

    # Calculate the middle point
    middle = width // 2

    # Separate the image into left and right halves
    left_half = image.crop((0, 0, middle, height))
    right_half = image.crop((middle, 0, width, height))

    # Mirror the right half
    mirrored_right_half = right_half.transpose(Image.FLIP_LEFT_RIGHT)
    return left_half, mirrored_right_half


def process_folder(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".PNG"):
            input_image_path = os.path.join(input_folder, filename)
            left_half, mirrored_right_half = separate_and_mirror(input_image_path)

            # Save the resulting images
            left_half.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_left_half.png"))
            mirrored_right_half.save(os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_mirrored_right_half.png"))


def main():
    input_folder = "/workspaces/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/input_folder/Segmented_Images"
    output_folder = "/workspaces/Accurate-localization-of-key-points-Scleral-Spur-based-on-VGG19-on-ACA-and-measurement-of-relevant/output_folder"
    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
