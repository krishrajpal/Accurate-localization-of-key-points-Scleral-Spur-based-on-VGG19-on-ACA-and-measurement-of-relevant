#type: ignore
from PIL import Image
import numpy as np
import math

def bresenham_circle(center_x, center_y, radius):
    points = []
    x = radius
    y = 0
    decision = 1 - radius
    while x >= y:
        points.append((x + center_x, y + center_y))
        points.append((y + center_x, x + center_y))
        points.append((-x + center_x, y + center_y))
        points.append((-y + center_x, x + center_y))
        points.append((-x + center_x, -y + center_y))
        points.append((-y + center_x, -x + center_y))
        points.append((x + center_x, -y + center_y))
        points.append((y + center_x, -x + center_y))
        y += 1
        if decision <= 0:
            decision += 2 * y + 1
        else:
            x -= 1
            decision += 2 * (y - x) + 1
    return points

def count_green_pixels(image, center_x, center_y, radius):
    pixels = image.load()
    width, height = image.size
    green_pixel_count = 0
    points = bresenham_circle(center_x, center_y, radius)
    for point in points:
        x, y = point
        # Convert x and y to integers
        x = int(x)
        y = int(y)
        if 0 <= x < width and 0 <= y < height:
            if pixels[x, y] == (0,255,0,255): # Green pixel
                green_pixel_count += 1
    return green_pixel_count

def count_green_pixels_in_circle(image, center_x, center_y, radius):
    pixels = image.load()
    width, height = image.size
    # Initialize count
    green_pixel_count = 0
    # Iterate over all pixels within the circle
    for x in range(center_x - radius, center_x + radius + 1):
        for y in range(center_y - radius, center_y + radius + 1):
            # Check if the pixel is within the circle
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                # Check if the pixel is within the image bounds
                if 0 <= x < width and 0 <= y < height:
                    # Check if the pixel is green
                    if pixels[x, y] == (0, 255, 0,255):  # Green pixel
                        green_pixel_count += 1
    
    return green_pixel_count