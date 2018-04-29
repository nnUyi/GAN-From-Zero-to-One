# coding='utf-8'
'''
author: Youzhao Yang
 date:  04/29/2018
github: https://github.com/nnUyi

Requirements:
    python3.*
    PIL
    numpy
'''

'''
    PIL: python imaging library
    mainly derived from: http://pillow.readthedocs.io/en/3.1.x/handbook/tutorial.html
'''

from PIL import Image
import numpy as np

'''
-----------------------------------------------------------
                        Concept
-----------------------------------------------------------
# Modes
    The mode of an image defines the type and depth of a pixel in the image. 
    The current release supports the following standard modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, black and white)
        P (8-bit pixels, mapped to any other mode using a color palette)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        LAB (3x8-bit pixels, the L*a*b color space)
        HSV (3x8-bit pixels, Hue, Saturation, Value color space)
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)

# Size
    You can read the image size through the size attribute. 
    This is a 2-tuple, containing the horizontal and vertical size in pixels.

# Coordinate System
    The Python Imaging Library uses a Cartesian pixel coordinate system, 
    with (0,0) in the upper left corner. Note that the coordinates refer to 
    the implied pixel corners; the centre of a pixel addressed as (0, 0) actually lies at (0.5, 0.5).

    Coordinates are usually passed to the library as 2-tuples (x, y). 
    Rectangles are represented as 4-tuples, with the upper left corner given first. 
    For example, a rectangle covering all of an 800x600 pixel image is written as (0, 0, 800, 600).
'''

'''
-----------------------------------------------------------
                        Tutorial
-----------------------------------------------------------
        Using the Image class
        Reading and writing images
        Cutting, pasting, and merging images
        Geometrical transforms
        Color transforms
        Image enhancement
        Image sequences
        More on reading images
-----------------------------------------------------------
'''

infile = './data/test.png'
outfile = './data/test_duplicate.jpeg'

# Using the Image class
im = Image.open(infile)
print('format:{}, size:{}, mode:{}'.format(im.format, im.size, im.mode))

# Reading and writing images
## convert to jpeg format
try:
    im.save(outfile)
    im_ = Image.open(outfile)
except IOError:
    print('fail to convert')
print('format:{}, size:{}, mode:{}'.format(im_.format, im_.size, im_.mode))

# Cutting, pasting, and merging images
## Copying a subrectangle from an image
box = (100,100,400,400)
region = im.crop(box)

## Processing a subrectangle, and pasting it back
region = region.transpose(Image.ROTATE_180)
im.paste(region, box)

## Splitting and merging bands
r,g,b = im.split()
im = Image.merge('RGB', (r,g,b))

# Geometrical transforms
## Simple geometry transforms
out = im.resize((800,500))
out = im.rotate(45)

## Transposing an image
out = im.transpose(Image.FLIP_LEFT_RIGHT)
out = im.transpose(Image.FLIP_TOP_BOTTOM)
out = im.transpose(Image.ROTATE_90)
out = im.transpose(Image.ROTATE_180)
out = im.transpose(Image.ROTATE_270)

# Color transforms
## Converting between modes
im_1 = Image.open(outfile).convert('L')

# Image enhancement
## Filters
from PIL import ImageFilter
'''
    im1 = im.filter(ImageFilter.BLUR)
    im2 = im.filter(ImageFilter.MinFilter(3))
    im3 = im.filter(ImageFilter.MinFilter) # same as MinFilter(3)
    Filters
            BLUR,
            CONTOUR, 
            DETAIL, 
            EDGE_ENHANCE, 
            EDGE_ENHANCE_MORE, 
            EMBOSS, 
            FIND_EDGES, 
            SMOOTH, 
            SMOOTH_MORE, 
            SHARPEN.

    Kernel
        Kernel(size, kernel, scale=None, offset=0)
            RankFilter(size, rank)
            MinFilter(size=3)
            MedianFilter(size=3)
            MaxFilter(size=3)
            ModeFilter(size=3)
    
    derived from: http://www.effbot.org/imagingbook/imagefilter.htm
'''
out = im.filter(ImageFilter.DETAIL)

## Enhancement
from PIL import ImageEnhance
'''
    ImageEnhance.Color
    ImageEnhance.Contrast
    ImageEnhance.Brightness
    ImageEnhance.Sharpness
'''
enh = ImageEnhance.Sharpness(im)
enh.enhance(30)

# Image sequences
'''
im = Image.open("animation.gif")
im.seek(1)                                              # skip to the second frame

try:
    while 1:
        im.seek(im.tell()+1)
        # do something to im
except EOFError:
    pass # end of sequence
'''

# More on reading images
## Reading from a tar archive
from PIL import TarIO
fp = TarIO.TarIO('data.tar', 'data/test.png')
im_tar = Image.open(fp)

# PIL to numpy
## PIL to numpy array
im_ndarray = np.array(im_tar)
## numpy array to PIL
im = Image.fromarray(im_ndarray)
im.show()
