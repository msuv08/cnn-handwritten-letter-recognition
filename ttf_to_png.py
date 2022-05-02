#!/usr/bin/env python
import sys
import os
import subprocess
import shutil
import glob

from PIL import Image, ImageOps
from fontTools.ttLib import TTFont

TEXTS_DIR = "texts"
IMAGES_DIR = "images"
# Get path using sys args, set up font loader
FINAL_PATH = sys.argv[1]
for file in os.listdir(FINAL_PATH):
    TTF_PATH = FINAL_PATH + file
    FONT_SIZE = sys.argv[2]
    TTF_NAME, TTF_EXT = os.path.splitext(os.path.basename(TTF_PATH))
    ttf = TTFont(TTF_PATH, 0, verbose=0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
    # Make temporary directories
    for d in [TEXTS_DIR, IMAGES_DIR]:
        if not os.path.isdir(d):
            os.mkdir(d)
    # Writing each individual font character to file as texts
    for x in ttf["cmap"].tables:
        for y in x.cmap.items():
            char_unicode = chr(y[0])
            char_utf8 = char_unicode.encode('utf_8')
            char_name = y[1]
            f = open(os.path.join(TEXTS_DIR, char_name + '.txt'), 'wb')
            f.write(char_utf8)
            f.close()
    ttf.close()
    # Writing each text as a png through converting it
    files = os.listdir(TEXTS_DIR)
    for filename in files:
        name, ext = os.path.splitext(filename)
        input_txt = TEXTS_DIR + "/" + filename
        output_png = IMAGES_DIR + "/" + TTF_NAME + "_" + name + "_" + FONT_SIZE + ".png"
        subprocess.call(["convert", "-font", TTF_PATH, "-pointsize", FONT_SIZE, "label:@" + input_txt, output_png])

    print("TTF -> Image Conversion Finished")

    # Normalizing the images to all be of the correct size
    src_dir = 'images'
    dst_dir = 'cropped_images'
    thumb_width = 28

    files = glob.glob(os.path.join(src_dir, '*.png'))

    for f in files:
        im = Image.open(f).convert('RGB')
        im = ImageOps.invert(im)
        im_thumb = im.resize((thumb_width, thumb_width), Image.LANCZOS)
        ftitle, fext = os.path.splitext(os.path.basename(f))
        # Ommitting all non-readable UTF characters
        if "CR" in ftitle:
            continue
        elif "nonmarkingreturn" in ftitle:
            continue
        elif "nbsp" in ftitle:
            continue
        elif "space" in ftitle:
            continue
        else:
            im_thumb.save(os.path.join(dst_dir, ftitle + '_thumbnail' + fext), quality=95)
            
    print("Images -> Cropped Images Finished")
    # Deleting temporary directories
    for d in [TEXTS_DIR, IMAGES_DIR]:
        shutil.rmtree(d)
