#!/usr/bin/env python

from PIL import Image

import os, sys

def cmp(a, b):
    return (a > b) - (a < b)

def resizeImage(infile, output_dir="", size=(256,256)):
     outfile = os.path.splitext(infile)[0]+"_resized"
     extension = os.path.splitext(infile)[1]

     if (cmp(extension, ".png")):
        return

     if infile != outfile:
        try :
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(output_dir+outfile+extension,"PNG")
        except IOError:
            print("cannot reduce image for ", infile)


if __name__=="__main__":
    output_dir = "../resized_256"
    dir = os.getcwd()
    print(dir)
    print("start rescale")

    if not os.path.exists(os.path.join(dir,output_dir)):
        os.mkdir(output_dir)
    for file in os.listdir(dir):
        resizeImage(file, output_dir)
    print("end rescale")
