import openslide
import os

cancer_img_dir = os.getcwd() + "/data/cancer_img/"
control_img_dir = os.getcwd() + "/data/control_img/"
training_files = os.listdir(cancer_img_dir)

for i in range(1):
    wsi = openslide.OpenSlide(filename=cancer_img_dir + training_files[i])
    print "wsi.level_count", wsi.level_count
    print "wsi.dimensions", wsi.dimensions
    print "wsi.level_dimensions", wsi.level_dimensions
    print "wsi.level_downsamples", wsi.level_downsamples
    print "wsi.properties", wsi.properties
    print "wsi.associated_images", wsi.associated_images
    region_image = wsi.read_region(location=(0, 0), level=0, size=(629, 1130))
    print region_image
