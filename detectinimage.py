from __future__ import division
import PIL
from PIL import Image, ImageDraw
import math
import os
import torchvision.transforms as tf
import torch
from settings import BATCH_SIZE,WORKERS
from dataset import UnknownDataset
import shutil
from shutil import copyfile

from settings import IMAGE_PATH, CONFIDENCE

step = 4
scale = 1.2
def long_slice(image_path, out_name, outdir, slice_size, net):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    imgout = Image.open(image_path)
    orw, orh = img.size
    width, height = img.size
    slicesh = int(math.ceil(height/slice_size))
    slicesw = int(math.ceil(width/slice_size))
    img = img.resize((slicesw*slice_size, slicesh*slice_size), PIL.Image.ANTIALIAS)
    imgout = imgout.resize((slicesw*slice_size, slicesh*slice_size), PIL.Image.ANTIALIAS)
    orw, orh = imgout.size
    width, height = img.size
    print(img.size)
    r = 1
    draw = ImageDraw.Draw(imgout)

    flag_continue = True
    while flag_continue:
        if os.path.exists("./testsliceimage/list.txt"):
            os.remove("./testsliceimage/list.txt")
        file = open("./testsliceimage/list.txt", "w+")
        for sliceh in range(slicesh*step):
            for slicew in range(slicesw*step):
                #set the bounding box! The important bit
                bbox = (int(slicew*slice_size/step), int(sliceh*slice_size/step), int(slicew*slice_size/step)+slice_size, int(sliceh*slice_size/step)+slice_size)
                working_slice = img.crop(bbox)

                working_slice.save(os.path.join(outdir, "slice_" + str(height) + "_" + str(width) + "_" + out_name + "_" + str(sliceh) + "_" + str(slicew) +".png"))
                file.write("slice_" + str(height) + "_" + str(width) + "_" + out_name + "_" + str(sliceh) + "_" + str(slicew) +".png\n")

                if sliceh == 16 and slicew == 27 and width == 450 :
                    print (int(slicew*slice_size/step), int(sliceh*slice_size/step),int(slicew*slice_size/step)+slice_size,int(sliceh*slice_size/step)+slice_size)

        file.close()
        transform_test = tf.Compose([tf.Grayscale(), tf.ToTensor(), tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        testset = UnknownDataset("./testsliceimage/", "./testsliceimage/list.txt", transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=WORKERS)

        with torch.no_grad():
            N = 0
            for data in testloader:
                images, img_names = data['image'], data['image_name']
                outputs = net(images.float())
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                if max(predicted) == 1 :
                    ite = -1
                    for predic in predicted :
                        ite += 1
                        if predic == 1 and outputs[ite][1]-outputs[ite][0] > CONFIDENCE:
                            print(img_names[ite])
                            # print(outputs)
                            N += 1
                            #dessiner carre sur image
                            slh = int(img_names[ite].split('_')[4])
                            slw = int(img_names[ite].split('_')[5][:-4])
                            x1 = int(slh * slice_size / step)
                            x2 = x1 + slice_size
                            y1 = int(slw * slice_size / step)
                            y2 = y1 + slice_size

                            if slh == 16 and slw == 27 and width ==450 :
                                print (x1, y1, x2, y2)

                            print(r)
                            rh = orh / height
                            rw = orw / width
                            x1 = x1 * rh
                            x2 = x2 * rh
                            y1 = y1 * rw
                            y2 = y2 * rw

                            draw.rectangle(((y1, x1), (y2, x2)), outline="red")
                            # draw.text((y2,x2), img_names[0])
                            copyfile("./testsliceimage/"+img_names[ite], "./goodimage/"+ img_names[ite])

        if width <= 200 or height <= 200:
            flag_continue = False
        else:
            r = r * scale
            width, height = int(width/scale), int(height/scale)
            slicesh = int(math.ceil(height/slice_size))
            slicesw = int(math.ceil(width/slice_size))
            img = img.resize((slicesw*slice_size, slicesh*slice_size), PIL.Image.ANTIALIAS)
            width, height = img.size

            # imgout = imgout.resize((slicesw*slice_size, slicesh*slice_size), PIL.Image.ANTIALIAS)
    imgout.save("./rectangle/out", "PNG")

def searchfaces(net):
    import shutil
    if os.path.exists("./testsliceimage"):
        shutil.rmtree("./testsliceimage")

    if not os.path.exists("./testsliceimage"):
        os.makedirs("./testsliceimage")

    #slice_size is the max height of the slices in pixels
    long_slice(IMAGE_PATH,"nasa", os.getcwd()+"/testsliceimage", 36, net)
