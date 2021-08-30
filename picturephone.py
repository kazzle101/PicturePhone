#!/bin/env python
# -*- coding: utf-8 -*-

# https://medium.com/swlh/decoding-noaa-satellite-images-using-50-lines-of-code-3c5d1d0a08da

# some metadata may upset the reading of the wav file. Clear this out with:
# $ sox PicturePhoneDataMonoXXX.wav PicturePhoneDataMono.wav

from os import waitpid
import scipy.io.wavfile as wav
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import figure, draw, pause
from PIL import Image, ImageEnhance, ImageOps, ImageChops
import os, shutil
import cv2
from progress.bar import Bar

WAVFILE = "PicturePhoneDataMono2b.wav"
pwd = os.path.dirname(os.path.realpath(__file__))
OUTIMGDIR = os.path.join(pwd, "output")
OUTVIDEO = os.path.join(pwd,'ppVideo.mp4')
OUTVIDEOSHIFT = os.path.join(pwd,'ppVideoShift.mp4')
OUTVIDEOSIZE = { 'width': 1280, 'height': 720}

# PCT-15 image sizes:
# 160(H) x 100 (V) dot (normal)
# 96(H) x 100 (V) dot (quick)
magnification = 5
horizontal, vertical = 96, 100
PCT15SIZE = (horizontal * magnification, vertical * magnification)

def deleteOutput():

    for filename in os.listdir(OUTIMGDIR):
        file_path = os.path.join(OUTIMGDIR, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
        
    return

def imgNo(imageName):
    basename = os.path.basename(imageName)
    image = os.path.splitext(basename)
    return image[0].split("-")[1]


def addVideoText(result, filename, width, height, sw, sh, showWH=True):

    if showWH:
        ts = (f"W:{sw}, H:{sh}")
        (x,y) = 4, height - 38
        (tw,th), baseline = cv2.getTextSize(ts, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
        cv2.rectangle(result, (x, y), (x + (tw + 14), y + (th + baseline)), (0,0,0), -1)
        cv2.putText(result, ts, (10, result.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    mt = imgNo(filename)
    (x,y) = width - 150, height - 38
    cv2.rectangle(result, (x, y), (width - 10 , (height - 4)), (0,0,0), -1)
    cv2.putText(result, mt, (x + 10, result.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    return result

def addToBlankImage(sframe, blankImage):

    bh, bw, channels = blankImage.shape
    sh, sw, channels = sframe.shape

    yoff = round((bh-sh)/2)
    xoff = round((bw-sw)/2)
    result = blankImage.copy()
    result[yoff:yoff+sh, xoff:xoff+sw] = sframe

    return result

def makeVideoFromImages():
    images = [img for img in os.listdir(OUTIMGDIR) if img.endswith(".png")]

    width, height = OUTVIDEOSIZE["width"], OUTVIDEOSIZE["height"]
    blankImage = np.zeros((height,width,3), np.uint8)

    #96(H) x 100 (V) dot (quick)
    magnification = 5
    horizontal, vertical = 96, 100
    sDim = (horizontal * magnification, vertical * magnification)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUTVIDEO, fourcc, 25, (width,height))

    for image in images:
        frame = cv2.imread(os.path.join(OUTIMGDIR, image))

        sframe = cv2.resize(frame, sDim, interpolation = cv2.INTER_LANCZOS4)
        sh, sw, channels = sframe.shape

        result = addToBlankImage(sframe, blankImage)
        result = addVideoText(result, image, width, height, sw, sh)

        video.write(result)

    cv2.destroyAllWindows()
    video.release()
    return

def makeVideoFromData(fs, data_am):

    widthStart=10
    widthEnd=3000

    # rendered video size
    width, height = OUTVIDEOSIZE["width"], OUTVIDEOSIZE["height"]
    blankImage = np.zeros((height,width,3), np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUTVIDEO, fourcc, 25, (width,height))

    with Bar('Processing...', max = widthEnd-widthStart) as bar:
        for w in range(widthStart, widthEnd, 1):
            pct15img = plotData(data_am, fs, w)
            frame = np.array(pct15img)           
            frame = frame[:, :, ::-1].copy()    # Convert RGB to BGR
            fh, fw, channels = frame.shape

            sframe = cv2.resize(frame, PCT15SIZE, interpolation = cv2.INTER_LANCZOS4)

            result = addToBlankImage(sframe, blankImage)

            filename = os.path.join(OUTIMGDIR, f"ssdraw-{w:05}.png")
            result = addVideoText(result, filename, width, height, fw, fh)

            video.write(result)
            bar.next()

    cv2.destroyAllWindows()
    video.release()
    return

def makeVideoShiftImageLeft(fs, data_am, pctWidth):

    width, height = OUTVIDEOSIZE["width"], OUTVIDEOSIZE["height"]
    blankImage = np.zeros((height,width,3), np.uint8)

    pct15img = plotData(data_am, fs, pctWidth)

    pw, ph = PCT15SIZE[0], PCT15SIZE[1]
    pct15img = pct15img.resize((pw, ph), Image.LANCZOS)
    #pct15img = ImageOps.invert(pct15img)

    h = plt.imshow(pct15img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(OUTVIDEOSHIFT, fourcc, 25, (width,height))

    for w in range(1, int(pw/2)):
        pct15img = ImageChops.offset(pct15img, -1, 0)
        # h.set_data(pct15img)
        # draw(), pause(1e-3)


        frame = np.array(pct15img)           
        frame = frame[:, :, ::-1].copy()    # Convert RGB to BGR
        fh, fw, channels = frame.shape

        result = addToBlankImage(frame, blankImage)

        filename = os.path.join(OUTIMGDIR, f"ssdraw-{pctWidth:05}.png")
        result = addVideoText(result, filename, width, height, fw, fh, showWH=False)

        video.write(result)

    cv2.destroyAllWindows()
    video.release()

    return




def hilbert(data):
    analytical_signal = signal.hilbert(data)
    amplitude_envelope = np.abs(analytical_signal)
    return amplitude_envelope



def plotData(data_am, fs, width):

    w, h = width, data_am.shape[0] // width

    print(w, h)

    image = Image.new('RGB', (w, h))
    px, py = 0, 0
    for p in range(data_am.shape[0] -1):

        lum = int(data_am[p]//128) # 32-32

        if lum < 0: 
            lum = 0
        if lum > 255: 
            lum = 255

        #                         r,   g,   b
        image.putpixel((px, py), (lum, lum, lum))    
        px += 1

        if px >= w:
            px = 0
            py += 1
            #print ("---")
            if py >= h:
                break

    enhancer = ImageEnhance.Brightness(image)
    factor = 6.5 #brightens the image
    image = enhancer.enhance(factor)

    return image

def main():

    fs, data = wav.read(WAVFILE)
    time = np.linspace(0, len(data) / fs, num=len(data))
    #data_crop = data[20*fs:21*fs]

    data_am = hilbert(data)

    # plt.specgram(data, NFFT=1024, Fs=fs)
    # plt.title("Spectrum")
    # plt.xlabel("Time, sec")
    # plt.ylabel("Frequency")
    # plt.show()

    # plt.figure(figsize=(20,6))
    # plt.plot(time, data) # data_crop
    # plt.plot(time, data_am)
    # plt.xlabel("Time, sec")
    # plt.ylabel("Amplitude")
    # plt.title("Signal")
    # plt.show()



    # if not os.path.isdir(OUTIMGDIR):
    #     os.mkdir(OUTIMGDIR)
    # else:
    #     deleteOutput()

  #  makeVideoFromData(fs, data_am)

    makeVideoShiftImageLeft(fs, data_am, 2422)

    # widthStart=80
    # widthEnd=2000
    # # image = plotData(data_am, fs, widthStart-1)
    # # h = plt.imshow(image)

    # with Bar('Processing...',max = widthEnd) as bar:
    #     for w in range(widthStart, widthEnd, 1):
    #         #print(f"width: {w}")
    #         image = plotData(data_am, fs, w)
    #         # h.set_data(image)
    #         # draw(), pause(1e-3)
    #         fileout = os.path.join(OUTIMGDIR, f"ssdraw-{w:05}.png")
    #         plt.imsave(fileout, image)
    #         bar.next()

    # makeVideoFromImages()

    return

if __name__ == "__main__":
    main()


