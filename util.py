import cv2
from pathlib import Path
import sys
import os
import torch
from torch import nn
from torchvision import models
import numpy as np

def split_video(pathIn, pathOut, startFrame, endFrame):
    # check the existence of file
    if os.path.isfile(Path(pathOut)):
        return
    print(pathOut)
    # split a duration from the video
    video = cv2.VideoCapture(pathIn)
    fps = int(video.get(cv2.CAP_PROP_FPS))+1
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outVideo = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for i in range(startFrame, endFrame):
        video.set(1, i)
        ret, still = video.read()
        outVideo.write(still)
    outVideo.release()

def get_sequence(path, startSec, endSec):
    video = cv2.VideoCapture(str(path))
    fps = int(video.get(cv2.CAP_PROP_FPS))+1
    pathOut = str(path.parent) + '/' + str(fps * startSec) + '_' + str(fps * (endSec - startSec)) + '.avi'
    split_video(str(path), pathOut, fps*startSec, fps*endSec)

def write_vid_frames(path):
# export frames from video
    vids = list(sorted(path.glob("*.mp4")))  + list(sorted(path.glob("*.avi")))
    for v in vids:
        cap = cv2.VideoCapture(str(v))

        fpath = str(v.parents[1]) + '/JPEGImages/' + v.stem
        if not os.path.exists(fpath):
            os.makedirs(fpath, exist_ok=False)
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            cv2.imwrite(fpath + '/' + str(i).zfill(5) + '.jpg', frame)
            i += 1
        cap.release()

def merge_mask(frames, masks, out):
    imgs = list(sorted(frames.glob("*.png"))) + list(sorted(frames.glob("*.jpg")))
    img = cv2.imread(str(masks) + '/00000.png')
    h, w, l = img.shape
    size = (w, h)
    dstVideo = cv2.VideoWriter(str(out) + '/segmentedVideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
    for img in imgs:
        t1 = str(frames) + '/' + img.name
        t2 = str(masks) + '/' + img.stem + '.png'
        src1 = cv2.imread(t1)
        src2 = cv2.imread(t2)
        dst = cv2.addWeighted(src2, 0.5, src1, 0.5, 0)
        cv2.imwrite(str(out) + '/' + img.name, dst)

        dstVideo.write(dst)
    dstVideo.release()

def combine_frames_into_video():
    folder_name = "parrot_out"
    out = "/home/trung/Documents/projects/TransVOS/dataset/" + folder_name
    for i in range(1, 7):
        dir = str(i) + "/overlay"
        path = "/home/trung/Documents/projects/TransVOS/dataset/" + folder_name + "/"
        frames = Path(__file__).parent / str(path + dir)
        imgs = list(sorted(frames.glob("*.png"))) + list(sorted(frames.glob("*.jpg")))
        img = cv2.imread(str(imgs[0]))
        h, w, l = img.shape
        size = (w, h)
        if i==1:
            dstVideo = cv2.VideoWriter(str(out) + '/segmentedVideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
            print(str(out) + '/segmentedVideo.avi')
        for img in imgs:
            image = cv2.imread(str(frames) + '/' + img.name)
            dstVideo.write(image)
    dstVideo.release()

def t(h,m,s):
    return (h*60+m)*60+s

def convert_seq_series_video(folders, parent_folder, masks, out):
    img = cv2.imread(str(masks) + '/' + str(folders[0]) + '/00000.png')
    h, w, l = img.shape
    size = (w, h)
    dstVideo = cv2.VideoWriter(str(out) + '/segmentedVideo.avi', cv2.VideoWriter_fourcc(*'DIVX'), 24, size)
    for i in range(len(folders)):
        print(folders[i])
        frames = folders[i]
        fol = str(parent_folder) + "/" + str(folders[i])
        frames = Path(__file__).parent / fol
        imgs = list(sorted(frames.glob("*.png"))) + list(sorted(frames.glob("*.jpg")))
        for img in imgs:
            t1 = str(frames) + '/' + img.name
            t2 = str(masks) + '/' + folders[i] + '/' + img.stem + '.png'
            src1 = cv2.imread(t1)
            src2 = cv2.imread(t2)
            dst = cv2.addWeighted(src2, 0.5, src1, 0.5, 0)
            #cv2.imwrite(str(out) + '/' + img.name, dst)

            dstVideo.write(dst)
    dstVideo.release()

if __name__ == '__main__':
    p = str(Path(__file__).absolute().parents[2])
    if p not in sys.path:
        sys.path.append(p)

    p = Path(__file__).parent / "dataset/underwater_data/Videos"
    # generate frames from video
    #write_vid_frames(p)

    #convert frames from many folders to a video
    folders = ["24_24", "48_72", "120_72", "192_48", "240_48", "288_96",
               "384_48","432_48","480_72","552_72", "624_48","672_72","744_72",
               "816_48","864_96","960_48"]
    m_path = Path(__file__).parent / "output/own_datavalid-rn101_ytvos_fast/"
    f_path = Path(__file__).parent / "dataset/underwater_data/JPEGImages/"
    out = Path(__file__).parent / "output/own_datavalid-rn101_ytvos_fast/"
    #convert_seq_series_video(folders, f_path, m_path, out)


    #get sequences from video
    #p = Path("/home/trung/Documents/projects/MiVOS/MiVOS/underwater/video3.mp4")
    p = Path("/home/trung/Documents/projects/TransVOS/dataset/apple_sharp_island/Apple_Sharp Island.mp4")

    durations = [[t(0,0,0),t(0,0,1)],[t(0,0,1),t(0,0,2)],[t(0,0,2),t(0,0,5)]
        , [t(0,0,5), t(0,0,8)],[t(0,0,8),t(0,0,10)]
        , [t(0, 0, 10), t(0, 0, 12)], [t(0, 0, 12), t(0, 0, 16)]
        , [t(0, 0, 16), t(0, 0, 18)], [t(0, 0, 18), t(0, 0, 20)]
        , [t(0, 0, 20), t(0, 0, 23)], [t(0, 0, 23), t(0, 0, 26)]
        , [t(0, 0, 26), t(0, 0, 28)], [t(0, 0, 28), t(0, 0, 31)]
        , [t(0, 0, 31), t(0, 0, 34)], [t(0, 0, 34), t(0, 0, 36)]
        , [t(0, 0, 36), t(0, 0, 40)], [t(0, 0, 40), t(0, 0, 42)]
                 ]
    # for d in durations:
    #     get_sequence(p, d[0], d[1])

    #merge masks and frames
    scenes = ['Apple_Sharp Island']
    for scene in scenes:
        #print(scene)
        m_path = Path(__file__).parent / str("output/own_datavalid-rn101_ytvos_fast/" + scene)
        f_path = Path(__file__).parent / str("dataset/underwater_data/JPEGImages/" + scene)
        out = Path(__file__).parent / str("output/own_datavalid-rn101_ytvos_fast/merge/" + scene)
        if not os.path.exists(out):
            os.makedirs(out, exist_ok=False)

        #merge_mask(f_path, m_path, out)

    #combine_frames_into_video()
    checkpoint = torch.load("/home/trung/Documents/projects/TransVOS/weights/dense_ep0140.pth",map_location='cuda:0')