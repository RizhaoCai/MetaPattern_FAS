# Rizhao Version
import logging
logging.level = logging.ERROR

import os
import zipfile
import io
import cv2
from glob import glob
import zip_helper
from multiprocessing import Pool
# switch the SPLIT_SVM/SVM/CNN detector
import sys
sys.path.append('/home/rizhao/code/mxnet_mtcnn_face_detection/')
import mxnet_detector
import sys
from PIL import Image

ext_ratio = 0.0
DETECTOR = "SPLIT_SVM"


DATASET_DIR = {
    'OULU-NPU': '/home/Dataset/OULU-NPU/*/*.avi',  #
    'CASIA-FASD': '/home/Dataset/CASIA-FASD/*/*/*.avi',  #
    'REPLAY-ATTACK': '/home/Dataset/REPLAY-ATTACK/*/*/*.mov',  #
    'REPLAY-ATTACK-SPOOF': '/home/Dataset/REPLAY-ATTACK/*/*/*/*.mov',  #
    'SIW': '/home/Dataset/SIW/*/*/*/*.mov',
    'ROSE-YOUTU': '/home/Dataset/ROSE-YOUTU/*/*/*.mp4',  #
    'MSU-MFSD': '/home/Dataset/MSU-MFSD/scene01/*/*.mp4',
    'MSU-MFSD2': '/home/Dataset/MSU-MFSD/scene01/*/*.mov',
}
DATASET_DIR_ROOT = '/home/rizhao/data/FAS/'




def process_one_video(input_fn):
    # get the input_fn ext_ratio

    output_fn = os.path.relpath(input_fn, '/home/Dataset')
    output_fn = os.path.join(DATASET_DIR_ROOT, "MTCNN/align/+".format(ext_ratio) + output_fn + ".zip")
    output_folder_dir = os.path.dirname(output_fn)
    # import pdb; pdb.set_trace()
    print('input_fn: ', input_fn)
    print("output_fn: ", output_fn)

    # skip if output_fn exists
    if os.path.exists(output_fn):
        print("output_fn exists, skip")
        return
    elif not os.path.exists(output_folder_dir):
        print('Create ', output_folder_dir)
        os.makedirs(output_folder_dir, exist_ok=True)

    # init VideoCapture
    cap = cv2.VideoCapture(input_fn)

    # get frame
    face_count = 0
    frame_count = 0
    bbox_string = ''
    with io.BytesIO() as bio:
        with zipfile.ZipFile(bio, "w") as zip:
            # write pngs to zip in memory
            for frame_idx in range(1000000000):
                # for i in range(5):
                #    ret, im = cap.read()
                ret, im = cap.read()
                if not ret:
                    print("video ends")
                    assert im is None
                    break


                frame_count += 1

                # rescale the bounding box
                # t = rect.top() / rescale
                # b = rect.bottom() / rescale
                # l = rect.left() / rescale
                # r = rect.right() / rescale
                try:
                    crop = mxnet_detector.detect_and_align(im, 256)
                    face_count+=1
                except:
                    continue



                # save crop
                zip_helper.write_im_to_zip(zip, str(frame_idx) + ".png", crop)

            # write info.txt to zip in memory
            print(face_count, frame_count)
            # zip_helper.write_bytes_to_zip(zip, "bbox.txt", bytes(bbox_string, "utf-8"), )
            zip_helper.write_bytes_to_zip(zip, "info.txt", bytes("%d\t%d" % (face_count, frame_count), "utf-8"))

        # finally, flush bio to disk once
        path = os.path.dirname(output_fn)
        if path != "":
            os.makedirs(path, exist_ok=True)
        with open(output_fn, "wb") as f:
            f.write(bio.getvalue())

    cap.release()


def main(dataset_name):


    # pool = Pool(2)

    matching_pattern = DATASET_DIR[dataset_name]
    print(matching_pattern)
    video_fns = glob(matching_pattern)
    i = 1
    for fns in video_fns:
        print("Video {}/{}".format(i, len(video_fns)))
        process_one_video(fns)
        i+=1
    #pool.map(process_one_video, video_fns)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    print(dataset_name)
    if dataset_name in DATASET_DIR.keys():
        main(dataset_name)
    # preprocess()
    # single_test_case()
