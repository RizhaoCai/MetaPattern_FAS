import os
import zipfile
import numpy as np

import cv2


def write_bytes_to_zip(zip, bytes_fn, b):
    zip.writestr(bytes_fn, b)


def write_bytes_to_zip_file(zip_fn, bytes_fn, b):
    path = os.path.dirname(zip_fn)
    if path != "":
        os.makedirs(path, exist_ok=True)
    with zipfile.ZipFile(zip_fn, "a") as zip:
        write_bytes_to_zip(zip, bytes_fn, b)


def read_bytes_from_zip(zip, bytes_fn):
    b = zip.read(bytes_fn)
    return b


def read_bytes_from_zip_file(zip_fn, bytes_fn):
    with zipfile.ZipFile(zip_fn, "r") as zip:
        return read_bytes_from_zip(zip, bytes_fn)


##############################################################


def im_to_bytes(im, ext):
    r, b = cv2.imencode(ext, im)
    assert r
    b = b.tobytes()
    return b


def bytes_to_im(b, flags):
    b = np.frombuffer(b, dtype=np.uint8)
    im = cv2.imdecode(b, flags)
    return im


##############################################################


def write_im_to_zip(zip, im_fn, im):
    _, ext = os.path.splitext(im_fn)
    return write_bytes_to_zip(zip, im_fn, im_to_bytes(im, ext))


def write_im_to_zip_file(zip_fn, im_fn, im):
    _, ext = os.path.splitext(im_fn)
    return write_bytes_to_zip_file(zip_fn, im_fn, im_to_bytes(im, ext))


def read_im_from_zip(zip, im_fn, flags):
    return bytes_to_im(read_bytes_from_zip(zip, im_fn), flags)


def read_im_from_zip_file(zip_fn, im_fn, flags):
    return bytes_to_im(read_bytes_from_zip_file(zip_fn, im_fn), flags)


def read_list_from_zip_file(zip_fn, ext=""):
    with zipfile.ZipFile(zip_fn, "r") as zip:
        lst = zip.namelist()
    lst2 = []
    for im_fn in lst:
        if im_fn.endswith(ext):
            lst2.append(im_fn)
    return lst2


# for test
def main():
    # test im
    im = np.zeros([256, 256, 3], np.uint8)
    cv2.putText(im, "123", (50, 150), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255))

    write_im_to_zip_file("1.zip", "1.jpg", im)

    im2 = read_im_from_zip_file("1.zip", "1.jpg", cv2.IMREAD_UNCHANGED)

    cv2.imshow("im", im)
    cv2.imshow("im2", im2)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
