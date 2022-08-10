import time
from datetime import datetime
import os

import torch
import ffmpeg
import numpy as np
import pytz
import youtube_dl
from PIL import Image
from matplotlib import pyplot as plt
from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine

engine = create_engine(''.join(['sqlite:///', 'data/DetectedObjects.db']))

stream_url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk'

yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5m6')


def get_yt_dl_url(url):
    """Returns the actual video url of a YouTube video.

    Args:
        url: (string) url of a Youtube page.

    Returns:
        url of the video
    """

    ydl = youtube_dl.YoutubeDL({'quiet': True})

    with ydl:
        result = ydl.extract_info(url, download=False)

    if 'entries' in result:
        # Can be a playlist or a list of videos
        video = result['entries'][0]
    else:
        # Just a video
        video = result

    yt_dl_url = video['url']

    return yt_dl_url


def grab_frame(yt_dl_url):
    """Returns current frame and timestamp from the YouTube live stream.

    Args:
        yt_dl_url: (string) download url of a YouTube video.

    Returns:
        img: (numpy.array) current video frame
        timestamp: (datetime.datetime) current timestamp

    Raises:
    """

    out, _ = (ffmpeg
              .input(yt_dl_url)
              .output('pipe:',
                      vframes=1,
                      format='image2',
                      strftime=1,
                      loglevel='error',
                      pix_fmt='rgb24',
                      vcodec='rawvideo')
              .run(capture_stdout=True))

    timestamp = datetime.now(pytz.timezone('UTC'))

    probe = ffmpeg.probe(yt_dl_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    img = np.frombuffer(out, np.uint8).reshape([height, width, 3])

    return img, timestamp


def show_image(img):
    """Shows image

    Args:
        img: (numpy.array) image

    Returns:
    """

    fig_width = float(img.shape[1] / 100)
    fig_height = float(img.shape[0] / 100)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(img)


def save_image(img, filename):
    """Saves the image

    Args:
        img: (numpy.array) image
        filename: (string) path where to save the image

    Returns:

    """

    im = Image.fromarray(img)
    im.save(filename)


def detect_objects(model, img, classes=(2, 3, 5, 7), conf=0.5, iou=0.45):
    """Detects objects im the image

    Args:
        model: PyTorch model
        img: (numpy.array) image
        classes: (list) subset of 80 object classes to detect
        conf: (float) confidence threshold (0-1)
        iou: (float) NMS IoU threshold (0-1)

    Returns:
        img: (numpy.array) image with detected objects
        results_df: (pandas.Dataframe) dataframe with detected objects
    """

    model.conf = conf  # confidence threshold (0-1)
    model.iou = iou  # NMS IoU threshold (0-1)
    model.classes = list(classes)  # car, motorcycle, bus, truck

    results = model(img, size=img.shape[1])
    results.print()

    img_w_detections = results.render()[0]
    results_df = results.pandas().xyxy[0]

    return img_w_detections, results_df


def save_detections(db_engine, detections, timestamp, table_name='DetectedObjects'):
    """Saves detections to the database

    Args:
        db_engine: (sqlalchemy.Engine) database engine
        detections: (pandas.Dataframe) dataframe with detected objects
        timestamp: (datetime.datetime) timestamp of the detection
        table_name: (string) name of the table where to save the detections

    Returns:

    """

    detections_dict = detections['name'].value_counts().to_dict()
    detections_dict['date'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')

    metadata = MetaData(db_engine)
    tb = Table(table_name, metadata, autoload=True)

    db_engine.execute(tb.insert(), detections_dict)


def main():
    """Gets video stream url, grabs the current frame, detects vehicles,
    saves image, saves results to DB

    Returns:

    """

    while True:
        yt_dl_url = get_yt_dl_url(stream_url)

        img, timestamp = grab_frame(yt_dl_url)

        img_w_det, detections = detect_objects(yolov5_model, img)

        save_image(img_w_det, os.path.join('app', 'static', 'IMG', 'most_recent_detection.jpg'))

        save_detections(engine, detections, timestamp)

        time.sleep(30.0 - (timestamp.second % 30.0))


if __name__ == '__main__':

    main()
