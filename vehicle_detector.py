import time
from datetime import datetime

import torch
import ffmpeg
import numpy as np
import pytz
import youtube_dl
from PIL import Image
from matplotlib import pyplot as plt
from sqlalchemy import MetaData, Table
from sqlalchemy import create_engine

engine = create_engine(''.join(['sqlite:///', 'DetectedObjects.db']))

stream_url = 'https://www.youtube.com/watch?v=1EiC9bvVGnk'

yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')

tz = pytz.timezone('UTC')


def get_yt_dl_url(url):

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

    timestamp = datetime.now(tz=tz).strftime('%Y-%m-%d %H:%M:%S')

    probe = ffmpeg.probe(yt_dl_url)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width'])
    height = int(video_stream['height'])

    img = np.frombuffer(out, np.uint8).reshape([height, width, 3])

    return img, timestamp


def show_image(img):

    fig_width = float(img.shape[1] / 100)
    fig_height = float(img.shape[0] / 100)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(img)


def save_image(img, filename):

    im = Image.fromarray(img)
    im.save(filename)


def detect_objects(model, img, classes=(2, 3, 5, 7), conf=0.5, iou=0.45):

    model.conf = conf  # confidence threshold (0-1)
    model.iou = iou  # NMS IoU threshold (0-1)
    model.classes = list(classes)  # car, motorcycle, bus, truck

    results = model(img, size=img.shape[1])

    return results.render()[0], results.pandas().xyxy[0]


def save_detections(db_engine, detections, timestamp, table_name='DetectedObjects'):

    detections_dict = detections['name'].value_counts().to_dict()
    detections_dict['date'] = timestamp

    metadata = MetaData(db_engine)
    tb = Table(table_name, metadata, autoload=True)

    db_engine.execute(tb.insert(), detections_dict)


def main():

    while True:
        yt_dl_url = get_yt_dl_url(stream_url)

        img, timestamp = grab_frame(yt_dl_url)

        img_w_det, detections = detect_objects(yolov5_model, img)

        save_image(img_w_det, 'most_recent_detection.jpg')

        save_detections(engine, detections, timestamp)

        time.sleep(30.0 - (timestamp % 30.0))


if __name__ == '__main__':

    main()

