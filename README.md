# Vehicle detection and analysis in live traffic cam
### Table of Contents
1. [Installation](#Installation)
2. [Project Motivation](#Project Motivation)
3. [File Descriptions](#File Descriptions)
4. [Results](#Results)
5. [Licensing](#Licensing)

## Installation 

Apart from the Anaconda distribution of Python following libraries have been used: `plotly`, `pytorch`, `youtube-dl`,
`ffmpeg-python`, `flask`.
The code should run with no issues using `python>=3.8`. For more information check the
[YOLOv5 repo](https://github.com/ultralytics/yolov5/).

## Project Motivation

In this project I use the deep learning model YOLOv5 to detect vehicles in a video live stream. I wanted to answer the
questions such as 

- How many vehicles on average are present in the picture depending on the time of the day.
- What are the proportions of cars, truck, buses and motorcycles.

In the future work the results can be used for example for: 

- Prediction of traffic congestions
- Estimation of the waiting time at a border crossing

## File Descriptions

- `detect_vehicles.py` grabs the current frame from a [YouTube stream](https://www.youtube.com/watch?v=1EiC9bvVGnk), detects the vehicles
and then the saves results to a database. The stream url is hardcoded for now.
- `app/run.py` starts the web server which can be accessed at `0.0.0.0:3001`.


## Results

The analysis of a single image takes about 4.5s on my old Lenovo ThinkPad with Intel i7-3520m from 2012. 
However, I'm sure it can be made much faster using lower resolution and batch processing.
For a time analysis on hourly basis one image every 30s was enough for me.

The YOLOv5 model performs very consistently, detecting even those cars which appear tiny in the background.

The following graphs were created from this [YouTube stream](https://www.youtube.com/watch?v=1EiC9bvVGnk).
As expected there is almost no traffic during the night, and it is the busiest between 10am and 7pm.

<img src="images/number_of_vehicles.png" alt="drawing" width="50%"/>

Of all detected vehicles around 80% are cars and 17.5% trucks and pickups

<img src="images/vehicle_types.png" alt="drawing" width="50%"/>

## Licensing

Must give credit to [Ultralytics](https://ultralytics.com/) for the YOLOv5 model. Feel free to use my code here as you would like!