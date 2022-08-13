import json
import pytz
from datetime import datetime

import pandas as pd
import plotly
from flask import Flask
from flask import render_template
from plotly.graph_objs import Pie
from plotly.graph_objs import Bar

from sqlalchemy import create_engine
import os
app = Flask(__name__)

IMG_FOLDER = os.path.join('static', 'IMG')

app.config['UPLOAD_FOLDER'] = IMG_FOLDER

# load data
engine = create_engine('sqlite:///data/DetectedObjects.db')


def time_to_tz_naive(t, tz_in, tz_out):
    return tz_in.localize(datetime.combine(datetime.today(), t)).astimezone(tz_out).time()


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    example = os.path.join(IMG_FOLDER, 'most_recent_detection.jpg')

    # create visuals
    detected_objects = pd.read_sql_table('DetectedObjects', engine).set_index('id')
    detected_objects['date'] = pd.to_datetime(detected_objects['date'])
    object_types = ['car', 'motorcycle', 'bus', 'truck']
    detected_objects['total'] = detected_objects[object_types].sum(axis=1)
    # Group by time interval
    num_obj_by_hour = (detected_objects.groupby(detected_objects['date'].dt.floor('30Min').dt.time)
                                       .agg({'total': ['mean', 'std']}))

    num_obj_by_hour.columns = num_obj_by_hour.columns.droplevel()
    num_obj_by_hour.reset_index(inplace=True)
    num_obj_by_hour = num_obj_by_hour.rename(columns={'date': 'time'})

    # Convert to timezone of the cam location. Hardcoded for now since the stream url is also hardcoded
    tz_in = pytz.utc
    tz_out = pytz.timezone('US/Mountain')
    num_obj_by_hour['time'] = num_obj_by_hour['time'].apply(lambda x: time_to_tz_naive(x, tz_in, tz_out))
    num_obj_by_hour.sort_values('time', inplace=True)

    obj_by_type = detected_objects[object_types].sum().reset_index()
    obj_by_type.columns = ['vehicle_type', 'count']
    obj_by_type['count'] = obj_by_type['count'].astype(float) / obj_by_type['count'].sum()

    graphs = [
        {
            'data': [
                Bar(
                    x=num_obj_by_hour['time'],
                    y=num_obj_by_hour['mean'],
                    error_y=dict(
                        type='data',  # value of error bar given in data coordinates
                        array=num_obj_by_hour['std'],
                        thickness=1,
                        width=2,)
                )
            ],

            'layout': {
                'title': 'Average number of vehicles in the image',
                'yaxis': {
                    'title': "Average number of vehicles in the image"
                },
                'xaxis': {
                    'title': 'Local Time',

                },

            }
        },

        {
            'data': [
                Pie(
                    labels=obj_by_type['vehicle_type'],
                    values=obj_by_type['count'],
                )
            ],

            'layout': {
                'title': 'Detected Vehicle Types',
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,  user_image=example)


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()
