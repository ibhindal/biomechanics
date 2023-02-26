!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="ZA4si4Lrw5rPgUtn8h3a")
project = rf.workspace("ball-ahqgn").project("ball-5zhhg")
dataset = project.version(1).download("tfrecord")