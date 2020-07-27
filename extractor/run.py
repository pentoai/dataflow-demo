import apache_beam as beam
import click
import io
import logging
import pathlib
import numpy as np
import os

from datetime import datetime

from apache_beam.io import ReadFromText
from apache_beam.io.gcp.gcsio import GcsIO
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions

from PIL import Image

from tensorflow.keras.applications.resnet50 import ResNet50

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224


def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance


@singleton
class Extractor:
    """
    Extract image embeddings using a pre-trained ResNet50 model.
    """
    def __init__(self):
        self.model = ResNet50(
            input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
            include_top=False,
            weights="imagenet",
            pooling=None,
        )

    def extract(self, image):
        return self.model.predict(np.expand_dims(image, axis=0))[0]


def load(path):
    """
    Receives an image path and returns a dictionary containing
    the image path and a resized version of the image as a np.array.
    """
    buf = GcsIO().open(path, mime_type="image/jpeg")
    img = Image.open(io.BytesIO(buf.read()))
    img = img.resize((IMAGE_HEIGHT, IMAGE_WIDTH), Image.ANTIALIAS)
    return {"path": path, "image": np.array(img)}


def extract(item):
    """
    Extracts the feature embedding from item["image"].
    """
    extractor = Extractor()
    item["embedding"] = extractor.extract(item["image"])
    # We do not longer need the image,remove it from the dictonary to free memory.
    del item["image"]
    return item


def store(item, output_path):
    """
    Store the image embeddings item["embedding"] in GCS.
    """
    name = item["path"].split("/")[-1].split(".")[0]
    path = os.path.join(output_path, f"{name}.npy")

    fin = io.BytesIO()
    np.save(fin, item["embedding"])

    fout = beam.io.gcp.gcsio.GcsIO().open(path, mode="w")
    fin.seek(0)
    fout.write(fin.read())


@click.command()
@click.option("--job-name")
@click.option("--input", "input_path")
@click.option("--output", "output_path")
@click.option("--max-num-workers", default=None, type=int)
@click.option("--local", is_flag=True)
@click.option("--project-id", envvar="GCP_PROJECT_ID")
@click.option("--region", envvar="GCP_REGION", default="us-west1")
def run(
    job_name,
    input_path,
    output_path,
    max_num_workers,
    local,
    project_id,
    region
):
    if job_name is None:
        job_name = f"extractor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    click.echo(f"Running job: {job_name}")

    output_path = os.path.join(output_path, job_name)

    if local:
        # Execute pipeline in your local machine.
        runner_options = {
            "runner": "DirectRunner",
        }
    else:
        runner_options = {
            "runner": "DataflowRunner",
            "temp_location": os.path.join(output_path, "temp_location"),
            "staging_location.": os.path.join(output_path, "staging_location"),
            "max_num_workers": max_num_workers,
        }

    options = PipelineOptions(
        project=project_id,
        job_name=job_name,
        region=region,
        **runner_options
    )
    options.view_as(SetupOptions).save_main_session = True
    options.view_as(SetupOptions).setup_file = os.path.join(
        pathlib.Path(__file__).parent.absolute(), "..", "setup.py")

    with beam.Pipeline(options=options) as p:
        (p
         | "source" >> ReadFromText(input_path)
         | "load" >> beam.Map(load)
         | "extract" >> beam.Map(extract)
         | "store" >> beam.Map(store, output_path))


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
