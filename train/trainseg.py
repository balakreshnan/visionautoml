import azureml.core
from azureml.core import Workspace
from azureml.core import Keyvault
import os

from azureml.core import Workspace, Experiment
from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling
from azureml.train.hyperdrive import choice

from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Experiment

from azureml.core import Workspace
import urllib
from zipfile import ZipFile

from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, RandomParameterSampling, BayesianParameterSampling
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, uniform
from IPython.display import Image
from azureml.core import Experiment

print("SDK version:", azureml.core.VERSION)

import argparse 
import json
import os
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import numpy as np
import PIL.Image as Image

from simplification.cutil import simplify_coords
from skimage import measure

parse = argparse.ArgumentParser()
parse.add_argument("--tenantid")
parse.add_argument("--acclientid")
parse.add_argument("--accsecret")
    
args = parse.parse_args()


sp = ServicePrincipalAuthentication(tenant_id=args.tenantid, # tenantID
                                    service_principal_id=args.acclientid, # clientId
                                    service_principal_password=args.accsecret) # clientSecret

ws = Workspace.get(name="gputraining",
                   auth=sp,
                   subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773", resource_group="mlops")

#ws = Workspace.from_config()
keyvault = ws.get_default_keyvault()
tenantid = keyvault.get_secret(name="tenantid")
acclientid = keyvault.get_secret(name="acclientid")
accsvcname = keyvault.get_secret(name="accsvcname")
accsecret = keyvault.get_secret(name="accsecret")

print(accsvcname)

sp = ServicePrincipalAuthentication(tenant_id=tenantid, # tenantID
                                    service_principal_id=acclientid, # clientId
                                    service_principal_password=accsecret) # clientSecret

ws = Workspace.get(name="gputraining",
                   auth=sp,
                   subscription_id="c46a9435-c957-4e6c-a0f4-b9a597984773", resource_group="mlops")
ws.get_details()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

from azureml.core.compute import AmlCompute, ComputeTarget

cluster_name = "gpu-cluster"

try:
    compute_target = ws.compute_targets[cluster_name]
    print('Found existing compute target.')
except KeyError:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_NC6', 
                                                           idle_seconds_before_scaledown=1800,
                                                           min_nodes=0, 
                                                           max_nodes=4)

    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    
# Can poll for a minimum number of nodes and for a specific timeout.
# If no min_node_count is provided, it will use the scale settings for the cluster.
compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)


experiment_name = 'automl-image-notebook-segmentation' 
experiment = Experiment(ws, name=experiment_name)

# download data
download_url = 'https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip'
data_file = './odFridgeObjectsMask.zip'
urllib.request.urlretrieve(download_url, filename=data_file)

# extract files
with ZipFile(data_file, 'r') as zip:
    print('extracting files...')
    zip.extractall()
    print('done')
    
# delete zip file
os.remove(data_file)

from IPython.display import Image
Image(filename='./odFridgeObjectsMask/images/31.jpg')


def convert_mask_to_polygon(mask, max_polygon_points=100, score_threshold=0.5, max_refinement_iterations=25,
                            edge_safety_padding=1):
    """Convert a numpy mask to a polygon outline in normalized coordinates.

    :param mask: Pixel mask, where each pixel has an object (float) score in [0, 1], in size ([1, height, width])
    :type: mask: <class 'numpy.array'>
    :param max_polygon_points: Maximum number of (x, y) coordinate pairs in polygon
    :type: max_polygon_points: Int
    :param score_threshold: Score cutoff for considering a pixel as in object.
    :type: score_threshold: Float
    :param max_refinement_iterations: Maximum number of times to refine the polygon
    trying to reduce the number of pixels to meet max polygon points.
    :type: max_refinement_iterations: Int
    :param edge_safety_padding: Number of pixels to pad the mask with
    :type edge_safety_padding: Int
    :return: normalized polygon coordinates
    :rtype: list of list
    """
    # Convert to numpy bitmask
    mask = mask[0]
    mask_array = np.array((mask > score_threshold), dtype=np.uint8)
    image_shape = mask_array.shape

    # Pad the mask to avoid errors at the edge of the mask
    embedded_mask = np.zeros((image_shape[0] + 2 * edge_safety_padding,
                              image_shape[1] + 2 * edge_safety_padding),
                             dtype=np.uint8)
    embedded_mask[edge_safety_padding:image_shape[0] + edge_safety_padding,
                  edge_safety_padding:image_shape[1] + edge_safety_padding] = mask_array

    # Find Image Contours
    contours = measure.find_contours(embedded_mask, 0.5)
    simplified_contours = []

    for contour in contours:

        # Iteratively reduce polygon points, if necessary
        if max_polygon_points is not None:
            simplify_factor = 0
            while len(contour) > max_polygon_points and simplify_factor < max_refinement_iterations:
                contour = simplify_coords(contour, simplify_factor)
                simplify_factor += 1

        # Convert to [x, y, x, y, ....] coordinates and correct for padding
        unwrapped_contour = [0] * (2 * len(contour))
        unwrapped_contour[::2] = np.ceil(contour[:, 1]) - edge_safety_padding
        unwrapped_contour[1::2] = np.ceil(contour[:, 0]) - edge_safety_padding

        simplified_contours.append(unwrapped_contour)

    return _normalize_contour(simplified_contours, image_shape)


def _normalize_contour(contours, image_shape):

    height, width = image_shape[0], image_shape[1]

    for contour in contours:
        contour[::2] = [x * 1. / width for x in contour[::2]]
        contour[1::2] = [y * 1. / height for y in contour[1::2]]

    return contours


def binarise_mask(mask_fname):

    mask = Image.open(mask_fname)
    mask = np.array(mask)
    # instances are encoded as different colors
    obj_ids = np.unique(mask)
    # first id is the background, so remove it
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of binary masks
    binary_masks = mask == obj_ids[:, None, None]
    return binary_masks


def parsing_mask(mask_fname):

    # For this particular dataset, initially each mask was merged (based on binary mask of each object)
    # in the order of the bounding boxes described in the corresponding PASCAL VOC annotation file.
    # Therefore, we have to extract each binary mask which is in the order of objects in the annotation file.
    # https://github.com/microsoft/computervision-recipes/blob/master/utils_cv/detection/dataset.py
    binary_masks = binarise_mask(mask_fname)
    polygons = []
    for bi_mask in binary_masks:

        if len(bi_mask.shape) == 2:
            bi_mask = bi_mask[np.newaxis, :]
        polygon = convert_mask_to_polygon(bi_mask)
        polygons.append(polygon)

    return polygons


def convert_mask_in_VOC_to_jsonl(base_dir, workspace):

    src = base_dir
    train_validation_ratio = 5

    # Retrieving default datastore that got automatically created when we setup a workspace
    workspaceblobstore = workspace.get_default_datastore().name

    # Path to the annotations
    annotations_folder = os.path.join(src, "annotations")
    mask_folder = os.path.join(src, "segmentation-masks")

    # Path to the training and validation files
    train_annotations_file = os.path.join(src, "train_annotations.jsonl")
    validation_annotations_file = os.path.join(src, "validation_annotations.jsonl")

    # sample json line dictionary
    json_line_sample = \
        {
            "image_url": "AmlDatastore://" + workspaceblobstore + "/"
                         + os.path.basename(os.path.dirname(src)) + "/" + "images",
            "image_details": {"format": None, "width": None, "height": None},
            "label": []
        }

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, 'w') as train_f:
        with open(validation_annotations_file, 'w') as validation_f:
            for i, filename in enumerate(os.listdir(annotations_folder)):
                if filename.endswith(".xml"):
                    print("Parsing " + os.path.join(src, filename))

                    root = ET.parse(os.path.join(annotations_folder, filename)).getroot()

                    width = int(root.find('size/width').text)
                    height = int(root.find('size/height').text)
                    # convert mask into polygon
                    mask_fname = os.path.join(mask_folder, filename[:-4] + ".png")
                    polygons = parsing_mask(mask_fname)

                    labels = []
                    for index, object in enumerate(root.findall('object')):
                        name = object.find('name').text
                        isCrowd = int(object.find('difficult').text)
                        labels.append({"label": name,
                                       "bbox": "null",
                                       "isCrowd": isCrowd,
                                       'polygon': polygons[index]})

                    # build the jsonl file
                    image_filename = root.find("filename").text
                    _, file_extension = os.path.splitext(image_filename)
                    json_line = dict(json_line_sample)
                    json_line["image_url"] = json_line["image_url"] + "/" + image_filename
                    json_line["image_details"]["format"] = file_extension[1:]
                    json_line["image_details"]["width"] = width
                    json_line["image_details"]["height"] = height
                    json_line["label"] = labels

                    if i % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
                else:
                    print("Skipping unknown file: {}".format(filename))

from jsonl_converter import convert_mask_in_VOC_to_jsonl

data_path = "./odFridgeObjectsMask/"
convert_mask_in_VOC_to_jsonl(data_path, ws)

# Retrieving default datastore that got automatically created when we setup a workspace
ds = ws.get_default_datastore()
ds.upload(src_dir='./odFridgeObjectsMask', target_path='odFridgeObjectsMask')

from azureml.contrib.dataset.labeled_dataset import _LabeledDatasetFactory, LabeledDatasetTask
from azureml.core import Dataset

training_dataset_name = 'odFridgeObjectsMaskTrainingDataset'
if training_dataset_name in ws.datasets:
    training_dataset = ws.datasets.get(training_dataset_name)
    print('Found the training dataset', training_dataset_name)
else:
    # create training dataset
    training_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.IMAGE_INSTANCE_SEGMENTATION, path=ds.path('odFridgeObjectsMask/train_annotations.jsonl'))
    training_dataset = training_dataset.register(workspace=ws, name=training_dataset_name)
    
# create validation dataset
validation_dataset_name = "odFridgeObjectsMaskValidationDataset"
if validation_dataset_name in ws.datasets:
    validation_dataset = ws.datasets.get(validation_dataset_name)
    print('Found the validation dataset', validation_dataset_name)
else:
    validation_dataset = _LabeledDatasetFactory.from_json_lines(
        task=LabeledDatasetTask.IMAGE_INSTANCE_SEGMENTATION, path=ds.path('odFridgeObjectsMask/validation_annotations.jsonl'))
    validation_dataset = validation_dataset.register(workspace=ws, name=validation_dataset_name)
    
    
print("Training dataset name: " + training_dataset.name)
print("Validation dataset name: " + validation_dataset.name)

training_dataset.to_pandas_dataframe()

from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling
from azureml.train.hyperdrive import choice

image_config_maskrcnn = AutoMLImageConfig(
      task='image-instance-segmentation',
      compute_target=compute_target,
      training_data=training_dataset,
      validation_data=validation_dataset,
      hyperparameter_sampling=GridParameterSampling({'model_name': choice('maskrcnn_resnet50_fpn')}))

automl_image_run = experiment.submit(image_config_maskrcnn)

automl_image_run.wait_for_completion(wait_post_processing=True)

from azureml.train.automl import AutoMLImageConfig
from azureml.train.hyperdrive import GridParameterSampling, RandomParameterSampling, BayesianParameterSampling
from azureml.train.hyperdrive import BanditPolicy, HyperDriveConfig, PrimaryMetricGoal
from azureml.train.hyperdrive import choice, uniform

parameter_space = {
    'model_name': choice('maskrcnn_resnet50_fpn'),
    'learning_rate': uniform(0.0001, 0.001),
    #'warmup_cosine_lr_warmup_epochs': choice(0, 3),
    'optimizer': choice('sgd', 'adam', 'adamw'),
    'min_size': choice(600, 800)
}

tuning_settings = {
    'iterations': 20, 
    'max_concurrent_iterations': 4, 
    'hyperparameter_sampling': RandomParameterSampling(parameter_space),  
    'policy': BanditPolicy(evaluation_interval=2, slack_factor=0.2, delay_evaluation=6)
}


automl_image_config = AutoMLImageConfig(task='image-instance-segmentation',
                                        compute_target=compute_target,
                                        training_data=training_dataset,
                                        validation_data=validation_dataset,
                                        primary_metric='mean_average_precision',
                                        **tuning_settings)

automl_image_run = experiment.submit(automl_image_config)

automl_image_run.wait_for_completion(wait_post_processing=True)

# Register the model from the best run

best_child_run = automl_image_run.get_best_child()
model_name = best_child_run.properties['model_name']
model = best_child_run.register_model(model_name = model_name, model_path='outputs/model.pt')