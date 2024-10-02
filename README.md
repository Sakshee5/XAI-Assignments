# Overview
This repository demonstrates how to use LIME (Local Interpretable Model-Agnostic Explanations) to generate local explanations for predictions made by the Inception V3 model. Inception V3 is a pre-trained deep learning model on the ImageNet dataset, primarily used for image classification tasks. (`Explainable_ML_Inception_LIME.ipynb`)

(Optional) The original attempt involved using YOLO with LIME since LIME on object detection is a less explored topic but could not get it working. The LIME explainer ends up returning an empty `explainer` list. Will be interested in understanding the implementation if someone else is successful in doing it! (see `Not_working_YOLO_LIME.ipynb`)

# Strengths
1. Explains individual predictions, which makes it easier to understand and visualize predictions.
2. It is Model-Agnostic and can be applied to any pre-trained model.
3. Visualization are intuitive. For example, visual explanations in the form of superpixels for images, which highlight areas of an image is super easy to understand.

# Limitations
1. It is computationally expensive since it perturbs the input data and evaluates the model. In this case, it took ~4.5 mins to finish a 1000 perturbations.

2. The quality of explanations depends on how well the image is segmented into superpixels. If the segmentation is poor, the explanation might not be meaningful.

# Potential Improvements
1. Run on GPU for faster computation.
2. Experiment with hyperparameters (for example, reducing `num_samples`) to strike a balance between performance and computation.