# Investigating Model Bias - ResNet50 Camel Classification

## **Aim**
I am aiming to investigate potential bias in the ResNet50 model's interpretation of real-world scenes, especially with ditsinct environmental features. For example, here I am specifically focusing on "Arabian camel" images from the ImageNet dataset that have a distinctive "desert" background. The hypothesis examines whether the model relies more on environmental cues, such as deserts, than on the camel itself when making predictions.

## **Hypothesis**
- **H0 (Null Hypothesis)**: The ResNet50 model does not show a difference in its reliance on object-specific features compared to desert backgrounds when classifying images of "camels" in the ImageNet-V2 dataset.

- **H1 (Alternative Hypothesis)**: The ResNet50 model shows a difference in its reliance on desert backgrounds compared to object-specific features when classifying images of "camels" in the ImageNet-V2 dataset, i.e., it favors sandy or barren landscapes over the camels physical attributes.


## **Experimental Methods**
- ### Saliency Maps:
Saliency maps will be generated to visualize the pixels on which the ResNet50 model focuses when identifying camels. These maps help identify which pixels were weighted higher when making the prediction.

- ### Occlusion Sensitivity:
Here we occlude parts of the camel and the background in the images and measure the impact on the models classification confidence. By comparing occluded versions of images, we determine whether the removal of environmental features affects classification as much as removing parts of the camel itself.