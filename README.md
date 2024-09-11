# Adversarial Patch for Rotation-Based Class Confusion

## Creative Component
Involves training a single square patch to mislead a model into recognizing it as different classes depending on its orientation. This concept is tested in two distinct attempts:

#### Attempt 1: 
Train the patch to be recognized as four different classes based on its four possible orientations (0°, 90°, 180°, and 270°). This method aims to exploit the model’s reliance on orientation-specific features to misclassify the patch.

#### Attempt 2: 
To address the limitations observed in Attempt 1, where the top-1 attack accuracy averaged around 50% and top-5 attack accuracy averaged around 80%, tried simplifying the design by training the patch to be recognized as two different classes based on just two orientations (0° and 180°). This adjustment seeks to enhance the effectiveness of the adversarial patch by reducing potential confusion and improving its attack accuracy.

#### Trial and Error: How to choose the classes?
Theoretically, the most optimal results can be obtained when an object rotated by 90 degree looks like another object. For example, a horizontal banana when rotated by 90 degree may be trained to resemble a snake, which again rotated by 90 degree, may resemble a slug which when rotated again may be trained to be a goldfish. The selection of these classes is not perfect but a good baseline to start at.

## Practical Application
By using this patch to obscure or alter object detection systems, it can provide enhanced privacy protection by making objects appear differently depending on their angle. This can be particularly useful in environments where individuals seek to protect their identity or evade automated surveillance systems.

The goal is to assess the effectiveness of rotational adversarial patches in deceiving image classification models.

Reference: https://github.com/AIPI-590-XAI/Duke-AI-XAI
