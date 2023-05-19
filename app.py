import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

# Load the pre-trained DeepLabV3+ model from Torch Hub
model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# Define the input and output shapes for the model
input_shape = (512, 512, 3)
output_shape = (512, 512)

# Load and preprocess the image
image = cv2.imread("10590.jpg")
image = cv2.resize(image, input_shape[:2])
image = transforms.ToTensor()(image)
image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
image = image.unsqueeze(0)

# Perform semantic segmentation on the image
with torch.no_grad():
    output = model(image)['out']
output = torch.nn.functional.interpolate(output, size=output_shape, mode='bilinear', align_corners=False)
output = torch.argmax(output, dim=1)
output = output.cpu().numpy()[0]

# Display the segmented image
# cv2.imshow("Segmented Image", output)
cv2.imwrite('1.jpg', output)
cv2.waitKey(0)
