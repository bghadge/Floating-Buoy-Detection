from PIL import Image
from torchvision import transforms



image = Image.open("/home/bhushan/Projects/RobotX/Floating-Buoy-Detection/dataset/labelled/1440_whitebuoy/buoy_00000.png")
trans = transforms.Compose([
                            transforms.Resize((320, 224)),
                            transforms.ToTensor(),
                            ])

image = trans(image)
print(image.size())