import streamlit as st
from PIL import Image
import torch
import sys
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
import pretrainedmodels

class Network(nn.Module):
    """Network
    """
    def __init__(self, backbone="resnet50", num_classes=7, input_channel=3,
                 pretrained=True):
        super(Network, self).__init__()
        if backbone == "resnet50":
            model = ResNet50(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained)
        elif backbone == "resnet18":
            model = ResNet18(num_classes=num_classes,
                             input_channel=input_channel,
                             pretrained=pretrained)
        elif backbone == "PNASNet5Large":
            model = PNASNet5Large(num_classes=num_classes,
                                  input_channel=input_channel,
                                  pretrained=pretrained)
        elif backbone == "NASNetALarge":
            model = NASNetALarge(num_classes=num_classes,
                                 input_channel=input_channel,
                                 pretrained=pretrained)
        else:
            print("Need model")
            sys.exit(-1)
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)

    def print_model(self, input_size, device):
        """Print model structure
        """
        self.model.to(device)
        summary(self.model, input_size)

class ResNet18(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(ResNet18, self).__init__()
        self.features = nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=pretrained).
                  children())[:-1]
            )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):
    """AlexNet
    """
    def __init__(self, num_classes, input_channel, pretrained):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(
            *list(torchvision.models.resnet50(pretrained=pretrained).
                  children())[:-1]
            )
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Load the saved model
model_path = "10"  # Update with the correct path
loaded_model1 = Network(backbone= 'resnet18', num_classes=7,
                             input_channel=3, pretrained=True)
loaded_model1.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
loaded_model1.eval()

model_path2 = "your_model.pth"  # Update with the correct path
loaded_model2 = Network(backbone= 'resnet18', num_classes=7,
                             input_channel=3, pretrained=True)
loaded_model2.load_state_dict(torch.load(model_path2, map_location=torch.device('cpu')))
loaded_model2.eval()


def main():
    st.title("Image Processing App")
    st.write("Upload an image and let the model do the rest!")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        processed_image = process_image(image)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

def process_image(image):
    # Define preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    input_image = preprocess(image)
    input_image = input_image.unsqueeze(0)



    with torch.no_grad():
        output = loaded_model1(input_image)
        predicted_class = torch.argmax(output, dim=1).item()

    class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    predicted_class_name = class_names[predicted_class]

    # Create a visualization of the prediction
    visualization = Image.new("RGB", (224, 224))
    st.text(f"Predicted Class: {predicted_class_name}")

    return visualization

main()

