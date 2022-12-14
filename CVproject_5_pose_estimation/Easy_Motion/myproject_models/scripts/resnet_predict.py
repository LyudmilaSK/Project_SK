import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms

def get_predict(num_foto):
    path_img = '/content/drive/MyDrive/SF/Project_5/Easy_Motion/myproject_models/images_resnet/'

    # Загрузка модели и получение предсказания по num_foto фотографиям
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights='KeypointRCNN_ResNet50_FPN_Weights.DEFAULT'
    model_keypointrcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(progress=True, weights=weights)
    model_keypointrcnn.to(device)

    prediction_model = []
    prediction_person = []

    transform = transforms.Compose([transforms.ToTensor()])

    for foto in range(num_foto):

        path_model = os.path.join(path_img,'model/frame')+ str(foto)+'.jpg'
        path_person = os.path.join(path_img,'person/frame')+ str(foto)+'.jpg'

        image_model = Image.open(path_model).convert("RGB")
        img_tensor_model = transform(image_model)
        img_tensor_model = img_tensor_model.unsqueeze(0)

        image_person = Image.open(path_person).convert("RGB")
        img_tensor_person = transform(image_person)
        img_tensor_person = img_tensor_person.unsqueeze(0)

        model_keypointrcnn.eval()

        img_tensor_model.to(device)
        img_tensor_person.to(device)
        with torch.no_grad():
          prediction_model.append(model_keypointrcnn(img_tensor_model)[0])
          prediction_person.append(model_keypointrcnn(img_tensor_person)[0])

    return  prediction_model, prediction_person

