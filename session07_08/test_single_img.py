import pathlib
import torch
from torch import nn
from torchvision import datasets, transforms, models
import session03.utils.glob as CDglobs
import session03.utils.utils as Utils

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    val_data = datasets.ImageFolder('../session03/data/catsDogs/validation_set', transform=CDglobs.test_transforms)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=100, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    criterion = nn.CrossEntropyLoss()

    model_saved_name = "resnet18_lr_0.001_pretrained_true_bs100_ep_5"
    # load model
    model.load_state_dict(
        torch.load('../session03/export_models/' + model_saved_name + '.pt', map_location='cpu'))

    device = 'cpu'

    model.to(device)

    # get one batch from the data loader
    inputs, classes = next(iter(val_loader))

    idx = 4
    # get the ground truth (real label) cat=0; dog=1
    exp_label = classes[idx]
    # print image
    Utils.imshow(inputs[idx])

    # get one image and reformat
    img = inputs[idx].unsqueeze_(0)
    img.to(device)

    # predict the class for image
    pred = model(img)
    # print class in words
    pred_label = ("dog" if int(pred.argmax(dim=1)==1) else "cat")
    print("predicted label: " + pred_label)
    # print true label in words
    print("true label: " + ("dog" if int(exp_label == 1) else "cat"))
    Utils.imshow(inputs[idx], title=pred_label)
    print("End")