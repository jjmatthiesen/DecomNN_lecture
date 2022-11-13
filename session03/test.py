import pathlib
import torch
from torch import nn
from torchvision import datasets, transforms, models
import session03.utils.glob as CDglobs


if __name__ == '__main__':
    val_data = datasets.ImageFolder('data/catsDogs/validation_set', transform=CDglobs.test_transforms)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=100, shuffle=False)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    criterion = nn.CrossEntropyLoss()

    model_saved_name = "resnet18_lr_0.001_pretrained_true_bs100_ep_5"
    # load model
    model.load_state_dict(
        torch.load('export_models/' + model_saved_name + '.pt', map_location='cpu'))

    device = 'cpu'

    model.to(device)

    # TESTING
    # ------------------

    with torch.no_grad():
        model.eval()
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = ((val_output.argmax(dim=1) == label).float().mean())
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

        print('val_accuracy : {}, val_loss : {}'.format(epoch_val_accuracy, epoch_val_loss))
        pathlib.Path('results').mkdir(parents=True, exist_ok=True)
        with open('results/results_' + model_saved_name + '.csv', 'a', encoding='utf-8') as f:
            f.write(
                f'{"test_transforms"}, {epoch_val_accuracy}, {epoch_val_loss} \n')
        # resnet18_lr_0.001_pretrained_true_bs100_ep_5_augm: val_accuracy : 0.9617645740509033, val_loss : 0.10223330557346344
        # resnet18_lr_0.001_pretrained_false_bs100_ep_5_augm: val_accuracy : 0.6747059226036072, val_loss : 0.5825079083442688

