import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

import utils.transformations as transformation
import utils.utils as Utils
import pathlib


if __name__ == '__main__':
    train_data = datasets.ImageFolder('data/catsDogs/training_set', transform=transformation.train_transforms)
    # Utils.imshow(train_data[3201][0])
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    # Freeze parameters, so we don't backprop through them
    for name, p in model.named_parameters():
        if name == 'conv1.weight' or name == 'fc.weight' or name == 'fc.bias':
            p.requires_grad = True
        else:
            p.requires_grad = False

    for name, p in model.named_parameters():
        print("param name:", name, "requires_grad:", p.requires_grad)

    device = 'cpu'

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    model_saved_name = "resnet18_lr_0.001_pretrained_true_bs100_ep_5"

    # TRAINING
    # ------------------
    epochs = 5

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        batch = 0

        for data, label in train_loader:
            model.train()
            data = data.to(device)
            label = label.to(device)
            batch += 1

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = ((output.argmax(dim=1) == label).float().mean()).item()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss.item() / len(train_loader)

            print(f"batch {int(batch)}; "
                  f"batch loss: {loss:.3f} ")
        print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))

    pathlib.Path('export_models/').mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), 'export_models/' + model_saved_name + '.pt')
    # resnet18_lr_0.001_pretrained_true_bs100_ep_5_augm: train accuracy : 0.9421652555465698, train loss : 0.14786799252033234
    # resnet18_lr_0.001_pretrained_false_bs100_ep_5_augm: train accuracy : 0.7522544637322426, train loss : 0.512696891091764
