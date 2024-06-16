import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random

class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_size, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    models_path = 'models/'
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    target_class = input('Class name: ')
    existing_epochs = input('Epoch count: ')

    target_class_idx = classes.index(target_class)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainset = torch.utils.data.Subset(trainset, [i for i, x in enumerate(trainset) if x[1] == target_class_idx])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testset = torch.utils.data.Subset(testset, [i for i, x in enumerate(testset) if x[1] == target_class_idx])
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 200
    noise_size = 100
    
    discriminator = Discriminator().to(device)
    generator = Generator(noise_size).to(device)
    
    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.0002)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.0002)
    
    if existing_epochs:
        try:
            existing_epochs = int(existing_epochs)
        except:
            print('Bad input for epochs')
            exit()
        path = models_path + target_class + '_' + str(existing_epochs) + '.pt'
        if not os.path.exists(path):
            print('Model not found')
            exit()
        checkpoint = torch.load(path)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        d_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        g_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
    else:
        existing_epochs = 0

    data = list(trainloader)
    for epoch in range(existing_epochs, existing_epochs + num_epochs):
        random.shuffle(data)
        for i, (real_images, _) in enumerate(data):
            batch_size = real_images.size(0)
            noise = torch.randn(batch_size, noise_size, 1, 1).to(device)
            fake_images = generator(noise)
            d_real = discriminator(real_images.to(device))
            d_fake = discriminator(fake_images.detach().to(device))
            real_loss = loss_fn(d_real, torch.ones_like(d_real))
            fake_loss = loss_fn(d_fake, torch.zeros_like(d_fake))
            d_loss = real_loss + fake_loss
            discriminator.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            d_fake = discriminator(fake_images.to(device))
            g_loss = loss_fn(d_fake, torch.ones_like(d_fake))
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}, d_loss: {:.4f}, g_loss: {:.4f}'
                      .format(epoch + 1, existing_epochs + num_epochs, i + 1, len(trainloader), d_loss.item(), g_loss.item()))
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            torch.save({
                'epoch': epoch + 1,
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': d_optimizer.state_dict(),
                'generator_optimizer_state_dict': g_optimizer.state_dict(),
                }, models_path + target_class + '_' + str(epoch + 1) + '.pt')
    
    noise = torch.randn(10, noise_size, 1, 1).to(device)
    fake_images = generator(noise)
    for image in fake_images:
        plt.imshow(torch.permute(image.detach().cpu(), (1, 2, 0)))
        plt.show()