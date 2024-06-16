import torch
import matplotlib.pyplot as plt
import os
from gan import Generator

if __name__ == '__main__':
    models_path = 'models/'
    name = input('Class name: ')
    epoch = input('Epoch count: ')
    path = models_path + name + '_' + epoch + '.pt'
    if not os.path.exists(path):
        print('Model not found')
        exit()
    noise_size = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(noise_size).to(device)
    checkpoint = torch.load(path)
    gen.load_state_dict(checkpoint['generator_state_dict'])
    noise = torch.randn(10, noise_size, 1, 1).to(device)
    fake_images = gen(noise)
    for image in fake_images:
        plt.imshow(torch.permute(image.detach().cpu(), (1, 2, 0)))
        plt.show()