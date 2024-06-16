import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random
import sqlite3
import collections
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose3d(noise_size, 1024, 4, 1, 0, bias = False),
            nn.BatchNorm3d(1024),
            nn.ReLU(True),
            nn.ConvTranspose3d(1024, 512, 4, 2, 1, bias = False),
            nn.BatchNorm3d(512),
            nn.ReLU(True),
            nn.ConvTranspose3d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm3d(256),
            nn.ReLU(True),
            nn.ConvTranspose3d(256, 4, 4, 2, 1, bias = False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(4, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv3d(256, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    models_path = 'models/'
    
    existing_epochs = input('Epoch count: ')
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    batch_size = 4
    
    con = sqlite3.connect('voxel.db')
    with con:
        cur = con.cursor()
        mats = cur.execute('select * from material').fetchall()
        mats = {m[0] : (m[2], m[3], m[4]) for m in mats}
        blocks = cur.execute('select * from block').fetchall()
    
    builds = collections.defaultdict(lambda: [])
    for block in blocks:
        build_id, mat_id, x, y, z = block
        builds[build_id].append((mats[mat_id], (x, y, z)))
    
    training_set = []
    
    for k, v in builds.items():
        minx, maxx, miny, maxy, minz, maxz = float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')
        block_map = {}
        for block in v:
            rgb, xyz = block
            block_map[xyz] = rgb
            x, y, z = xyz
            minx = min(minx, x)
            maxx = max(maxx, x)
            miny = min(miny, y)
            maxy = max(maxy, y)
            minz = min(minz, z)
            maxz = max(maxz, z)
        sizex, sizey, sizez = maxx - minx + 1, maxy - miny + 1, maxz - minz + 1
        voxel_image = torch.zeros((4, 32, 32, 32)) - 1
        for x in range(32):
            for y in range(32):
                for z in range(32):
                    loc = (int(x * (sizex / 32)) + minx, int(y * (sizey / 32)) + miny, int(z * (sizez / 32)) + minz)
                    if loc in block_map:
                        rgb = block_map[loc]
                        r, g, b = rgb
                        voxel_image[:, x, y, z] = torch.tensor([(r / 255) * 2 - 1, (g / 255) * 2 - 1, (b / 255) * 2 - 1, 1])
        '''for block in v:
            rgb, xyz = block
            r, g, b = rgb
            x, y, z = xyz
            x, y, z = x - minx + (32 - maxx) // 2, y - miny + (32 - maxy) // 2, z - minz + (32 - maxz) // 2
            voxel_image[:, x, y, z] = torch.tensor([(r / 255) * 2 - 1, (g / 255) * 2 - 1, (b / 255) * 2 - 1, 1])'''
        training_set.append(voxel_image)
    
    training_set = training_set[:(len(training_set) // batch_size) * batch_size]
    training_set = torch.cat(training_set).reshape((-1, batch_size, 4, 32, 32, 32))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_epochs = 100000
    noise_size = 100
    
    discriminator = Discriminator().to(device)
    generator = Generator(noise_size).to(device)
    
    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = 0.00002)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr = 0.00002)
    
    if existing_epochs:
        try:
            existing_epochs = int(existing_epochs)
        except:
            print('Bad input for epochs')
            exit()
        path = models_path + 'voxel_' + str(existing_epochs) + '.pt'
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

    data = list(training_set)
    for epoch in range(existing_epochs, existing_epochs + num_epochs):
        random.shuffle(data)
        for i, real_images in enumerate(data):
            noise = torch.randn(batch_size, noise_size, 1, 1, 1).to(device)
            fake_images = generator(noise)
            d_real = discriminator(real_images.to(device))
            d_fake = discriminator(fake_images.detach().to(device))
            real_loss = loss_fn(d_real, torch.ones_like(d_real))
            fake_loss = loss_fn(d_fake, torch.zeros_like(d_fake))
            d_loss = real_loss + fake_loss
            if d_loss >= 0.1:
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
                      .format(epoch + 1, existing_epochs + num_epochs, i + 1, len(training_set), d_loss.item(), g_loss.item()))
        if (epoch + 1) % 100 == 0:
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            torch.save({
                'epoch': epoch + 1,
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_state_dict': generator.state_dict(),
                'discriminator_optimizer_state_dict': d_optimizer.state_dict(),
                'generator_optimizer_state_dict': g_optimizer.state_dict(),
                }, models_path + 'voxel_' + str(epoch + 1) + '.pt')
    
    noise = torch.randn(10, noise_size, 1, 1, 1).to(device)
    fake_images = generator(noise)
    print(fake_images)
    #for image in fake_images:
    #    plt.imshow(torch.permute(image.detach().cpu(), (1, 2, 0)))
    #    plt.show()