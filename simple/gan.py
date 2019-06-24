from modules import *
import Networks as Networks

batch_size = 64

dataset = datasets.ImageFolder(root='./data/celeba',
                               transform=transforms.Compose([
                                   transforms.Resize(64),
                                   transforms.CenterCrop(64),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5, 0.5, 0.5))
                               ]))
data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)


real_batch = next(iter(data_loader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Training Images')
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.show()


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight.data)
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# GENERATOR
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        x = self.generator(input)
        return x

# DISCRIMINATOR
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        x = self.discriminator(input)
        return x

class LossGAN(nn.Module):
    def __init__(self, loss_type):
        super(LossGAN, self).__init__()
        if loss_type == 'real':
            self.label = torch.full([64], 1, device=device)
        if loss_type == 'fake':
            self.label = torch.full([64], 0, device=device)
        self.loss_fn = nn.BCELoss()

    def forward(self, pred):
        loss = self.loss_fn(pred, self.label.detach())
        return loss



def check_hook(self, input, output):
    # print(self.__class__.__name__)
    if type(self) ==  nn.ConvTranspose2d or type(self) == nn.Conv2d:
        print(self.__class__.__name__)
        print('Input  :', input[0].size())
        print('Output :', output.data.size())

def check_size_seq(model):
    for layer in model:
        layer.register_forward_hook(check_hook)

def save_img(img, img_name):
    img = np.transpose(img, (1,2,0)) * 255.0
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8)
    cv2.imwrite(img_name, img)
    return img

g_net = Generator().to(device)
g_net.apply(init_weights)
print(g_net)
d_net = Discriminator().to(device)
d_net.apply(init_weights)
print(d_net)


# cfg
num_epochs = 100
iters = 0

fixed_noise = torch.randn(64, 100, 1, 1, device=device)


loss_fn = nn.BCELoss()
optimizer_d = optim.Adam(d_net.parameters(), lr=0.0002, betas=(0.5,0.999))
optimizer_g = optim.Adam(g_net.parameters(), lr=0.0002, betas=(0.5,0.999))


# OPTIMIZATION LOOP
for epoch in range(num_epochs):
    # EACH STEP
    for i, data  in enumerate(data_loader, 0):

        # Discriminator
        # ====================
        # clear gradient
        d_net.zero_grad()

        # Real data
        real_data = data[0].to(device) # initiate data and move to gpu
        if real_data.size()[0] < batch_size:
            continue
        output_real = d_net(real_data).view(-1) # feed to Discriminator
        # loss_real = LossGAN('real')(output_real) # calculate loss
        label = torch.full( [real_data.size()[0]], 1, device=device)
        loss_real = loss_fn(output_real, label)
        loss_real.backward() # backward
        d_x = output_real.mean().item() # get loss d from output

        # Fake data
        noise = torch.randn(batch_size, 100, 1, 1, device=device) # create random noise, move to gpu
        fake_data = g_net(noise) # Generate Fake data
        output_fake = d_net(fake_data.detach()).view(-1)
        label = torch.full([real_data.size()[0]], 0, device=device)
        # loss_fake = LossGAN('fake')(output_fake)
        loss_fake = loss_fn(output_fake, label)
        loss_fake.backward()
        d_g_z = output_fake.mean().item()

        # Total loss
        loss_d = loss_real + loss_fake

        # Update D
        optimizer_d.step()


        # Generator
        # ====================
        # clear gradient
        g_net.zero_grad()
        output_g = d_net(fake_data).view(-1)
        label = torch.full([real_data.size()[0]], 1, device=device)
        # loss_g = LossGAN('real')(output_g)
        loss_g = loss_fn(output_g, label)
        loss_g.backward()
        d_g = output_g.mean().item()

        # Update G
        optimizer_g.step()

        if i % 50 == 0:
            print('Epoch', epoch, '/', num_epochs, '-', i, 'Real', loss_d.item(), '|| Fake', loss_g.item())

        if iters % 500 == 0 or (epoch == num_epochs-1 and i == len(data_loader) - 1):
            with torch.no_grad():
                fake = g_net(fixed_noise).detach().cpu()
                img = vutils.make_grid(fake, padding=2, normalize=True)
                save_img(img.numpy(), './output/im_' + str(iters) + '.png')

        iters += 1















