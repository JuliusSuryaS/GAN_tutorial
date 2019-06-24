from base import *
from panodata import ToTensor, ToTensorResize, PanoData
from networks import DCGAN, CGAN

# Configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_train = '/home/juliussurya/work/360dataset/pano_data_train'
model_dir = './trained_model/model'
batch_sz = 1
workers = 1
train_len = 39730
train_shuffle = True
test_shuffle = False
total_epochs = 100
lr = 0.0002
betas = (0.5, 0.999)

# Prepare Dataset
transform_fn = transforms.Compose([ToTensor()])
custom_dataset = PanoData(data_train, train_len, transform=transform_fn)
data_loader = DataLoader(custom_dataset, batch_size=batch_sz, shuffle=train_shuffle, num_workers=workers)

# Initialize Network
# net = DCGAN(device, lr=lr, betas=betas)
net = CGAN(device, lr=lr, betas=betas)

# Evaluation
fixed_sample = torch.randn(batch_sz, 100, 1, 1, device=device)

# Training Loop
for epoch in range(total_epochs):
    for idx, batch in enumerate(data_loader):
        in_img, gt_img, gt_fov = batch['input'], batch['gt'], batch['fov']
        # in_img = in_img.to(device)
        gt_img = gt_img.to(device)

        # Skip next epoch
        if gt_img.size()[0] != batch_sz:
            break

        # Make input noise
        rand_noise = torch.randn(batch_sz, 100, 1, 1, device=device)

        # Load input and Train
        net.load_input(rand_noise, gt_img)
        # net.train()
        net.train_generator()

        # Print loss
        if idx % 20 == 0:
            net.print_loss(idx)
        # Add image to tensorboard
        if idx % 20 == 0:
            net.write_img_summary(idx, nrow=4)

        # Evaluate and save model
        if idx % 200 == 0:
            # net.evaluate(fixed_sample, idx)
            net.save_model(idx, model_name=model_dir)

