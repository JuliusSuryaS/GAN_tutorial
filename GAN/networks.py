from base import *
import utilities as utl
import base_model as bm

class BaseNet():
    def __init__(self):
        # Writer
        self.writer = SummaryWriter('./log')
        # Network
        self.Gnet = None
        self.Dnet = None
        # Total Loss
        self.loss_g_total = None
        self.loss_d_total = None
        # Out
        self.real_img = None
        self.g_out = None

        # Loss Fn
        self.loss_fn = nn.BCELoss()

    def initialize_weights(self):
        self.Gnet.apply(bm.weights_init)
        self.Dnet.apply(bm.weights_init)

    def save_model(self, step, model_name='model'):
        print('Saving model at step ' + str(step))
        model_path = os.path.join(model_name + '_' + str(step))
        torch.save({'Generator': self.Gnet.state_dict(),
                    'Discriminator' : self.Dnet.state_dict()},
                   model_path + '.pt')
        print('Finished saving model')

    def load_model(self, model_name, strict=True):
        print('Loading model')
        weights = torch.load(model_name)
        self.Gnet.load_state_dict(weights['Generator'], strict=True)
        self.Dnet.load_state_dict(weights['Discriminator'], strict=True)
        print('Finished loading model')

    def load_model_G(self, model_name, strict=True):
        print('Loading model')
        weights = torch.load(model_name)
        self.Gnet.load_state_dict(weights['Generator'], strict=True)
        print('Finished loading model')

    def evaluate(self, in_noise, step):
        g_out_eval = self.Gnet(in_noise)
        g_out_eval_im = make_grid(g_out_eval, normalize=True)
        self.writer.add_image('Out/Eval', g_out_eval_im, step)

    def print_loss(self, iter):
        loss_g = self.loss_g_total.mean().item()
        loss_d = self.loss_d_total.mean().item()
        print('[Iter %d] -> Loss G : %.4f | Loss D : %.4f' %(iter, loss_g, loss_d))
        self.writer.add_scalar('Discriminator', loss_d, iter)
        self.writer.add_scalar('Generator', loss_g, iter)

    def write_img_summary(self, step):
        gt_im = make_grid(self.real_img, normalize=True)
        out_im = make_grid(self.g_out, normalize=True)
        self.writer.add_image('GT', gt_im, step)
        self.writer.add_image('Out/Train', out_im, step)

    def set_phase(self, phase):
        if phase == 'test':
            self.Gnet.eval()
            self.Dnet.eval()
        else:
            self.Gnet.train()
            self.Dnet.train()

class DCGAN(BaseNet):
    def __init__(self, device, phase='train',lr=0.0002, betas=(0.5,0.999)):
        super(DCGAN, self).__init__()
        self.Gnet = bm.DCGAN_Generator().to(device)
        self.Dnet = bm.DCGAN_Discriminator().to(device)
        self.Gnet.apply(bm.weights_init)
        self.Dnet.apply(bm.weights_init)
        self.set_phase(phase)

        self.writer = SummaryWriter('./log')

        self.in_noise = None
        self.real_img = None
        self.loss_fn = nn.BCELoss()

        self.optimG = optim.Adam(self.Gnet.parameters(), lr=lr, betas=(betas))
        self.optimD = optim.Adam(self.Dnet.parameters(), lr=lr, betas=(betas))

    def load_input(self, in_noise, real_img):
        self.in_noise = in_noise
        self.real_img = real_img

    def train(self):
        # Clear gradient
        self.Gnet.zero_grad()
        self.Dnet.zero_grad()

        # Forward G and D
        d_out_real = self.Dnet(self.real_img)
        g_out_fake = self.Gnet(self.in_noise)
        d_out_fake = self.Dnet(g_out_fake.detach())
        d_out_adv = self.Dnet(g_out_fake)

        ones = torch.ones_like(d_out_real)
        zeros = torch.zeros_like(d_out_real)

        # Compute Loss D
        loss_d_real = self.loss_fn(d_out_real, ones)
        loss_d_real.backward()
        loss_d_fake = self.loss_fn(d_out_fake, zeros)
        loss_d_fake.backward()
        loss_d_total = loss_d_real + loss_d_fake
        self.optimD.step()

        # Compute Loss G
        loss_d_adv = self.loss_fn(d_out_adv, ones)
        loss_d_adv.backward()
        # Update G
        self.optimG.step()

        self.loss_d_total = loss_d_total
        self.loss_g_total = loss_d_adv
        self.g_out = g_out_fake


class CGAN(BaseNet):
    def __init__(self, device, phase='train', lr=0.0002, betas=(0.5,0.999)):
        super(CGAN, self).__init__()
        self.device = device
        self.Gnet = bm.MGenerator1().to(device)
        # self.Dnet = bm.MDiscriminator1().to(device)
        self.Dnet = bm.conv_sn_leak(10,10,3,2,1)
        self.Gnet.apply(bm.weights_init)
        # self.initialize_weights()
        self.set_phase(phase)

        self.loss_fn = nn.BCELoss()
        self.loss_pix_fn = nn.L1Loss()
        self.optimG = optim.Adam(self.Gnet.parameters(), lr=lr, betas=betas)
        self.optimD = optim.Adam(self.Dnet.parameters(), lr=lr, betas=betas)

    def load_input(self, in_img, gt_img):
        mask = utl.create_mask_ul().to(self.device)
        self.in_img = gt_img * mask
        self.gt_img = gt_img

        # self.in_img = F.interpolate(self.in_img, (64,64), mode='bilinear')
        # self.gt_img = F.interpolate(self.gt_img, (64,64), mode='bilinear')
        self.in_img = F.interpolate(self.in_img, scale_factor=4, mode='bilinear')
        self.gt_img = F.interpolate(self.gt_img, scale_factor=4, mode='bilinear')

    def train_generator(self):
        self.Gnet.zero_grad()
        g_out_fake = self.Gnet(self.in_img)
        loss_g_pix = self.loss_pix_fn(g_out_fake, self.gt_img)
        loss_g_pix.backward()
        self.optimG.step()

        self.loss_d_total = loss_g_pix
        self.loss_g_total = loss_g_pix
        self.g_out = g_out_fake

    def train(self):
        self.Dnet.zero_grad()

        # Update D
        d_out_real = self.Dnet(self.gt_img)
        g_out_fake = self.Gnet(self.in_img)
        d_out_fake = self.Dnet(g_out_fake.detach())

        ones = torch.ones_like(d_out_real)
        zeros = torch.zeros_like(d_out_real)

        loss_d_real = self.loss_fn(d_out_real, ones)
        loss_d_fake = self.loss_fn(d_out_fake, zeros)
        loss_d_total = 0.5 * (loss_d_fake + loss_d_real)
        loss_d_total.backward()
        self.optimD.step()

        self.Gnet.zero_grad()

        # Update G
        d_out_adv = self.Dnet(g_out_fake)
        loss_adv = self.loss_fn(d_out_adv, ones)
        loss_g_pix = self.loss_pix_fn(g_out_fake, self.gt_img)
        loss_g_total = loss_adv + 10 * loss_g_pix
        loss_g_total.backward()
        self.optimG.step()

        self.loss_d_total = loss_d_total
        self.loss_g_total = loss_g_total
        self.g_out = g_out_fake

    def write_img_summary(self, step, nrow=8):
        in_im = make_grid(self.in_img, nrow, normalize=True)
        gt_im = make_grid(self.gt_img, nrow, normalize=True)
        out_im = make_grid(self.g_out, nrow, normalize=True)
        self.writer.add_image('GT', gt_im, step)
        self.writer.add_image('Out/Train', out_im, step)
        self.writer.add_image('Input', in_im, step)

