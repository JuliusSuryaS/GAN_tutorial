from base import *
import glob

class PanoData(Dataset):
    def __init__(self, root_dir, data_len, transform=None):
        self.root_dir = root_dir
        self.data_len = data_len - 1
        self.transform = transform

    def __getitem__(self, idx):

        sub_dir = 'pano_' + str(idx + 1)

        # in_img_cat = self._concat_img_pad(sub_dir, 'im_').astype(np.float) # cube input
        # in_img_cat = self._read_pano(sub_dir, prefix='pre_input45.jpg', scale=1.0)
        # in_img_cat = self._read_pano(sub_dir, prefix='trained_input.jpg', scale=1.0)
        in_img_cat = self._read_pano(sub_dir, prefix='trained_input_gt.jpg', scale=1.0)
        # in_img_cat = self._read_pano(sub_dir, prefix='trained_input.jpg', scale=1.0)
        gt_img_cat = self._read_pano(sub_dir, prefix='pano_*.jpg.jpg', scale=1.0) # pano groundtruth
        fov = self._read_fov(sub_dir, prefix='fov.txt')
        #in_img_cat = gt_img_cat
        #gt_img_cat = self._concat_img(sub_dir, 'gt_').astype(np.float) # cube groundtruth

        in_img_cat = in_img_cat/127.5 - 1
        gt_img_cat = gt_img_cat/127.5 - 1

        sample = {'input': in_img_cat, 'gt': gt_img_cat, 'fov': fov, 'dir': sub_dir}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.data_len

    def _concat_img(self, sub_dir, prefix='gt_'):
        images = []
        for i in range(6):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread(im_path))

        empty = np.zeros_like(images[0])
        img_top = np.hstack((empty, images[4], empty, empty))
        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_bot = np.hstack((empty, images[5], empty, empty))

        img_concat_full = np.vstack((img_top, img_concat, img_bot))
        return img_concat_full

    def _concat_img_pad(self, sub_dir, prefix='im_'):
        images = []
        for i in range(4):
            im_name = prefix + str(i+1) + '.jpg'
            im_path = os.path.join(self.root_dir, sub_dir, im_name)
            images.append(self._imread_pad(im_path))

        img_concat = np.hstack((images[2], images[0], images[3], images[1]))
        img_concat_full = np.vstack((np.zeros_like(img_concat), img_concat, np.zeros_like(img_concat)))
        return img_concat_full

    def _read_fov(self, sub_dir, prefix='fov.txt'):
        out = np.zeros((1,256))
        file_path = os.path.join(self.root_dir, sub_dir, prefix)
        with open(file_path) as f:
            for line in f:
                idx = line.strip()
                idx = int(idx)
                out[0, idx-1] = 1
        return out

    def _read_pano(self, sub_dir, prefix='pano_', scale=1.0):
        im_path_ = os.path.join(self.root_dir, sub_dir, prefix)
        im_list = glob.glob(im_path_)
        img = self._imread(im_list[0])
        img_rsz = cv2.resize(img, (0,0), fx=scale, fy=scale)
        return img_rsz

    def _imread(self, x):
        img = cv2.imread(x)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _imread_pad(self,x):
        img = self._imread(x)
        img = cv2.resize(img, (128, 128))
        pad = 64
        img = cv2.copyMakeBorder(img, 64, 64, 64, 64, cv2.BORDER_CONSTANT)
        return img

class ToTensor():
    def __call__(self, sample):
        gt_img, in_img, fov, sub_dir = sample['gt'], sample['input'], sample['fov'], sample['dir']

        gt = torch.from_numpy(gt_img.transpose(2,0,1)) # NCHW
        image = torch.from_numpy(in_img.transpose(2,0,1)) # NCHW
        fov = torch.from_numpy(fov)

        gt = gt.type(torch.FloatTensor)
        image = image.type(torch.FloatTensor)
        fov = fov.type(torch.LongTensor)

        return {'input': image, 'gt': gt, 'fov': fov, 'dir': sub_dir}


class ToTensorResize():
    def __call__(self, sample):
        gt_img, in_img, fov, sub_dir = sample['gt'], sample['input'], sample['fov'], sample['dir']

        gt_img = cv2.resize(gt_img, (64,64))
        gt = torch.from_numpy(gt_img.transpose(2,0,1)) # NCHW
        image = torch.from_numpy(in_img.transpose(2,0,1)) # NCHW
        fov = torch.from_numpy(fov)

        gt = gt.type(torch.FloatTensor)
        image = image.type(torch.FloatTensor)
        fov = fov.type(torch.LongTensor)

        return {'input': image, 'gt': gt, 'fov': fov, 'dir': sub_dir}