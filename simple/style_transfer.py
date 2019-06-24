from modules import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128

tsfm = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name)
    # image = cv2.imread(image_name)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = tsfm(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader('./data/style.jpg')
content_img = image_loader('./data/content.jpg')



# ANIMATE PLT SHOW
plt.ion()
def imshow(tensor, title=None):
    tensorToImg = transforms.ToPILImage()

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = tensorToImg(image)
    plt.imshow(image)
    plt.pause(2)

plt.figure()
imshow(style_img, 'Style Image')
imshow(content_img, 'Content Image')


#  content loss
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a * b * c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# style loss
class StyleLoss(nn.Module):
    def __init__(self, target_features):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_features).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.losss = F.mse_loss(x, self.target)
        return x


# PRETRAINED MODEL
vgg = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, x):
        return (x - self.mean)/self.std




