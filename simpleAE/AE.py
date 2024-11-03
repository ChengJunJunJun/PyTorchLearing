import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import  matplotlib.pyplot as plt
from torch.autograd import Variable

transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
])

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=48, shuffle=True, num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(# 28 28 1
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),# 28 28 64
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),#14 14 64
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),#14 14 128
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)#7 7 128
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),#14 14 128
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),#14 14 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 28 28 64
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),#28 28 1
            nn.Tanh()
        )

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return decoder

net = Net()
optimzer = optim.Adam(net.parameters(), lr=0.002, weight_decay=1e-5)
loss_func = nn.MSELoss()

loss_list = []
epoch = 10

for i in range(epoch):
    run_loss = 0
    run_acc = 0

    for data in trainloader:
        input, _ = data
        input = Variable(input)
        out = net(input)
        optimzer.zero_grad()
        loss = loss_func(out, input)
        #pred =torch.max(out, 1)[1]
        #correct = (pred == label).sum()
        #run_acc += correct.item()
        run_loss += loss
        loss.backward()
        optimzer.step()

    print('epoch: {:.6f} Train Loss: {:.6f}'.format(i, run_loss / (len(
        trainset))))
    loss_list.append(run_loss)
    if i % 10 == 0:
        pic = to_img(out.cpu().data)
        pic2 = to_img(input.cpu().data)
        torchvision.utils.save_image(pic, './dc_img/image_{}.png'.format(i))
        torchvision.utils.save_image(pic2, './dc_img/image2_{}.png'.format(i))



x = range(epoch)
y = loss_list

plt.plot(x, y)
plt.show()


print('Finished Training')

print('Save Model')