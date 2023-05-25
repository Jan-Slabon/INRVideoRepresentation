from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch import optim
import matplotlib.pyplot as plt
import PIL
import torchvision
from torchvision.transforms import ToTensor, Compose
from PIL import Image

def embeds(x: Tensor) -> Tensor:
    """return tensor of embeddings in trigonometric base function"""
    frequencies = torch.linspace(0,10, steps = 10)
    fun = [torch.sin, torch.cos]
    embed = [f(x * freq) for f in fun for freq in frequencies]
    return torch.concat([x, torch.concat(embed, dim = -1)], dim = -1)



def sample_tensor(video : torch.Tensor) -> List[Tuple[Tensor, Tensor]]:
    """Takes a Tensor and returns list with [ ( [ t, x, y ], [ r, g, b ] ) ]"""
    pixels : List[Tuple[Tensor, Tensor]] = []
    val_set : List[Tensor] = []
    for (t, image) in enumerate(video): 
        for (x, row) in enumerate(image):
            for (y, col) in enumerate(row):
                pixels.append((Tensor([t/2,x/16,y/16]),col))
                val_set.append(torch.unsqueeze( Tensor([t/2,x/16,y/16]), dim = 0))
    return pixels, torch.cat(val_set, dim = 0)


def query_network(net : nn.Sequential, data : Tensor) -> Tensor:
    """Reconstructs image from neural representation"""
    w = 64 ; h = 64 ; t = 7 ; fps = 4
    frame_size = w*h
    net.eval()
    video : Tensor = torch.zeros([fps*t,w,h,3])
    for time in range(fps*t-1):
        with torch.no_grad():
            rgb = net(embeds(data[time*frame_size:time*frame_size+frame_size])).cpu().detach()
            #print(rgb.size(), "pixels from",time*frame_size, "to", time*frame_size+frame_size, "at time", time)
            video[time] = torch.reshape(rgb, (64,64,3))
    net.train()
    return video

def plot_dataset(model : nn.Sequential, epoch_num : int, data : Tensor):
    """Saves trainig progress"""
    video = query_network(model, data)
    frame = video[2].clone().detach()
    frame = torch.moveaxis(frame, 2, 0)
    torchvision.io.write_png(torch.tensor(frame * 255, dtype=torch.uint8),"results/frame"+str(epoch_num)+".png")
    plt.close()
    torchvision.io.write_video("results/video"+str(epoch_num)+".mp4", torch.tensor(video * 255, dtype = torch.uint8), 4)

def get_data() -> Tensor:
    """loads data from file of photos"""
    transform = Compose([ToTensor()])
    framelist : List[Tensor] = []
    for i in range(1,28):
        photo = Image.open("data/image"+str(i)+".jpg").convert('RGB')
        photo = photo.resize((64,64), PIL.Image.BILINEAR)
        photo = transform(photo)
        photo = torch.moveaxis(photo, 0, 2).unsqueeze(0)
        framelist.append(photo)
    return torch.cat(framelist, dim = 0)

network : nn.Sequential = nn.Sequential(nn.Linear(63,250),
    nn.ReLU(), nn.Linear(250, 250),
    nn.ReLU(), nn.Linear(250, 250),
    nn.ReLU(), nn.Linear(250, 250),
    nn.ReLU(), nn.Linear(250, 125),
    nn.ReLU(), nn.Linear(125, 75),
    nn.ReLU(), nn.Linear(75, 3), nn.Sigmoid()).cuda()




frame_tensor = get_data()
data, val_data = sample_tensor(frame_tensor)
val_data = val_data.cuda()
epochs : int = 200
lr : float = 0.002
batch_size : int = 500
optimizer = optim.Adam(network.parameters(),lr=lr)
cost_fn = nn.MSELoss()
dataset = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)

for epoch in range(epochs):
    record_loss = 0
    for (x,y) in dataset:
        (x,y) = (x.cuda(), y.cuda())
        network.zero_grad()
        res = network(embeds(x))
        loss = cost_fn(res, y)
        loss.backward()
        record_loss += loss.detach().item()
        optimizer.step()
    if epoch % 10 == 0 and epoch != 0 or epoch == 199:
        plot_dataset(network, epoch, val_data)
    print("Epoch: "+str(epoch)+" loss = " + str(record_loss))
