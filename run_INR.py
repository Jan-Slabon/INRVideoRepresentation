from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch import optim
import PIL
import torchvision
from torchvision.transforms import ToTensor, Compose
from PIL import Image

def embeds(x: Tensor) -> Tensor:
    """return tensor of embeddings in trigonometric base function"""
    frequencies = torch.linspace(0,20, steps = 10)
    fun = [torch.sin, torch.cos]
    embed = [f(x * freq) for f in fun for freq in frequencies]
    return torch.concat([x, torch.concat(embed, dim = -1)], dim = -1)

def is_zero(input : Tensor):
    """Checks if all entries of the tensor are zero"""
    are_entries_zero = True
    for el in input:
        are_entries_zero = are_entries_zero and (el == 0)
    return are_entries_zero

def sample_tensor(video : torch.Tensor, class1_weight : int, class2_weight : int) -> List[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Takes a Tensor and returns list with [ ( [ t, x, y ], [ r, g, b ], [ r, g, b ] ) ]
        where second rgb triplet is from t - delta_t at (x , y)"""
    pixels : List[Tuple[Tensor, Tensor]] = []
    val_set : List[Tensor] = []
    i : int = 0
    max_t = 0
    for (t, image) in enumerate(video):
        for (x, row) in enumerate(image):
            for (y, col) in enumerate(row):
                pixels.append((Tensor([t,x,y]).to(device),col.to(device)))
                val_set.append(torch.unsqueeze( Tensor([t,x,y]), dim = 0))
                i+=1
    return pixels, torch.cat(val_set, dim = 0)

def query_network(net : nn.Sequential, data : Tensor, first_image : Tensor) -> Tensor:
    """Reconstructs image from neural representation"""
    w = 64 ; h = 64 ; t = 2 ; fps = 15
    frame_size = w*h
    net.eval()
    inter = Interpolation(first_image)
    video : Tensor = torch.zeros([fps*t,w,h,3])
    video_rgb : Tensor = torch.zeros([fps*t,w,h,3])
    video_vector : Tensor = torch.zeros([fps*t,w,h,3])
    for time in range(fps*t-1):
        with torch.no_grad():
            vector, rgb = net(embeds(data[time*frame_size:time*frame_size+frame_size]))
            #vector, rgb = (vector.cpu().detach(), rgb.cpu().detach())
            #print(rgb.size(), "pixels from",time*frame_size, "to", time*frame_size+frame_size, "at time", time)
            if time != 0:
                video[time] = torch.reshape(inter(vector * 63) + rgb, (64,64,3))
                video_rgb[time] = torch.reshape(rgb, (64,64,3))
                video_vector[time] = torch.reshape(inter(vector * 63), (64,64,3))
            else:
                video[time] = first_image
                video_rgb[time] = first_image
                video_vector[time] = first_image
    net.train()
    return video, video_rgb, video_vector

def plot_dataset(model : nn.Sequential, epoch_num : int, data : Tensor, first_image : Tensor):
    """Saves trainig progress"""
    video, rgb, vector= query_network(model, data, first_image)
    frame = video[2].clone().detach()
    frame = torch.moveaxis(frame, 2, 0)
    torchvision.io.write_video("results/video"+str(epoch_num)+"_base.mp4", torch.tensor(video * 255, dtype = torch.uint8), 15)
    torchvision.io.write_video("results/video"+str(epoch_num)+"_rgb.mp4", torch.tensor(rgb * 255, dtype = torch.uint8), 15)
    torchvision.io.write_video("results/video"+str(epoch_num)+"_vector.mp4", torch.tensor(vector * 255, dtype = torch.uint8), 15)

def get_data() -> Tensor:
    """loads data from directory of photos"""
    transform = Compose([ToTensor()])
    framelist : List[Tensor] = []
    for i in range(1,31):
        photo = Image.open("data/image"+str(i)+".jpg").convert('RGB')
        photo = photo.resize((64,64), PIL.Image.BILINEAR)
        photo = transform(photo)
        photo = torch.moveaxis(photo, 0, 2).unsqueeze(0)
        framelist.append(photo)
    return torch.cat(framelist, dim = 0)

def tensor_size(input : Tensor):
    n = 1
    for el in input.size():
        if el != 0:
            n*=el
    return n

class INR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(63,250),
            nn.ReLU(), nn.Linear(250, 250),
            nn.ReLU(), nn.Linear(250, 250),
            nn.ReLU(), nn.Linear(250, 250),
            nn.ReLU(), nn.Linear(250, 250))
        self.vector_head = nn.Sequential(
            nn.ReLU(), nn.Linear(250, 125),
            nn.ReLU(), nn.Linear(125, 75),
            nn.ReLU(), nn.Linear(75, 50),
            nn.ReLU(), nn.Linear(50, 2),
            nn.Tanh())
        self.rgb_head = nn.Sequential(
            nn.ReLU(), nn.Linear(250, 250),
            nn.ReLU(), nn.Linear(250, 125),
            nn.ReLU(), nn.Linear(125, 75),
            nn.ReLU(), nn.Linear(75, 3),
            nn.Tanh())

    def forward(self, x):
        embed = self.mlp(x)
        sigma = self.vector_head(embed)
        rgb = self.rgb_head(embed)
        return sigma, rgb

class Interpolation(nn.Module):
    def __init__(self, image) -> None:
        super().__init__()
        self.image = image
    
    def forward(self, x):
        low = torch.floor(x).to(int)
        high = (low + 1)
        left_bottom_pixel = self.image[torch.minimum(low, torch.tensor([63.]).to(device)).T.tolist()]
        left_top_pixel = self.image[(torch.minimum(low + torch.tensor([0, 1]).to(device), torch.tensor([63.]).to(device))).T.tolist()] # 
        right_bottom_pixel = self.image[(torch.minimum(high + torch.tensor([0, -1]).to(device), torch.tensor([63.]).to(device))).T.tolist()]
        right_top_pixel = self.image[torch.minimum(high, torch.tensor([63.]).to(device)).T.tolist()]

        high = high.T
        x = x.T
        low = low.T

        fy1 = (high[0] - x[0])/(high[0] - low[0])*left_bottom_pixel.T
        + (x[0] - low[0])/(high[0] - low[0])*right_bottom_pixel.T

        fy2 = (high[0] - x[0])/(high[0] - low[0])*left_top_pixel.T
        + (x[0] - low[0])/(high[0] - low[0])*right_top_pixel.T

        fxy = (high[1] - x[1])/(high[1] - low[1])*fy1
        + (x[1] - low[1])/(high[1] - low[1])*fy2

        return fxy.T

device = 'cuda'

network = INR().to(device)
frame_tensor = get_data()
benchmark = frame_tensor[0].to(device)
inter = Interpolation(benchmark).to(device)
weight_nonzero = 0.4
weight_zero = 0.6
data, val_data = sample_tensor(frame_tensor, weight_nonzero, weight_zero)
#torchvision.io.write_video("results/benchmark.mp4", torch.tensor(results * 255, dtype = torch.uint8), 4)
val_data = val_data.to(device)
epochs : int = 1000
lr : float = 0.001
batch_size : int = 4096
C : float = 10
optimizer = optim.Adam(network.parameters(),lr=lr) #Adam makes learning very fast but unstable
cost_mse = nn.MSELoss()
dataset = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)
cost_bce = nn.BCELoss()
for i in range(2):
    l = 0
    for (x,y) in dataset:
        network.zero_grad()
        (vector, rgb) = network(embeds(x))
        loss = cost_mse(vector*20, torch.tensor([0.]).to(device))
        loss.backward()
        l += loss.detach().item()
        optimizer.step()
    print("pretraining loss:", l)

for epoch in range(1,epochs+1):
    record_loss = 0
    for (x,y) in dataset:
        network.zero_grad()
        (vector, rgb) = network(embeds(x))
        loss = cost_mse(inter(x[:,1:2] + vector*20), y)
        loss.backward()
        record_loss += loss.detach().item()
        optimizer.step()
        
    if epoch % 20 == 0:
        plot_dataset(network, epoch, val_data, benchmark)
    print("Epoch: "+str(epoch)+" loss = " + str(record_loss))
