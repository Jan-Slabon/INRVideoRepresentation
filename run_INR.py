from typing import List, Tuple
import torch
from torch import nn, Tensor
from torch import optim
import PIL
import torchvision
from torchvision.transforms import ToTensor, Compose
from PIL import Image
W = 128
H = 128
T = 6
FPS = 15
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
    prev_val : List[Tensor] = []
    true_output : List[Tensor] = []
    i : int = 0
    max_t = 0
    for (t, image) in enumerate(video):
        for (x, row) in enumerate(image):
            for (y, col) in enumerate(row):
                diff = col - ( pixels[i-H * W][1] if i - H * W >= 0 else Tensor([0,0,0]))
                sigma = (Tensor([0]) if is_zero(diff) else Tensor([1]))
                w = Tensor([class1_weight]) if sigma[0] == 1 else Tensor([class2_weight])
                pixels.append((Tensor([t/2,x/4,y/4]),col,diff,sigma,w))
                val_set.append(torch.unsqueeze( Tensor([t/2,x/4,y/4]), dim = 0))
                prev_val.append( torch.unsqueeze( pixels[i-H * W][1] if i - H * W >= 0 else Tensor([0,0,0]), dim = 0) )
                true_output.append(diff)
                i+=1
        max_t = t
    prev_val = torch.cat(prev_val, dim = 0)
    true_output = torch.cat(true_output, dim = 0)
    return pixels, torch.cat(val_set, dim = 0), torch.reshape(prev_val, (max_t+1, W, H, 3)), torch.reshape(true_output, (max_t+1, W, H, 3))


def query_network(net : nn.Sequential, data : Tensor, prev_tensor : Tensor) -> Tensor:
    """Reconstructs image from neural representation"""
    frame_size = W * H
    net.eval()
    video : Tensor = torch.zeros([FPS*T,W,H,3])
    for time in range(FPS*T-1):
        with torch.no_grad():
            sigma, rgb = net(embeds(data[time*frame_size:time*frame_size+frame_size]))
            sigma, rgb = (sigma.cpu().detach(), rgb.cpu().detach())
            #print(rgb.size(), "pixels from",time*frame_size, "to", time*frame_size+frame_size, "at time", time)
            if time != 0:
                video[time] = (torch.reshape(sigma * rgb, (W,H,3)) + video[time - 1]).minimum(torch.tensor(1)).maximum(torch.tensor(0))
            else:
                video[time] = torch.reshape(sigma * rgb, (W,H,3))
    net.train()
    return video

def plot_dataset(model : nn.Sequential, epoch_num : int, data : Tensor, prev_data : Tensor):
    """Saves trainig progress"""
    video = query_network(model, data, prev_data)
    frame = video[2].clone().detach()
    frame = torch.moveaxis(frame, 2, 0)
    torchvision.io.write_png(torch.tensor(frame * 255, dtype=torch.uint8),"results/frame"+str(epoch_num)+".png")
    torchvision.io.write_video("results/video"+str(epoch_num)+".mp4", torch.tensor(video.minimum(torch.tensor(1)).maximum(torch.tensor(0)) * 255, dtype = torch.uint8), 15)

def get_data() -> Tensor:
    """loads data from directory of photos"""
    transform = Compose([ToTensor()])
    framelist : List[Tensor] = []
    for i in range(1,FPS*T+1):
        photo = Image.open("data/image"+str(i)+".jpg").convert('RGB')
        photo = photo.resize((W,H), PIL.Image.BILINEAR)
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
        self.mlp = nn.Sequential(nn.Linear(63,256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 256))
        self.sigma_head = nn.Sequential(
            nn.ReLU(), nn.Linear(256, 128),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 25),
            nn.ReLU(), nn.Linear(25,1),
            nn.Sigmoid())
        self.rgb_head = nn.Sequential(
            nn.ReLU(), nn.Linear(256, 256),
            nn.ReLU(), nn.Linear(256, 128),
            nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 3),
            nn.Tanh())

    def forward(self, x):
        embed = self.mlp(x)
        sigma = self.sigma_head(embed)
        rgb = self.rgb_head(embed)
        return sigma, rgb




device = "cuda"
network = INR().to(device)
frame_tensor = get_data()
weight_nonzero = 0.4
weight_zero = 0.6
data, val_data, prev_val_data, results = sample_tensor(frame_tensor, weight_nonzero, weight_zero)
    # non_zeros = torch.count_nonzero(results)
    # full = tensor_size(results)
    # print(non_zeros/full)
val_data = val_data.to(device)
epochs : int = 1000
lr : float = 0.001
batch_size : int = H * W
optimizer = optim.Adam(network.parameters(),lr=lr)
cost_mse = nn.L1Loss()
dataset = torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = True)

for epoch in range(1,epochs+1):
    record_loss_rgb = 0
    record_loss_sigma = 0
    for (x,y,exp,sig,w) in dataset:

        (x,y,exp,sig,w) = (x.to(device), y.to(device), exp.to(device), sig.to(device), w.to(device))
        network.zero_grad()

        cost_bce = nn.BCELoss(w)
        (sigma, _) = network(embeds(x))
        loss_sig = cost_bce(sigma, sig)
        record_loss_sigma += loss_sig.detach().item()
        loss_sig.backward()
        optimizer.step()

        network.zero_grad()
        (_, rgb) = network(embeds(x))
        loss_rgb = cost_mse(sig * rgb, exp)
        record_loss_rgb += loss_rgb.detach().item()
        loss_rgb.backward()
        optimizer.step()
        
    if epoch % 20 == 0:
        plot_dataset(network, epoch, val_data, prev_val_data)
    print("Epoch: "+str(epoch)+" loss sigma = " + str(record_loss_sigma), "loss rgb", record_loss_rgb)
