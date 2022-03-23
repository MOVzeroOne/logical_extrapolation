import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt 
from maze_loaders import maze_dataset
from tqdm import tqdm 


class recurrent_block(nn.Module):
    def __init__(self,num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=(3,3), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=(3,3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=num_channels,out_channels=num_channels,kernel_size=(3,3), stride=1, padding=1)
        self.act_f = nn.ReLU()

    def forward(self,x):
        res1 = self.act_f(self.conv2(self.act_f(self.conv1(x))))+x
        res2 = self.act_f(self.conv4(self.act_f(self.conv3(res1))))+x
        return res2

class network_with_recall(nn.Module):
    def __init__(self,input_channels= 2,recurrent_block_channels=32):
        super().__init__()
        self.act_f = nn.ReLU()

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=recurrent_block_channels, kernel_size=(3,3), stride=1, padding=1),self.act_f)
        self.combiner = nn.Sequential(nn.Conv2d(in_channels=input_channels+recurrent_block_channels, out_channels=recurrent_block_channels, kernel_size=(3,3), stride=1, padding=1),self.act_f)
        self.recurrent_block = recurrent_block(recurrent_block_channels)
        self.decoder = nn.Sequential(nn.Conv2d(in_channels=recurrent_block_channels,out_channels=32,kernel_size=(3,3), stride=1, padding=1),self.act_f,nn.Conv2d(in_channels=recurrent_block_channels,out_channels=8,kernel_size=(3,3), stride=1, padding=1),self.act_f,nn.Conv2d(in_channels=8,out_channels=1,kernel_size=(3,3), stride=1, padding=1),nn.Sigmoid())

    
    def forward(self,x,num_recurrent_steps,start=0,step=1):
        unprocessed_input = x 
        x = self.encoder(x) 
        img_list = []
        for i in tqdm(range(num_recurrent_steps),ascii=True,desc="thinking"):
            combined = torch.cat((x,unprocessed_input),dim=1)
            x = self.combiner(combined)
            x = self.recurrent_block(x)
            if(i >= start and i %step==0):
                img_list.append(self.decoder(x).cpu())
        
        return img_list

        



if __name__ == "__main__":
    
    network = network_with_recall()
    network.load_state_dict(torch.load("./maze_models/DT_with_recall87.pt"))
    network.cuda()
    #maze_9by9 = maze_dataset("maze_9by9.csv")
    #maze_13by13 = maze_dataset("maze_13by13.csv")
    maze_59by59 = maze_dataset("maze_59by59.csv") 

    
    fig,axis = plt.subplots(4)

    plt.ion()
    for x,y in DataLoader(maze_59by59,shuffle=True,batch_size=1): 
        with torch.no_grad():
            image_list =network(x.cuda(),num_recurrent_steps=100)
            for index, pred in enumerate(image_list):       
                axis[0].cla()
                axis[1].cla()
                axis[2].cla()
                axis[3].cla()
                axis[0].imshow(x.squeeze()[0])
                axis[1].imshow(x.squeeze()[1])
                axis[2].imshow(y[0].squeeze(dim=0))
                axis[3].imshow(pred[0].squeeze(dim=0))
                            
                plt.pause(0.1)

