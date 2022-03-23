import torch 
import torch.nn as nn 
import torch.optim as optim 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader 
from maze_loaders import maze_dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import numpy as np 
import torchvision




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

    
    def forward(self,x,num_recurrent_steps):
        unprocessed_input = x 
        x = self.encode(x) 
        x = self.recurrent_process(x,unprocessed_input,num_recurrent_steps)
        return self.decode(x)
    
    def encode(self,x):
        return self.encoder(x) 

    def decode(self,x):
        return self.decoder(x)

    def recurrent_process(self,x,unprocessed_input,num_recurrent_steps):
        for _ in range(num_recurrent_steps):
            combined = torch.cat((x,unprocessed_input),dim=1)
            x = self.combiner(combined)
            x = self.recurrent_block(x)
        return x



if __name__ == "__main__":

    #datasets
    maze_9by9 = maze_dataset("maze_9by9.csv")
    maze_13by13 = maze_dataset("maze_13by13.csv")
    maze_59by59 = maze_dataset("maze_59by59.csv")
    #hyperparam
    lr = 0.0001
    epochs = 1000
    train_dataset = maze_9by9
    batch_size = 32
    num_images = 1
    M = 30 #max recurrence length
    alpha = 0.1 #weights the partial and standard loss
    beta = 0.1 #weights reconstruction loss and the standard loss
    noise_ratio = 0.1
    test_depth =500 
    #init
    net = network_with_recall()
    net.cuda()
    optimizer = optim.Adam(net.parameters(),lr=lr)
    writer = SummaryWriter()

    #train 
    j = 0
    for i in range(epochs):
        #train
        for x,y in tqdm(DataLoader(train_dataset,shuffle=True,batch_size=batch_size,num_workers=3),ascii=True,desc="training"):
            x,y = x.cuda(),y.cuda()
            optimizer.zero_grad()
            #calculate partial loss
            n = np.random.choice(np.arange(M)) #[0,M-1] 
            k = np.random.choice(np.arange(M-n)+1) #[1,M-n]

            with torch.no_grad(): #n steps without tracking gradients
                encoded_input = net.encode(x)
                partial_solution = net.recurrent_process(encoded_input,x,n)
            partial_output = net.decode(net.recurrent_process(partial_solution,x,k)) #k steps with tracking gradients

            progressive_loss = nn.BCELoss()(partial_output,y)


            #calculate reconstruction loss
            n = np.random.choice(np.arange(M)) #[0,M-1] 
            k = np.random.choice(np.arange(M-n)+1) #[1,M-n]

            with torch.no_grad(): #n steps without tracking gradients
                encoded_input = net.encode(x)
                partial_solution = net.recurrent_process(encoded_input,x,n)
            partial_output = net.decode(net.recurrent_process(partial_solution+noise_ratio*torch.randn(partial_solution.size()).cuda(),x,k)) #k steps with tracking gradients

            reconstruction_loss = nn.BCELoss()(partial_output,y)

            #calculate standard loss
            output = net(x,M)
            standard_loss = nn.BCELoss()(output,y)

            #total loss calc
            total_loss =  (1-alpha-beta)*standard_loss + alpha*progressive_loss + beta*reconstruction_loss
            total_loss.backward()
            optimizer.step()
            #visualization 
            writer.add_scalar("loss",total_loss.detach().item(),j)
            j += 1

        #save model visualize epoch loss
        torch.save(net.state_dict(),"./maze_models/DT_with_recall"+str(i) + ".pt")

        #testing
        with torch.no_grad():
            #IMAGES batchsize 1 for a range of different M.
            
            image_grid = []
            for index,(x,y) in enumerate(DataLoader(maze_9by9,shuffle=False,batch_size=1)):
                x,y = x.cuda(),y.cuda()
                image = net(x,test_depth)
                image_grid.append(image)
                if(index >= num_images):
                    break
            img_grid = torchvision.utils.make_grid(torch.cat(image_grid,dim=0))
            writer.add_image("9by9", img_grid,global_step=i)

            image_grid = []
            for index,(x,y) in enumerate(DataLoader(maze_13by13,shuffle=False,batch_size=1)):
                x,y = x.cuda(),y.cuda()
                image = net(x,test_depth)
                image_grid.append(image)
                if(index >= num_images):
                    break 
            img_grid = torchvision.utils.make_grid(torch.cat(image_grid,dim=0))
            writer.add_image("13by13", img_grid,global_step=i)
            
            image_grid = []
            for index,(x,y) in enumerate(DataLoader(maze_59by59,shuffle=False,batch_size=1)):
                x,y = x.cuda(),y.cuda()
                image = net(x,test_depth)
                image_grid.append(image)              
                if(index >= num_images):
                    break 
            img_grid = torchvision.utils.make_grid(torch.cat(image_grid,dim=0))
            writer.add_image("59by59", img_grid,global_step=i)
    writer.close()
