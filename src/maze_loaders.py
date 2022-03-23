import torch 
from torch.utils.data import Dataset
import pandas as pd 
import pickle 


class maze_dataset(Dataset):
    def __init__(self,path,name_series="maze"):
        self.path = path 
        self.name = name_series
        self.data = pd.read_csv(self.path)[self.name]
        
    def __len__(self):
        return self.data.size
    
    def __getitem__(self,index):
        maze,path,start_end = pickle.loads(eval(self.data[index]))
        x = torch.cat((maze.unsqueeze(dim=0),start_end.unsqueeze(dim=0)),dim=0)
        y = path.unsqueeze(dim=0)
        return (x,y)
