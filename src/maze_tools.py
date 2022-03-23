import torch 
import numpy as np 
from collections import defaultdict
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import pandas as pd 
from os.path import exists
import pickle 


class maze_generator():
    """
    implementation of Wilson's algorithm for maze generation    
    """
    def __init__(self,num_height_cells,num_width_cells):

        self.num_width_cells = num_width_cells
        self.num_height_cells = num_height_cells
        self.width = self.num_width_cells*2+1
        self.height = self.num_height_cells*2 +1
    
    def size(self):
        return (self.num_height_cells+self.num_height_cells-1,self.num_width_cells+ self.num_width_cells-1)
    
    def _set_cell_road(self,node):
        """sets a cell to be a road"""
        self.maze[node[1]][node[0]] = 1
    
    def _set_cell_wall(self,node):
        """sets a cell to be a wall"""
        self.maze[node[1]][node[0]] = 0

    def _sample_univisted_node(self):
        """returns a randomly sampled unvisted node"""
        return self.unvisted_list[np.random.choice(np.arange(len(self.unvisted_list)))]

    def _visit(self,node):
        """sets a node as visited and sets it as a road (as walls cant be visited)"""
        if(not self.visited[node]):
            self.unvisted_list.remove(node)
            self.visited[node] = True
            self._set_cell_road(node)

    def generate(self,maze_num=0,plot=False,plot_delay=10):
        """generates maze"""
        self.maze = torch.zeros(self.height,self.width)
        self.visited = defaultdict(bool)
        self.unvisted_list = [(x,y) for x in range(self.width) for y in range(self.height) if (x % 2 == 1 and y % 2 == 1)]
        self.directions = defaultdict(tuple)

        self.start_node = self._sample_univisted_node()
        self._visit(self.start_node)
        plt.ion()
        i = 0
        with tqdm(total=len(self.unvisted_list),ascii=True,desc="maze "+str(maze_num)+ " generation") as pbar:
            while(len(self.unvisted_list) > 0):
                before = len(self.unvisted_list)
                self._random_walk()
                after = len(self.unvisted_list)
                pbar.update(before-after)
                
                if(plot and i % plot_delay == 0):
                    plt.cla()
                    plt.imshow(self.maze)
                    plt.pause(0.1)
                i += 1
        if(plot):
            plt.cla()
            plt.imshow(self.maze)
            plt.waitforbuttonpress()
        plt.ioff()
        return self.maze 

    def _valid_node(self,node):
        """tests if a node is valid/legal"""
        return node[0] >= 1 and node[1] >= 1 and node[0] < self.width and node[1] < self.height

    def _generate_neighbors(self,node):
        """returns a list of neighbors"""
        up_neighbor = (node[0],node[1]+2)
        down_neighbor = (node[0],node[1]-2)
        left_neighbor = (node[0]-2,node[1])
        right_neighbor = (node[0]+2,node[1])
        neighbor_list = [neighbor for neighbor in [up_neighbor,down_neighbor,left_neighbor,right_neighbor] if(self._valid_node(neighbor))]
        return neighbor_list
    
    def _random_walk(self):
        
        start_node = self._sample_univisted_node()
        current_node = start_node
        neighbor_list = self._generate_neighbors(current_node)

        while(True):
            chosen_neighbor = neighbor_list[np.random.choice(np.arange(len(neighbor_list)))]
            self.directions[current_node] = chosen_neighbor

            if(self.visited[chosen_neighbor]):
                break
            else:
                current_node = chosen_neighbor
                neighbor_list = self._generate_neighbors(current_node)
        self.create_path_from_directions(start_node,current_node)

    def create_path_from_directions(self,start_node,end_node):
        """makes a path following the directions from start node till the end node"""
        current_node = start_node
        while(True):
            if(not (current_node == end_node)):
                next_node = self.directions[current_node]
                self._connect_adjacent_nodes(current_node, next_node)
                current_node = next_node
            else:
                next_node = self.directions[current_node]
                self._connect_adjacent_nodes(current_node, next_node)
                break
        

    def _connect_adjacent_nodes(self,node1,node2):
        """connect two adjacent nodes with a road"""
        if(node1[0] > node2[0] and node1[1] == node2[1]):
            #node1 on the right
            x = node1[0] - 1
            y = node1[1]

        elif(node1[0] < node2[0] and node1[1] == node2[1]):
            #node1 on the left
            x = node1[0] + 1
            y = node1[1]
        elif(node1[1] > node2[1] and node1[0] == node2[0]):
            #node1 above
            x = node1[0]
            y = node1[1] - 1 
        elif(node1[1] < node2[1] and node1[0] == node2[0]):
            #node1 below
            x = node1[0]
            y = node1[1] + 1
        else:
            return
        self._visit(node1)
        self._visit(node2)
        self._set_cell_road((x,y))    


def floodfill(maze):
    non_zero_pos_list = (maze.T == 1).nonzero()
    default_dict_maze = defaultdict(bool)
    visted_pos = defaultdict(bool)
    for pos in non_zero_pos_list:
        default_dict_maze[(pos[0].item(),pos[1].item())] = True
    
    direction_list = [torch.tensor([1.0,0.0]).type(torch.long),torch.tensor([-1.0,0.0]).type(torch.long),torch.tensor([0.0,1.0]).type(torch.long),torch.tensor([0.0,-1.0]).type(torch.long)]
    start_pos = non_zero_pos_list[np.random.choice(np.arange(len(non_zero_pos_list)))]
    frontier = [start_pos]
    

    maze_overlay = torch.zeros([*maze.size(),1])
    distance = 0
    plt.ion()
    while(len(frontier) > 0):
        for pos in frontier:
            maze_overlay[pos[1],pos[0]] += distance 
            visted_pos[(pos[0].item(),pos[1].item())] = True

        frontier = [(pos + direction) for pos in frontier for direction in direction_list if default_dict_maze[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())] and not visted_pos[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())]]
        distance +=1 
        plt.cla()
        plt.imshow(maze_overlay)
        plt.pause(0.01)
    
    plt.savefig("analysis.png",bbox_inches='tight')


class bfs_solver():
    def __init__(self):
        pass 
    
    def initilize(self,maze):
        self.non_zero_pos_list = (maze.T == 1).nonzero()
        self.roads = defaultdict(bool)
        self.visted_pos = defaultdict(bool)
        for pos in self.non_zero_pos_list:
            self.roads[(pos[0].item(),pos[1].item())] = True
        
        self.direction_list = [torch.tensor([1.0,0.0]).type(torch.long),torch.tensor([-1.0,0.0]).type(torch.long),torch.tensor([0.0,1.0]).type(torch.long),torch.tensor([0.0,-1.0]).type(torch.long)]
        self.start_pos = self.non_zero_pos_list[np.random.choice(np.arange(len(self.non_zero_pos_list)))]
        self.end_pos = self.non_zero_pos_list[np.random.choice(np.arange(len(self.non_zero_pos_list)))]
        
        self.maze_distances = torch.zeros([*maze.size(),1])



    def search(self):
        """does breadth first search for the end position and builds a distance map """
        distance = 0
        frontier = [self.start_pos]
        progress = 0
        
        with tqdm(total=len(self.non_zero_pos_list),ascii=True,desc="search") as pbar:
            while(True):
                for pos in frontier:
                    self.maze_distances[pos[1],pos[0]] += distance 
                    self.visted_pos[(pos[0].item(),pos[1].item())] = True
                    if(pos[0] == self.end_pos[0] and pos[1] == self.end_pos[1]):
                        pbar.update(len(self.non_zero_pos_list)-progress)
                        return 
                
                pbar.update(len(frontier))
                progress += len(frontier)
                frontier = [(pos + direction) for pos in frontier for direction in self.direction_list if self.roads[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())] and not self.visted_pos[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())]]
                distance +=1 

    def build_target_map(self,maze):
        """given the distance map from the search it calculates the shortest path between the end and the start and draws this on the matrix it returns"""
        target_map = torch.zeros(maze.size())
        pos = self.end_pos
        while(True):
            target_map[pos[1]][pos[0]] = 1
            if(pos[0] == self.start_pos[0] and pos[1] == self.start_pos[1]):
                break
            possible_next_pos = [(pos + direction) for direction in self.direction_list if self.roads[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())] and self.visted_pos[(pos[0].item() + direction[0].item(),pos[1].item()+direction[1].item())]]
            best_next_pos = None 
            value_best_next_pos = -1 
            for next_pos in possible_next_pos:
                if(value_best_next_pos == -1 or value_best_next_pos > self.maze_distances[next_pos[1],next_pos[0]]):
                    best_next_pos = next_pos
                    value_best_next_pos = self.maze_distances[next_pos[1],next_pos[0]]
            pos = best_next_pos
        return target_map

    def solve(self,maze,plot=False):
        """solves the maze returns matrix with start position and end position, the road to walk and the maze itself""" 
        self.initilize(maze)
        self.search()
        target = self.build_target_map(maze)
        start_end_matrix = torch.zeros(maze.size())
        start_end_matrix[self.start_pos[1],self.start_pos[0]] = 1
        start_end_matrix[self.end_pos[1],self.end_pos[0]] = 1
               
        if(plot):
            fig,axis = plt.subplots(3)
            axis[0].imshow(maze)
            axis[1].imshow(target)
            axis[1].scatter(*self.end_pos)
            axis[1].scatter(*self.start_pos)
            axis[0].scatter(*self.end_pos)
            axis[0].scatter(*self.start_pos)
            axis[2].imshow(start_end_matrix)
            plt.show()
        
        return (maze,target,start_end_matrix)

class Dataset_manager():
    def __init__(self,path="maze_dataset.csv",name_series="maze"):
        self.path = path 
        self.name = name_series

    def load_series(self):
        """returns the series of the data"""
        return pd.read_csv(self.path)[self.name]
    
    def load_iterable(self):
        """returns iterable that iterates over the entire series"""
        for maze in self.load_series():
            yield pickle.loads(eval(maze))

    def save(self,element):
        if(exists(self.path)):
            #file exists 
            series = pd.Series(pickle.dumps(element)) 
            series.to_csv(self.path,index=False,header=False,mode='a')
        else:
            #doesnt exist
            series = pd.Series(pickle.dumps(element),name=self.name) 
            series.to_csv(self.path,index=False)


if __name__ == "__main__":
    solver = bfs_solver()
    for num_cells in [5,7,30,101,401]: #9x9, 13x13, 59x59, 201x201, 801x801
        generator = maze_generator(num_cells,num_cells)
        print(generator.size())

        manager = Dataset_manager("test_maze_"+ str(generator.size()[0]) +"by" + str(generator.size()[1]) + ".csv")

        for i in range(1000): #1000
            maze = generator.generate(maze_num=i)
            for _ in range(100): #100
                manager.save(solver.solve(maze))
            