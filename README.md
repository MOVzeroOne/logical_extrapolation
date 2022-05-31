# logical_extrapolation
Code based upon End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking (arXiv:2202.05826)
<br> 
Trained on 9x9 with at most 30 "thinking/recurrent" steps. <br>
Below is the inference on 9x9 (inside the training set) and 13x13, 59x59 (both outside of the training set and using more recurrent steps than during training)
<br>
first image of each gif is the maze, the second start and goal, third the ground truth, fourth the output of the network at each "thinking" step.
<br><h2>9x9 (30 steps)</h2><br>
![](9x9.gif)
<br><h2>13x13 (50 steps)</h2><br>
![](13x13.gif)
<br><h2>59x59 (100 steps)</h2><br>
![](59x59.gif)
<br><h2>201x201 (1000 steps)</h2><br>
![](201x201.gif)
<br><h2>801x801 (7000 steps, the visualization has 100 steps per frame update)</h2><br>
![](801x801.gif)
