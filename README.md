# Add a new code (Flocking-avoiding.py) for multi-UAV passing obstacles, and I see the your code seems not contain the function to draw animation, so I add it to code Flocking.py and Flocking-avoiding.py.
## Modifying time:
1. 25,June. Add axis to the figure
2. 26,June. Add animation to the figure, the function name is plot_time_series which lie in the struct Flock both Flocking.py and Flocking-avoiding.py.
3. 1,July. Add code Flocking-avoiding.py to the respositories
4. 3,July. Modify the axis in order to let the obstacle look like circle not ellipse. But the result is not show in the next section. The result of next section is also ellipse for obstacle.

## Result-shwoing
### The results for algorithm 1,2 and 3
This section only shows a few results because Github has the limitation of file size.

- Algorithm 1: 40 arfa-agent without gamma-agent and obstacles.

![](https://github.com/arthur-yh/Consensus/blob/yh/results/flock-algorithm1-40.gif)

- Algorithm 1: 150 arfa-agent without gamma-agent and obstacles.

![](https://github.com/arthur-yh/Consensus/blob/yh/results/flock-algorithm1-150.gif)

+ Algorithm 2: 10 arfa-agent with gamma-agent but no obstacles.

![](https://github.com/arthur-yh/Consensus/blob/yh/results/flock-algorithm2-10.gif)

- Algorithm 3: 50 arfa-agent with gamma-agent and obstacles. 

The parameters is based on the paper in reference but it doesn't shows the specific for beda-agent-Cq, arfa-agent-Cq and gamma-agent-Cq. It just shows beda-agent-Cq > gamma-agent-Cq > arfa-agent-Cq. So I set them to this: 
```
agent beda-agent-Cq = 16，arfa-agent-Cq = 2.5, gamma-agent-Cq = 4.5
```
![image](https://github.com/arthur-yh/Consensus/blob/yh/results/TIM%E6%88%AA%E5%9B%BE20190701194014.png)

![image](https://github.com/arthur-yh/Consensus/blob/yh/results/avoid44.24.png)

![image](https://github.com/arthur-yh/Consensus/blob/yh/results/aviod51.54.png)


## Reference 
[1] Flocking for Multi-Agent Dynamic Systems:Algorithms and Theory,Reza Olfati-Saber,June 22, 2004

