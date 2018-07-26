# DeepRL 四轴飞行器控制器

_指导四轴飞行器学会飞行！_

在本次项目中，你将设计一个深度强化学习智能体，来控制几个四轴飞行器的飞行任务，包括起飞、盘旋和着陆。

## 项目说明

1. 复制代码库，并浏览下载文件夹。

```
git clone https://github.com/udacity/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. 创建并激活一个新的环境。

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. 为 `quadcop` 环境创建一个 [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html)。 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. 打开 notebook。
```
jupyter notebook Quadcopter_Project.ipynb
```

5. 在运行代码之前，请使用 drop-down 菜单（**Kernel > Change kernel > quadcop**） 修改 kernel 以适应 `quadcop` 环境。接着请按照 notebook 中的说明进行操作。

6. 为了完成本项目，你也许还需要安装额外的 pip 包。请查看代码库中的 `requirements.txt` 文件，以了解运行项目所需的包。

##项目完成情况

1.给四轴飞行器设计的是起飞任务，该任务的定义在Takeoff_task.py中，让飞机从平地起飞到高度为10的位置。
根据任务，奖励函数设计如下：

def get_reward(self):
    
        '''根据飞行器的位置及速度信息综合设计reward。首先根据位置坐标及速度及时间来决定reward的值，每个循环周期给-1的惩罚，
        给与x、y坐标绝对值大小的惩罚，z轴方向速度相关的奖励，正向速度越大奖励越大，最后对reward进行尺度缩放'''
                          
        reward = -1-1.*abs(self.sim.pose[:2] - self.target_pos[:2]).sum()
        #reward += 1.*(self.sim.pose[2])
        if self.sim.v[2]>0:
            reward += 5*self.sim.v[2]
        elif self.sim.v[2]<0:
            reward -=abs(5*self.sim.v[2])                       
        reward=0.3*reward#尺度缩放
        return reward

def step(self, rotor_speeds):
    """Uses action to obtain next state, reward, done."""
    reward = 0
    pose_all = []
    for _ in range(self.action_repeat):
        done = self.sim.next_timestep(rotor_speeds)
        
        '''根据done即当前阶段是否结束，来分别给与不同的reward设置。done=True，若飞行器达到目标位置，则奖励0.3*20，
        否则惩罚0.3*50；done=False,若飞行器达到目标位置，则奖励0.3*10，且done置为True，否则reward按get_reward计算。'''
        
        if done:                
            if self.sim.pose[2] >= self.target_pos[2]:
                reward+=0.3*20                    
            else:
                reward-=0.3*50
        else:
            if self.sim.pose[2] >= self.target_pos[2]:
                reward+=0.3*20
                done=True
            else:
                reward += self.get_reward() 
        pose_all.append(self.sim.pose)
    next_state = np.concatenate(pose_all)
    return next_state, reward, done

    '''首先，根据飞行器位置与目标位置的绝对距离差来计算奖励；然后，判断若垂直距离大于目标垂直距离，则增加奖励；
    再判断若运行时间大于最大执行时间，则惩罚减少奖励。
    这样设计奖励函数可使飞行器又稳又快的起飞到目标位置。'''
