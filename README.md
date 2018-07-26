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

## 项目完成情况
项目的细节见Quadcopter_Project-Copy4.ipynb
1. 给四轴飞行器设计的是起飞任务，该任务的定义在Takeoff_task.py中，让飞机从平地起飞到高度为10的位置。
根据任务，奖励函数设计如下：
```
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
```
2. 智能体agent的设计在文件夹agents/policy_gradients_ddpg.py中，采用DDPG算法，算法详情可参考论文[DDPG](https://arxiv.org/abs/1509.02971)
根据项目提示，选择了DDPG的深度确定性策略梯度，构建了行动者-评论者方法，设计了回放缓冲区及Ornstein–Uhlenbeck 噪点。
超参数：
```
        Actor_lr=0.0003#actor的学习率
        Critic_lr=0.001#Critic的学习率
        self.gamma = 0.99  # discount factor
        self.tau = 0.0001  # for soft update of target parameters
        
        # Noise process 噪声参数
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        
        # Replay memory循环存储区参数
        self.buffer_size = 100000
        self.batch_size = 64
```        
神经网络：
        行动者网络结构：1个输入层，3个隐藏层，1个输出层，隐藏层激活函数采用relu，输出层激活函数采用sigmoid
```
         # Define input layer (states)   输入层，并做normalization处理
        states = layers.Input(shape=(self.state_size,), name='states')
        layers.normalization.BatchNormalization()                

        # Add hidden layers  隐含层分别采用400、300、200单元数，使用relu激活，并用dropout防止过拟合
        net = layers.Dense(units=400)(states)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        net = layers.Dense(units=300)(net)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        net = layers.Dense(units=200)(net)
        #net=layers.BatchNormalization()(net)
        net=layers.Activation('relu')(net)
        net=layers.Dropout(0.2)(net)
        
        # Add final output layer with sigmoid activation  输出层使用权重初始化，给予较小初始权重，使用sigmoid激活函数
        raw_actions = layers.Dense(units=self.action_size,
                kernel_initializer=initializers.RandomUniform(minval=-0.0003, maxval=0.0003),
                                   activity_regularizer=regularizers.l2(0.0001),name='raw_actions')(net)
        raw_actions=layers.Activation('sigmoid')(raw_actions)        
 ```
 
   评论者网络结构：1个输入层，2个隐藏层，1个输出层，隐藏层激活函数采用relu
 ```
        # Define input layers  输入层，并对state输入做normalization处理
        states = layers.Input(shape=(self.state_size,), name='states')
        layers.normalization.BatchNormalization()
        actions = layers.Input(shape=(self.action_size,), name='actions')
        #layers.normalization.BatchNormalization()


        # Add hidden layer(s) for state pathway  状态向量的隐含层，2层，单元数分别是400、300，relu激活，并加权重正则化和dropout
        net_states = layers.Dense(units=400)(states)       
        #net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.2)(net_states)
        net_states = layers.Dense(units=300)(net_states)
        #net_states=layers.BatchNormalization()(net_states)
        net_states=layers.Activation('relu')(net_states)
        net_states=layers.Dropout(0.2)(net_states)

        # Add hidden layer(s) for action pathway  动作向量的隐含层，2层，单元数分别是400、300，relu激活，并加权重正则化和dropout
        net_actions = layers.Dense(units=400)(actions)
        #net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)
        net_actions=layers.Dropout(0.2)(net_actions)
        net_actions = layers.Dense(units=300)(actions)
        #net_actions=layers.BatchNormalization()(net_actions)
        net_actions=layers.Activation('relu')(net_actions)        
        net_actions=layers.Dropout(0.2)(net_actions)

        # Combine state and action pathways  将state和action子网络合并
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        # Add final output layer to prduce action values (Q values) 输出层
        Q_values = layers.Dense(units=1,activity_regularizer=regularizers.l2(0.0001),name='q_values')(net)
 ```  
3. 项目的主要任务
该任务对我来说有不小难度，主要是网络结构的设计、奖励函数的设置，及超参数的调试，花了很多时间，踩了许多坑。
进行了1000个阶段的训练，观察奖励曲线，可以发现奖励在训练过程中存在急速上升和下降的过程，这应该就是智能体在不断的探索尝试，
最终reward趋于平稳。训练1000次后，用该智能体控制飞行器完成起飞任务，将飞行器相关数据存在DDPG_data.txt中，画出位置坐标变化曲线，
可以看出x,y坐标基本维持在0附近，z坐标由0快速平滑上升到10，用时约1.2秒。圆满完成任务，性能很好。
