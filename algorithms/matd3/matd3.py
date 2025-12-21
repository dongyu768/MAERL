import torch
import torch.optim as optim
import sys, os
import numpy as np
from core.archpy.matd3.actor_critic import Actor, Critic
from utils.config_tools import init_writter

class MATD3:
    def __init__(self, config, agent_id):  
        self.config = config
        self.agent_id = agent_id
        self.n_agents = config['n_agents']
        self.train_step = 0
        self.device = config['device']  # 新增：存储设备信息

        # 创建actor和critic网络，并将其移动到指定设备
        self.actor_network = Actor(config, agent_id).to(self.device) 
        self.critic_network = Critic(config).to(self.device)  # 双critic网络

        # 创建目标网络，并将其移动到指定设备
        self.actor_target_network = Actor(config, agent_id).to(self.device) 
        self.critic_target_network = Critic(config).to(self.device)  

        # 初始化目标网络
        self._hard_update(self.actor_target_network, self.actor_network)
        self._hard_update(self.critic_target_network, self.critic_network)
        
        # 创建优化器
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=config['lr_actor'])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=config['lr_critic'])

        self.model_path = self.config['save_dir']
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        
        self.writer = init_writter(self.config)
        
    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.config['tau'] * param.data + (1.0 - self.config['tau']) * target_param.data)    
    
    def train(self, transitions, other_agents):
        for key in transitions.keys():
            transitions[key] = torch.tensor(transitions[key], dtype=torch.float32).to(self.device)  # 修改：将数据移动到指定设备
        r = transitions['r_%d' % self.agent_id] # 训练时只需要自己的reward

        o, u, o_next = [], [], [] # 用来装每个agent经验中的各项
        for agent_id in range(self.n_agents):
            o.append(transitions['o_%d' % agent_id].to(self.device))  
            u.append(transitions['u_%d' % agent_id].to(self.device))  
            o_next.append(transitions['o_next_%d' % agent_id].to(self.device)) 
        
        # calculate the target Q value function
        u_next = []
        ############## Update Critic ##############
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.n_agents):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next1, q_next2 = self.critic_target_network(o_next, u_next)
            q_next = torch.min(q_next1, q_next2).detach()
            target_q = (r.unsqueeze(1) + self.config['gamma'] * q_next).detach()  # 目标值计算
        
        # the q loss
        q_value1, q_value2 = self.critic_network(o, u)
        critic_loss1 = (target_q - q_value1).pow(2).mean()   # 价值网络损失函数
        critic_loss2 = (target_q - q_value2).pow(2).mean()
        critic_loss = critic_loss1 + critic_loss2

        if self.train_step % self.config['log_interval'] == 0 and self.agent_id==0:
            self.writer.add_scalar('critic_loss', critic_loss.item(), self.train_step)
        # 更新双critic网络
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = -self.critic_network.Q1(o, u).mean()  # 策略梯度更新

        if self.train_step % self.config['log_interval'] == 0 and self.agent_id==0:
            # print('critic_loss is {}, actor_loss is {}'.format(critic_loss.item(), actor_loss.item()))
            self.writer.add_scalar('actor_loss_%d'%self.agent_id, actor_loss.item(), self.train_step)
        
        # 更新actor网络
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


        # 软更新目标网络
        self._soft_update(self.actor_target_network, self.actor_network)
        self._soft_update(self.critic_target_network, self.critic_network)

        if self.train_step > 0 and self.train_step % self.config['save_interval'] == 0:
            self.save_model(self.train_step)
        self.train_step += 1
        
        return critic_loss.item(), actor_loss.item()

    def save_model(self, train_step):
        model_path = os.path.join(self.model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + 'actor.pth')
        torch.save(self.critic_network.state_dict(), model_path + '/' + 'critic.pth')