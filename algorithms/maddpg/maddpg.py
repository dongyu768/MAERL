import torch
import os
from core.archpy.maddpg.actor_critic import Actor, Critic
from utils.config_tools import init_writter

class MADDPG:
    def __init__(self, args, agent_id): # 因为不同的agent的obs,act维度可能不一样，所以神经网络不同，需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.device = torch.device(self.args['device'])
        self.train_step = 0
        # create the network
        self.actor_network = Actor(args, agent_id).to(self.device)
        self.critic_network = Critic(args).to(self.device)
        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(self.device)
        self.critic_target_network = Critic(args).to(self.device)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args['lr_actor'])
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args['lr_critic'])

        # path to save the model
        # self.model_path = self.args['save_dir']
        # # self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        # if not os.path.exists(self.model_path):
        #     os.mkdir(self.model_path)
        if not self.args['evaluate']:
            self.writer = init_writter(args)

    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1-self.args['tau']) * target_param.data + self.args['tau'] * param.data)
        for target_param, param in zip(self.critic_target_network.parameters(), self.critic_network.parameters()):
            target_param.data.copy_((1-self.args['tau']) * target_param.data + self.args['tau'] * param.data)

    def train(self, transitions, other_agents):
        # self.writer = init_writter(self.args)
        for key in transitions.keys():
            transitions[key] = torch.from_numpy(transitions[key]).float().to(self.device)
        r = transitions['r_%d' % self.agent_id] # 训练时只需要自己的reward
        o, u, o_next = [], [], [] # 用来装每个agent经验中的各项
        for agent_id in range(self.args['n_agents']):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])
        # calculate the target Q value function
        u_next = []
        with torch.no_grad():
            # 得到下一个状态对应的动作
            index = 0
            for agent_id in range(self.args['n_agents']):
                if agent_id == self.agent_id:
                    u_next.append(self.actor_target_network(o_next[agent_id]))
                else:
                    # 因为传入的other_agents要比总数少一个，可能中间某个agent是当前agent，不能遍历去选择动作
                    u_next.append(other_agents[index].policy.actor_target_network(o_next[agent_id]))
                    index += 1
            q_next = self.critic_target_network(o_next, u_next).detach()
            target_q = (r.unsqueeze(1) + self.args['gamma'] * q_next).detach()  # 目标值计算
        # the q loss
        q_value = self.critic_network(o, u)
        critic_loss = (target_q - q_value).pow(2).mean()                     # 价值网络损失函数
        # the actor loss
        # 重新选择联合动作中当前agent的动作，其他agent的动作不变
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        actor_loss = -self.critic_network(o, u).mean() # 策略梯度更新
        if self.train_step % self.args['log_interval'] == 0 and self.agent_id==0:
            # print('critic_loss is {}, actor_loss is {}'.format(critic_loss.item(), actor_loss.item()))
            self.writer.add_scalar('actor_loss_%d'%self.agent_id, actor_loss.item(), self.train_step)
            self.writer.add_scalar('critic_loss', critic_loss.item(), self.train_step)
        # update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self._soft_update_target_network()
        # if self.train_step > 0 and self.train_step % self.args['save_interval'] == 0:
        #     self.save_model()
        self.train_step += 1
    def save_model(self):
        model_path = os.path.join(self.model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + 'actor.pth')
        torch.save(self.critic_network.state_dict(), model_path + '/' + 'critic.pth')
