import torch
import numpy as np
import os

from network import Perturbation
from network import Qnet
from network import VAE

class BCQ:
    """
    BCQ Agent 클래스
    """
    def __init__(self, s_dim, a_dim, gamma, tau, lam):

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = a_dim * 2

        self.gamma = gamma
        self.tau = tau 
        self.lam = lam

        self.perturb = Perturbation(s_dim, a_dim)
        self.perturb_target = Perturbation(s_dim, a_dim)
        self.perturb_target.load_state_dict(self.perturb.state_dict())

        self.qnet = Qnet(s_dim, a_dim)
        self.qnet_target = Qnet(s_dim, a_dim)
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        
        self.vae = VAE(s_dim, a_dim, self.z_dim)

        self.perturb_optim = torch.optim.Adam(self.perturb.parameters(), lr=1e-3)
        self.qnet_optim = torch.optim.Adam(self.qnet.parameters(), lr=1e-4)
        self.vae_optim = torch.optim.Adam(self.vae.parameters())

        self.mse = torch.nn.MSELoss()
        self.path = os.getcwd() + '/Metrics'

    def get_action(self, s):
        with torch.no_grad():
            s = s.repeat(100, 1)
            actions = self.vae.decoder(s)
            pactions = self.perturb(s, actions)
            q1 = self.qnet.q1(s, pactions)
            
            ind = torch.argmax(q1, 0)
            action = pactions[ind].squeeze()
            action = action.numpy()
            return action
        
    def train(self, buffer, steps, batch_size):
        """
        BCQ 업데이트 함수
        """

        for step in range(steps + 1):
            sample = buffer.sample(batch_size)
            sample = self.make_batch(sample)

            s, a, r, ns, done = sample

            # VAE 업데이트
            recon, mean, std = self.vae(s, a)
            recon_loss = self.mse(recon, a)
            KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_optim.zero_grad()
            vae_loss.backward()
            self.vae_optim.step()

            # Soft Clipped Double Q 업데이트
            with torch.no_grad():
                
                ns = torch.repeat_interleave(ns, 10, 0)
                next_a = self.vae.decoder(ns)
                next_pa = self.perturb_target(ns, next_a)
                next_q1 = self.qnet_target.q1(ns, next_pa)
                next_q2 = self.qnet_target.q2(ns, next_pa)

                target = self.lam * torch.min(next_q1, next_q2) \
                    + (1 - self.lam) * torch.max(next_q1, next_q2)
                
                # Max에서 (Value, Indices) 리턴
                target = target.reshape(batch_size, -1).max(1)[0]
                target = target.reshape(batch_size, 1)
                target = r + self.gamma * target * (1-done)

            q1 = self.qnet.q1(s, a)
            q2 = self.qnet.q2(s, a)

            q_loss = self.mse(q1, target) + self.mse(q2, target)
            self.qnet_optim.zero_grad()
            q_loss.backward()
            self.qnet_optim.step()

            # Perturbation net 업데이트
            actions = self.vae.decoder(s)
            pactions = self.perturb(s, actions)
            
            perturb_loss = -self.qnet.q1(s, pactions).mean()
            self.perturb_optim.zero_grad()
            perturb_loss.backward()
            self.perturb_optim.step()

            self.soft_target_update()

            if step % 300 == 0 or step == steps:
                print(f"{step} done")
                print(f"q loss:{q_loss.detach()}")
                print(f"recon loss:{recon_loss.detach()}")
                print(f"KL loss: {KL_loss.detach()}")
                print(f"VAE loss: {vae_loss.detach()}")
                print(f"perturb loss: {perturb_loss.detach()} \n")
                self.save_model()


    def make_batch(self, sample):
        s, a, r, ns, d = list(zip(*sample))
        
        s = torch.cat(s)
        a = torch.cat(a)
        r = torch.cat(r).reshape(-1, 1)
        ns = torch.cat(ns)
        d = list(map(float, d))
        d = torch.tensor(d).reshape(-1, 1)
        return s, a, r, ns, d
    

    def save_model(self):
        torch.save(self.perturb.state_dict(), self.path + '/perturb.pth')
        torch.save(self.qnet.state_dict(), self.path + '/qnet.pth')
        torch.save(self.vae.state_dict(), self.path + '/vae.pth')


    def soft_target_update(self):
        """
        network와 target network를 soft copy update
        """
        for param, target_param in zip(self.qnet.parameters(), self.qnet_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.perturb.parameters(), self.perturb_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)            



