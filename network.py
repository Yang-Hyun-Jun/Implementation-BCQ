import torch
import torch.nn as nn

class Perturbation(nn.Module):
    """
    State와 Action을 input으로 하여 
    [-phi, phi] 범위에서 perturb 된 action 리턴
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.act = nn.ReLU()
        self.out = nn.Tanh()
        self.l1 = nn.Linear(s_dim + a_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, a_dim)

        self.phi = 0.05

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        p = self.out(self.l3(x))
        purturb = self.phi * p
        purturb_action = (a + purturb).clamp(-1.0, 1.0)
        return purturb_action


class Qnet(nn.Module):
    """
    State와 Action을 input으로 하여 Q value 리턴 
    Clipped Double Q learning을 위한 Q1, Q2
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim

        self.act = nn.ReLU()
        self.l1 = nn.Linear(s_dim + a_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        self.l4 = nn.Linear(s_dim + a_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

    def q1(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.l3(x)
        return x
    
    def q2(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.act(self.l4(x))
        x = self.act(self.l5(x))
        x = self.l6(x)
        return x


class VAE(nn.Module):
    """
    Batch constrained action 생성을 위한 Generator VAE
    """
    def __init__(self, s_dim, a_dim, z_dim):
        super().__init__()

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.act = nn.ReLU()
        self.out = nn.Tanh()
        
        # Encoder
        self.e1 = nn.Linear(s_dim + a_dim, 750)
        self.e2 = nn.Linear(750, 750)
        self.mean = nn.Linear(750, z_dim)
        self.log_std = nn.Linear(750, z_dim)

        # Decoder
        self.d1 = nn.Linear(s_dim + z_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, a_dim)

    def forward(self, s, a):
        z, mean, std = self.encoder(s, a)
        a = self.decoder(s, z)
        return a, mean, std
    
    def encoder(self, s, a):
        x = torch.cat([s, a], 1)
        x = self.act(self.e1(x))
        x = self.act(self.e2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = self.log_std(x).clamp(-4, 15)
        std = torch.exp(log_std)

        z = mean + std * torch.randn_like(std)
        return z, mean, std
    
    def decoder(self, s, z=None):

        z_shape = (s.shape[0], self.z_dim)
        z = torch.randn(z_shape).clamp(-0.5, 0.5) if z is None else z
        
        x = torch.cat([s, z], 1)
        a = self.act(self.d1(x))
        a = self.act(self.d2(a))
        a = self.out(self.d3(a))
        return a


if __name__ == '__main__':
    s_dim = 11
    a_dim = 3
    z_dim = a_dim * 2

    s = torch.rand((2, 11))
    a = torch.rand((1, 3))

    perturbation = Perturbation(s_dim, a_dim)
    qnet = Qnet(s_dim, a_dim)
    vae = VAE(s_dim, a_dim, z_dim)

    perturbation_action = perturbation(s, a)
    q1 = qnet.q1(s, a)
    q2 = qnet.q2(s, a)
    
    action, mean, std = vae(s, a)
    
    






