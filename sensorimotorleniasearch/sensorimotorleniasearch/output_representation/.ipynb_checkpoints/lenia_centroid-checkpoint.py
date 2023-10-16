from sensorimotorleniasearch import OutputRepresentation
import torch


class LeniaCentroidRepresentation(OutputRepresentation):

    @staticmethod
    def default_config():
        default_config = OutputRepresentation.default_config()
        default_config.env_size = (256, 256)
        default_config.distance_function = "L2"
        return default_config

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.n_latents = 3


    def calc(self, observations):
        """
            Maps the observations of a system to an embedding vector
            Return a torch tensor
        """
        
        # filter low values
        filtered_im =observations.states[-1,:,:,0]
        
        # recenter
        mu_0 = filtered_im.sum()
        
        # implementation of meshgrid in torch
        x = torch.arange(self.config.env_size[0])
        y = torch.arange(self.config.env_size[1])
        yy = y.repeat(self.config.env_size[0], 1)
        xx = x.view(-1, 1).repeat(1, self.config.env_size[1])
        
        X = (xx - int(self.config.env_size[0] / 2)).double()
        Y = (yy - int(self.config.env_size[1] / 2)).double()

        centroid_x = ((X * filtered_im).sum() / (mu_0+1e-10))
        centroid_y = ((Y * filtered_im).sum() / (mu_0+1e-10))
        X = (xx -centroid_x-self.config.env_size[0] / 2).double()
        Y = (yy - centroid_y-self.config.env_size[1] / 2).double()
        
        # distance to center in normalized space
        D = torch.sqrt(X ** 2 + Y ** 2)/ (35)
        
        mask=0.85*(D<0.5).float()+0.15*(D<1).float()
        loss=(filtered_im-0.9*mask).pow(2).sum().sqrt() 
        
        embedding = torch.zeros(3)
        embedding[0]=loss/230
        embedding[1]=centroid_x.mean()/self.config.env_size[0]
        embedding[2]=centroid_y.mean()/self.config.env_size[1] 
        if(mu_0<1e-4):
          embedding[1]=embedding[1]-10
          embedding[2]=embedding[2]-10


        # print(embedding)
        
        

        return embedding





    def calc_distance(self, embedding_a, embedding_b):
        """
            Compute the distance between 2 embeddings in the latent space
            /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

        else:
            raise NotImplementedError

        return dist
