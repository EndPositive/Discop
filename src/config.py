import os
import torch


class Settings:

    def __init__(self,
                 model_name: str = 'gpt2',
                 temp: float = 1.0,
                 top_p: float = 0.92,
                 length: int = 100,
                 seed: int = os.urandom(1),
                 device=torch.device('cpu')):

        self.model_name = model_name

        if temp is None:
            temp = 1.0
        self.temp = temp

        if top_p is None:
            top_p = 1.0
        elif top_p <= 0 or top_p > 1:
            raise ValueError('`top_p` must be in (0, 1]!')
        self.top_p = top_p

        self.length = length
        self.seed = seed
        self.device = device

    def __call__(self):
        return self.temp, self.top_p, self.length, self.seed

    def __str__(self):
        return '\n'.join('{} = {}'.format(key, value) for (key, value) in self.__dict__.items())

text_default_settings = Settings(model_name='gpt2', top_p=0.92, length=100)
