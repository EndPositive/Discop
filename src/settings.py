import torch


class Settings:

    def __init__(
        self,
        temp: float = 1.2,
        seed: int = 1,
        eos_token_id: int = -1,
        dot_token_id: int = -1,
        dash_token_id: int = -1,
        device=torch.device("cpu"),
    ):

        if temp is None:
            temp = 1.0
        self.temp = temp
        self.seed = seed
        self.eos_token_id = eos_token_id
        self.dot_token_id = dot_token_id
        self.dash_token_id = dash_token_id
        self.device = device

    def __str__(self):
        return "\n".join(
            "{} = {}".format(key, value) for (key, value) in self.__dict__.items()
        )


text_default_settings = Settings()
