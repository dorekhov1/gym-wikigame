import configparser


class Configs:

    def __init__(self, section_name="DEFAULT", config_path="config.ini"):
        parser = configparser.RawConfigParser()
        parser.read(config_path)

        section = parser[section_name]

        self.env_name = section.get('env_name')
        self.graph_path = section.get('graph_path')
        self.max_episodes = section.getint('max_episodes')
        self.max_timesteps = section.getint('max_timesteps')

        if section_name == "PPO":
            self.render = section.getboolean('render')
            self.solved_reward = section.getint('solved_reward')
            self.log_interval = section.getint('log_interval')
            self.update_timestep = section.getint('update_timestep')
            self.action_std = section.getfloat('action_std')
            self.k_epochs = section.getint('k_epochs')
            self.eps_clip = section.getfloat('eps_clip')
            self.gamma = section.getfloat('gamma')
            self.lr = section.getfloat('lr')
            self.betas = (section.getfloat('beta1'), section.getfloat('beta2'))
            self.random_seed = section.getfloat('random_seed')
