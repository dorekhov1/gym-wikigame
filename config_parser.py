import configparser


class Configs:

    def __init__(self, config_path="config.ini"):
        self.parser = configparser.RawConfigParser()
        self.parser.read(config_path)

        section = self.parser["DEFAULT"]

        self.env_name = section.get('env_name')
        self.graph_path = section.get('graph_path')
        self.max_episodes = section.getint('max_episodes')
        self.max_timesteps = section.getint('max_timesteps')
        self.log_interval = section.getint('log_interval')
        self.save_interval = section.getint('save_interval')
        self.random_goal = section.getboolean('random_goal')
        self.n_steps_away = section.getint('n_steps_away')
        self.random_seed = section.getfloat('random_seed')
        self.reward_fn = section.get('reward_fn')
        self.solved_reward = section.getint('solved_reward')
        self.agent = section.get('agent')

    def get_agent_kwargs(self):
        agent_section = self.parser[self.agent]
        agent_kwargs = None

        if self.agent == "PPO":
            agent_kwargs = {
                "optimize_timestep": agent_section.getint('optimize_timestep'),
                "k_epochs": agent_section.getint('k_epochs'),
                "eps_clip": agent_section.getfloat('eps_clip'),
                "gamma": agent_section.getfloat('gamma'),
                "lr": agent_section.getfloat('lr'),
                "betas": (agent_section.getfloat('beta1'), agent_section.getfloat('beta2')),
                "emb_dim": agent_section.getint('emb_dim'),
            }

        return agent_kwargs
