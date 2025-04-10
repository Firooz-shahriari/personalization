import yaml


class Config:
    def __init__(self, path="config/config.yml"):
        self.config = Config.load_config(path)

        self.reproducibility = self.config["reproducibility"]
        self.seed = self.config["seed"]

        # Load graph configurations
        self.graph = GraphConfig(self.config["graph"])

        # Load problem configurations
        self.pr = ProblemConfig(self.config["problem"])

        # Load optimizer configurations
        # self.p_dagp = p_dagp_OptimizerConfig(self.config["optimizer"]["p_dagp"])
        self.pcgd = pcgd_OptimizerConfig(self.config["optimizer"]["pcgd"])

    @staticmethod
    def load_config(path):
        with open(path) as f:
            config = yaml.load(f, yaml.FullLoader)

        return config


class GraphConfig:
    def __init__(self, config):
        self.num_nodes = config["num_nodes"]
        self.edge_prob = config["edge_prob"]
        self.directed = config["directed"]
        self.laplacian_factor = config["laplacian_factor"]


class ProblemConfig:
    def __init__(self, config):
        self.local_dimension = config["synthetic"]["local_dimension"]
        self.global_dim = config["synthetic"]["global_dimension"]
        self.epsilon = config["synthetic"]["epsilon"]
        self.global_objective_exists = config["synthetic"]["global_objective_exists"]
        # self.limted_labels = config["logistic_regression"]["limited_labels"]
        # self.balanced = config["logistic_regression"]["balanced"]
        # self.train_size = config["logistic_regression"]["train_size"]
        # self.regularization = config["logistic_regression"]["regularization"]
        # self.reg = config["logistic_regression"]["regularization_strength"]


class p_dagp_OptimizerConfig:
    def __init__(self, config):
        self.max_iter = config["max_iter"]
        self.alpha = config["alpha"]
        self.rho = config["rho"]
        self.lr = config["learning_rate"]["initial"]
        self.min = config["learning_rate"]["min"]
        self.decay_rate = config["learning_rate"]["decay_rate"]
        self.decay_steps = config["learning_rate"]["decay_steps"]


class pcgd_OptimizerConfig:
    def __init__(self, config):
        self.max_iter = config["max_iter"]
        self.lr = config["learning_rate"]["initial"]
        self.min = config["learning_rate"]["min"]
        self.decay_rate = config["learning_rate"]["decay_rate"]
        self.decay_steps = config["learning_rate"]["decay_steps"]
        self.projection_iters = config["projection_iters"]
