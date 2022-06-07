from drl.algo.erpg import ERPG, RPG
from drl.runners.eppo_trainer import EPPOTrainer
from drl.utils.registry import drl_registry


@drl_registry.register_runner
class ERPGTrainer(EPPOTrainer):
    algo_cls = ERPG

@drl_registry.register_runner
class RPGTrainer(EPPOTrainer):
    algo_cls = RPG