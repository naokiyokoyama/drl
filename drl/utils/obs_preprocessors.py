from drl.utils.registry import drl_registry


@drl_registry.register_obs_preprocessor
def identity(obs):
    return obs

@drl_registry.register_obs_preprocessor
def isaac_get_obs(obs):
    return obs["obs"]
