import collections
from typing import Any, Callable, DefaultDict, Dict, Optional, Type

import gym
from torch import nn


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Registry(metaclass=Singleton):
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
        override_assert: bool = False,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None and not override_assert:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name

            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def _get_impl(cls, _type: str, name: str):
        cls_ = cls.mapping[_type].get(name, None)
        if cls_ is None:
            print(
                f"{name} not in {_type} registry! Here's what's available:\n",
                "\n".join(list(cls.mapping[_type].keys())),
            )
        return cls_

    @classmethod
    def register_runner(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("runner", to_register, name)

    @classmethod
    def get_runner(cls, name: str):
        return cls._get_impl("runner", name)

    @classmethod
    def register_envs(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("envs", to_register, name, assert_type=gym.Env)

    @classmethod
    def get_envs(cls, name: str) -> gym.Env:
        return cls._get_impl("envs", name)

    @classmethod
    def register_actor_critic(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("actor_critic", to_register, name)

    @classmethod
    def get_actor_critic(cls, name: str):  # -> ActorCritic:
        return cls._get_impl("actor_critic", name)

    @classmethod
    def register_nn_base(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("nn_base", to_register, name)

    @classmethod
    def get_nn_base(cls, name: str):  # -> NNBase:
        return cls._get_impl("nn_base", name)

    @classmethod
    def register_act_dist(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("act_dist", to_register, name, assert_type=nn.Module)

    @classmethod
    def get_act_dist(cls, name: str) -> nn.Module:
        return cls._get_impl("act_dist", name)

    @classmethod
    def register_scheduler(cls, to_register=None, *, name: Optional[str] = None):
        return cls._register_impl("scheduler", to_register, name)

    @classmethod
    def get_scheduler(cls, name: str) -> nn.Module:
        return cls._get_impl("scheduler", name)


drl_registry = Registry()
