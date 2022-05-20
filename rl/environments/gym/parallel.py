"""Parallel environments"""

import gym
import numpy as np
import multiprocessing as mp

from typing import Union, List, Callable


class CloudPickleWrapper(object):
    """Uses `cloudpickle` to serialize contents (otherwise multiprocessing tries to use pickle)
        - Source: OpenAI baselines
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class AbstractParallelEnv(gym.Env):
    """Abstract parallel-env interface"""
    def __init__(self, num: int, processes: int):
        assert num >= processes

        self.num_environments = num
        self.num_workers = processes

    def get_evaluation_env(self) -> gym.Env:
        """Instantiates a single environment, only for agent evaluation"""
        raise NotImplementedError

    @staticmethod
    def _flatten_obs(x: Union[list, tuple]) -> Union[dict, np.ndarray]:
        assert isinstance(x, (list, tuple))
        assert len(x) > 0

        if isinstance(x[0], dict):
            keys = x[0].keys()
            return {k: np.stack([obs[k] for obs in x]) for k in keys}

        return np.stack(x)

    @staticmethod
    def _flatten_list(x: Union[list, tuple]) -> list:
        assert isinstance(x, (list, tuple))
        assert len(x) > 0
        assert all([len(inner_list) > 0 for inner_list in x])

        return [item for inner_list in x for item in inner_list]


class SequentialEnv(AbstractParallelEnv):
    """An environment that wraps multiple environments in sequence, simulating env parallelism"""

    def __init__(self, env: Union[str, Callable], num=2, seed=None, **kwargs):
        super().__init__(num, processes=1)

        if callable(env):
            self.make_env = lambda: env(**kwargs)

        elif isinstance(env, str):
            self.make_env = lambda: gym.make(id=env, **kwargs)
        else:
            raise ValueError(f'Argument `env` must be "str" or "callable" not "{type(env)}".')

        # instantiate environments
        self.envs = [self.make_env() for _ in range(self.num_environments)]
        self.seed(seed)

        self.observation_space: gym.Space = self.envs[0].observation_space
        self.action_space: gym.Space = self.envs[0].action_space

    def step(self, action: list) -> tuple:
        assert len(action) == self.num_environments
        actions = action
        experiences = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done, info = env.step(action)

            if done:
                info['__terminal_state'] = state
                state = env.reset()

            experiences.append((state, reward, done, info))

        states, rewards, terminals, info = zip(*experiences)
        info = {i: info_ for i, info_ in enumerate(info)}

        return self._flatten_obs(states), np.stack(rewards), np.stack(terminals), info

    def reset(self, **kwargs) -> np.ndarray:
        states = [env.reset(**kwargs) for env in self.envs]
        return self._flatten_obs(states)

    def render(self, mode='human', **kwargs):
        raise NotImplementedError

    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed=None):
        if seed is None:
            return

        if isinstance(seed, (list, tuple)):
            assert len(seed) == self.num_environments

            for env, seed_ in zip(self.envs, seed):
                env.seed(seed=seed_)
        else:
            assert isinstance(seed, (int, float))

            for i, env in enumerate(self.envs):
                env.seed(seed=int(seed + i))

    def get_evaluation_env(self) -> gym.Env:
        return self.make_env()


def work(make_env: CloudPickleWrapper, rank: int, num_envs: int, seed: int, remote, parent_remote):
    """Process's work function. Based on:
        - https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py#L7-L36
    """
    def step(environment, action, index: int):
        state, reward, done, info = environment.step(action)

        if done:
            info['__terminal_state'] = state
            state = environment.reset()

        return state, reward, done, (rank + index, info)

    parent_remote.close()
    envs = [make_env.x() for _ in range(num_envs)]

    for i, env in enumerate(envs):
        env.seed(seed=int(seed + rank + i))

    try:
        while True:
            cmd, data = remote.recv()

            # TODO: set seed here? (would not require `seed` argument on env creation)
            if cmd == 'reset':
                remote.send([env.reset(**data) for env in envs])

            elif cmd == 'step':
                remote.send([step(env, action, index=i) for i, (env, action) in enumerate(zip(envs, data))])

            elif cmd == 'get_spaces_spec':
                remote.send(CloudPickleWrapper((envs[0].observation_space, envs[0].action_space)))
            else:
                # signal to close both environments and communication
                break
    finally:
        # close command
        for env in envs:
            env.close()

        remote.close()


# TODO: too high RAM usage
class MultiProcessEnv(AbstractParallelEnv):
    """Vector environment that uses multiprocessing for parallelism. Based on:
        - https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
    """
    def __init__(self, env: Union[str, Callable], num: int, seed: int, processes: int = None, **kwargs):
        context = mp.get_context('spawn')

        def make_env() -> gym.Env:
            if isinstance(env, str):
                return gym.make(id=env, **kwargs)

            assert callable(env)
            return env(**kwargs)

        env_fn = CloudPickleWrapper(make_env)
        num = int(num)
        self.make_env = make_env

        # number of processes to spawn
        if processes is None:
            processes = context.cpu_count()
            processes = min(num, processes)
        else:
            assert processes >= 1

        super().__init__(num, processes=int(processes))

        # init processes and pipes
        self.pipes = [context.Pipe() for _ in range(self.num_workers)]
        self.workers = []

        rank = 0
        for i, amount in enumerate(self._get_amount_per_process()):
            parent_pipe, worker_pipe = self.pipes[i]

            worker = context.Process(target=work,
                                     args=(env_fn, rank, amount, seed, worker_pipe, parent_pipe))
            self.workers.append(worker)
            rank += amount

        # with context.Pool(processes=self.num_workers,) as pool:
        #     pass

        # start processes
        for w in self.workers:
            w.daemon = True
            w.start()

        for _, worker_pipe in self.pipes:
            worker_pipe.close()

        # retrieve observation and action spaces
        parent_pipe, _ = self.pipes[0]
        parent_pipe.send(('get_spaces_spec', None))

        obs_space, action_space = parent_pipe.recv().x  # `.x` due to CloudPickleWrapper

        self.observation_space = obs_space
        self.action_space = action_space

    def _get_amount_per_process(self) -> List[int]:
        """Determines the number of sequential environments to be assigned to each process."""
        envs_per_proc = self.num_environments // self.num_workers
        amounts = [envs_per_proc] * self.num_workers
        remaining = self.num_environments - sum(amounts)

        i = 0
        while remaining > 0:
            amounts[i] += 1
            remaining -= 1
            i += 1

            if i >= len(amounts):
                i = 0

        return amounts

    def step(self, action: list) -> tuple:
        actions = np.array_split(action, self.num_workers)  # list of lists of action

        # async
        for action_list, (remote, _) in zip(actions, self.pipes):
            remote.send(('step', action_list))

        # wait
        results = self.receive()
        results = self._flatten_list(results)

        states, rewards, terminals, info = zip(*results)
        info = {env_id: env_info for env_id, env_info in info}

        return self._flatten_obs(states), np.stack(rewards), np.stack(terminals), info

    def render(self, mode='human'):
        raise NotImplementedError

    def reset(self, **kwargs) -> np.ndarray:
        self.broadcast(message=('reset', kwargs))

        states = self.receive()
        states = self._flatten_list(states)
        return self._flatten_obs(states)

    def close(self):
        self.broadcast(message=('close', None))

        for w in self.workers:
            w.join()

    def broadcast(self, message: tuple):
        for remote, _ in self.pipes:
            remote.send(message)

    def receive(self) -> list:
        return [remote.recv() for remote, _ in self.pipes]

    def get_evaluation_env(self) -> gym.Env:
        return self.make_env()


if __name__ == '__main__':
    from memory_profiler import profile

    @profile
    def sequential_env(num: int, env_name='CartPole-v1'):
        env = SequentialEnv(env=env_name, num=num, seed=42)
        env.reset()

        for _ in range(100):
            env.step([env.action_space.sample() for _ in range(16)])

        env.close()

    @profile
    def multiproc_env(num: int, proc: int, env_name='CartPole-v1'):
        env = MultiProcessEnv(env=env_name, num=num, seed=42, processes=proc)
        env.reset()
        for _ in range(100):
            env.step([env.action_space.sample() for _ in range(num)])
        env.close()

    # sequential_env(num=16)
    multiproc_env(num=16, proc=8)
