import multiprocessing as mp
from typing import List, Callable

import numpy as np


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    # Auto-reset if done
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'close':
                env.close()
                remote.close()
                break
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == 'set_reward_function':
                env.set_reward_function(data)
                remote.send(None)
            elif cmd == 'get_reward_state':
                remote.send(env.get_reward_state())
            elif cmd == 'set_reward_state':
                env.set_reward_state(data)
                remote.send(None)
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        env.close()

class SubprocVecEnv:
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    """
    def __init__(self, env_fns: List[Callable], start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            # Fork is not thread safe on MP
            start_method = 'spawn'

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.ps = [ctx.Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.n_envs = n_envs

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return self._flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return self._flatten_obs(obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def set_reward_function(self, reward_fn):
        for remote in self.remotes:
            remote.send(('set_reward_function', reward_fn))
        for remote in self.remotes:
            remote.recv()

    def get_reward_state(self):
        for remote in self.remotes:
            remote.send(('get_reward_state', None))
        return [remote.recv() for remote in self.remotes]

    def set_reward_state(self, states):
        for remote, state in zip(self.remotes, states):
            remote.send(('set_reward_state', state))
        for remote in self.remotes:
            remote.recv()

    def _flatten_obs(self, obs_list):
        """
        Flatten a list of observation dicts into a dict of arrays
        [{'image': ..., 'state': ...}, ...] -> {'image': np.array([...]), 'state': np.array([...])}
        """
        keys = obs_list[0].keys()
        flattened = {}
        for k in keys:
            flattened[k] = np.stack([o[k] for o in obs_list])
        return flattened

    @property
    def action_space(self):
        # Assume all envs have same action space
        # We need to get it from the first env, but we can't easily access it directly
        # So we'll just return a dummy or require it to be passed in
        pass

