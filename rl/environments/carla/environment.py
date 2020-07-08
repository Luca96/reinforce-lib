import os
import gym
import time
import enum
import random
import carla
import pygame
import numpy as np

from gym import spaces
from typing import Callable, Dict, Union
from pygame.constants import K_q, K_UP, K_w, K_LEFT, K_a, K_RIGHT, K_d, K_DOWN, K_s, K_SPACE, K_ESCAPE, KMOD_CTRL

from rl import utils as rl_utils
from rl.environments.carla import env_utils
from rl.environments.carla.sensors import Sensor, SensorSpecs

from rl.environments.carla.navigation.behavior_agent import BehaviorAgent
from rl.environments.carla.navigation import Route, RoutePlanner, RoadOption

from rl.environments.carla.tools import misc, utils
from rl.environments.carla.tools.utils import WAYPOINT_DICT
from rl.environments.carla.tools.synchronous_mode import CARLASyncContext


# TODO: add more events
class CARLAEvent(enum.Enum):
    """Available events (callbacks) related to CARLAEnvironment"""
    RESET = 0


# -------------------------------------------------------------------------------------------------
# -- Base Class and Wrappers
# -------------------------------------------------------------------------------------------------

# TODO: implement 'observation skip'
# TODO: implement 'action repetition'?
class CARLABaseEnvironment(gym.Env):
    """Base extendable environment for the CARLA driving simulator"""

    def __init__(self, address='localhost', port=2000, timeout=5.0, image_shape=(150, 200, 3),
                 window_size=(800, 600), vehicle_filter='vehicle.tesla.model3', fps=30.0, render=True, debug=True,
                 path: dict = None, skip_frames=30):
        super().__init__()
        env_utils.init_pygame()

        self.timeout = timeout
        self.client = env_utils.get_client(address, port, self.timeout)
        self.world: carla.World = self.client.get_world()
        self.synchronous_context = None
        self.sync_mode_enabled = False
        self.num_frames_to_skip = skip_frames

        # TODO: loading map support
        # Map
        self.map: carla.Map = self.world.get_map()

        # set fix fps:
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=False,
            fixed_delta_seconds=1.0 / fps))

        # Vehicle
        self.vehicle_filter = vehicle_filter
        self.vehicle: carla.Vehicle = None
        self.control = carla.VehicleControl()

        # TODO: weather support
        # Weather (default is ClearNoon)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Path: origin, destination, and path-length:
        self.origin_type = 'map'  # 'map' means sample a random point from the world's map
        self.origin = None
        self.destination_type = 'map'
        self.destination = None
        self.path_length = None
        self.use_planner = True
        self.sampling_resolution = 2.0

        if isinstance(path, dict):
            origin_spec = path.get('origin', None)
            destination_spec = path.get('destination', None)
            self.path_length = path.get('length', None)

            # Origin:
            if isinstance(origin_spec, carla.Transform):
                self.origin = origin_spec
                self.origin_type = 'fixed'

            elif isinstance(origin_spec, dict):
                if 'point' in origin_spec:
                    self.origin = origin_spec['point']
                    self.origin_type = origin_spec.get('type', 'fixed')

                    assert isinstance(self.origin, carla.Transform)
                    assert self.origin_type in ['map', 'fixed', 'route']

                elif 'points' in origin_spec:
                    self.origins = origin_spec['points']
                    self.origin = None
                    self.origin_index = -1
                    self.origin_type = origin_spec.get('type', 'random')

                    assert isinstance(self.origins, list) and len(self.origins) > 0
                    assert all(isinstance(x, carla.Transform) for x in self.origins)
                    assert self.origin_type in ['random', 'sequential']

            # Destination:
            if isinstance(destination_spec, carla.Location):
                self.destination = destination_spec
                self.destination_type = 'fixed'

            elif isinstance(destination_spec, dict):
                if 'point' in destination_spec:
                    self.destination = destination_spec['point']
                    self.destination_type = destination_spec.get('type', 'fixed')

                    assert isinstance(self.destination, carla.Location)
                    assert self.destination_type in ['map', 'fixed']

                elif 'points' in destination_spec:
                    self.destinations = destination_spec['points']
                    self.destination = None
                    self.destination_index = -1
                    self.destination_type = destination_spec.get('type', 'random')

                    assert isinstance(self.destinations, list) and len(self.destinations) > 0
                    assert all(isinstance(x, carla.Location) for x in self.destinations)
                    assert self.destination_type in ['random', 'sequential']

            # Path stuff:
            self.path_length = path.get('length', None)
            self.use_planner = path.get('use_planner', True)
            self.sampling_resolution = path.get('sampling_resolution', 2.0)

            if self.origin_type == 'route':
                assert self.destination_type == 'fixed'
                assert self.use_planner is True

        elif path is not None:
            raise ValueError('Argument [path] must be either "None" or a "dict".')

        # Path-planning:
        if self.use_planner:
            self.route = Route(planner=RoutePlanner(map=self.map, sampling_resolution=self.sampling_resolution))
        else:
            self.route = None

        # Visualization and Debugging
        self.image_shape = image_shape
        self.image_size = (image_shape[1], image_shape[0])
        self.fps = fps
        self.tick_time = 1.0 / self.fps
        self.should_render = render
        self.should_debug = debug
        self.clock = pygame.time.Clock()

        if self.should_render:
            self.render_data = None  # some sensor_data to be rendered in render()
            self.window_size = window_size
            self.font = env_utils.get_font(size=13)
            self.display = env_utils.get_display(window_size)

        # vehicle sensors suite
        self.sensors = dict()

        # events and callbacks
        self.events: Dict[CARLAEvent, Callable] = dict()

    @property
    def observation_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def action_space(self) -> spaces.Space:
        raise NotImplementedError

    @property
    def reward_range(self) -> tuple:
        raise NotImplementedError

    def reset(self) -> dict:
        print('env.reset')
        self.reset_world()
        self.trigger_event(event=CARLAEvent.RESET)

        if not self.sync_mode_enabled:
            self.__enter__()

        self.control = carla.VehicleControl()
        self.skip(num_frames=self.num_frames_to_skip)

        observation = env_utils.replace_nans(self.get_observation(sensors_data={}))
        return observation

    def reward(self, actions, **kwargs):
        """Agent's reward function"""
        raise NotImplementedError

    @staticmethod
    def consume_pygame_events():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    return True

        return False

    def step(self, actions):
        pygame.event.get()
        self.clock.tick()

        sensors_data = self.world_step(actions)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = env_utils.replace_nans(self.get_observation(sensors_data))

        return next_state, reward, terminal, {}

    def terminal_condition(self, **kwargs) -> Union[bool, int]:
        """Tells whether the episode is terminated or not."""
        raise NotImplementedError

    def close(self):
        print('env.close')
        super().close()

        if self.vehicle:
            self.vehicle.destroy()

        if self.sync_mode_enabled:
            self.__exit__()

        for sensor in self.sensors.values():
            sensor.destroy()

    def define_sensors(self) -> dict:
        """Define which sensors should be equipped to the vehicle"""
        raise NotImplementedError

    def on_collision(self, event: carla.CollisionEvent, **kwargs):
        raise NotImplementedError

    def register_event(self, event: CARLAEvent, callback):
        """Registers a given [callback] to a specific [event]"""
        assert isinstance(event, CARLAEvent)
        assert callable(callback)

        callbacks = self.events.get(event, [])
        callbacks.append(callback)
        self.events[event] = callbacks

    def trigger_event(self, event: CARLAEvent, **kwargs):
        """Cause the call of every callback registered for event [event]"""
        print(f'Event {str(event)} triggered.')
        for callback in self.events.get(event, []):
            callback(**kwargs)

    def render(self, mode='human'):
        """Renders sensors' output"""
        raise NotImplementedError

    def debug(self, actions):
        env_utils.display_text(self.display, self.font, text=self.debug_text(actions), origin=(16, 12),
                               offset=(0, 16))

    def debug_text(self, actions):
        raise NotImplementedError

    def skip(self, num_frames=10):
        """Skips the given amount of frames"""
        for _ in range(num_frames):
            self.synchronous_context.tick(timeout=self.timeout)

        if num_frames > 0:
            print(f'Skipped {num_frames} frames.')

    def control_to_actions(self, control: carla.VehicleControl):
        raise NotImplementedError("Implement only if needed for pre-training.")

    def before_world_step(self):
        """Callback: called before world.tick()"""
        pass

    def after_world_step(self, sensors_data: dict):
        """Callback: called after world.tick()."""
        pass

    @staticmethod
    def on_sensors_data(data: dict) -> dict:
        """Callback. Triggers when a world's 'tick' occurs, meaning that data from sensors are been collected because a
        simulation step of the CARLA's world has been completed.
            - Use this method to preprocess sensors' output data for: rendering, observation, ...
        """
        return data

    def __enter__(self):
        """Enables synchronous mode.
           Usage:
              with carla_env as env:
                 # code...
        """
        self.synchronous_context.__enter__()
        self.sync_mode_enabled = True
        return self

    def __exit__(self, *args):
        """Disables synchronous mode"""
        self.synchronous_context.__exit__()
        self.sync_mode_enabled = False

        # propagate exception
        return False

    def world_step(self, actions):
        """Applies the actions to the vehicle, and updates the CARLA's world"""
        # [pre-tick updates] Apply control to update the vehicle
        self.actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        self.before_world_step()

        # Advance the simulation and wait for sensors' data.
        data = self.synchronous_context.tick(timeout=self.timeout)
        data = self.on_sensors_data(data)

        # [post-tick updates] Update world-related stuff
        self.after_world_step(data)

        # Draw and debug:
        if self.should_render:
            self.render_data = data
            self.render()
            self.render_data = None

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

        return data

    def reset_world(self):
        # choose origin (spawn point)
        if self.origin_type == 'map':
            self.origin = env_utils.random_spawn_point(self.map)

        elif self.origin_type == 'random':
            self.origin = random.choice(self.origins)

        elif self.origin_type == 'sequential':
            self.origin_index = (self.origin_index + 1) % len(self.origins)
            self.origin = self.origins[self.origin_index]

        # choose destination (final point)
        if self.destination_type == 'map':
            self.destination = env_utils.random_spawn_point(self.map, different_from=self.origin.location).location

        elif self.destination_type == 'random':
            self.destination = random.choice(self.destinations)  # TODO: ensure different from origin?

        elif self.destination_type == 'sequential':
            self.destination_index = (self.destination_index + 1) % len(self.destinations)
            self.destination = self.destinations[self.destination_index]

        # plan path between origin and destination
        if self.use_planner:
            self.route.plan(origin=self.origin.location, destination=self.destination)

        # spawn actor
        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle: carla.Vehicle = env_utils.spawn_actor(self.world, blueprint, self.origin)

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            self.vehicle.apply_control(carla.VehicleControl())
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))

            if self.origin_type == 'route':
                new_origin = self.route.random_waypoint().transform
                self.vehicle.set_transform(new_origin)
            else:
                self.vehicle.set_transform(self.origin)

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        raise NotImplementedError

    def get_observation(self, sensors_data: dict) -> dict:
        raise NotImplementedError

    def _create_sensors(self):
        for name, args in self.define_sensors().items():
            kwargs = args.copy()
            sensor = Sensor.create(sensor_type=kwargs.pop('type'), parent_actor=self.vehicle, **kwargs)

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor


class CARLAWrapper:
    pass


class CARLAPlayWrapper(CARLAWrapper):
    """Makes an already instantiated CARLAEnvironment be playable with a keyboard"""
    CONTROL = dict(type='float', shape=(5,), min_value=-1.0, max_value=1.0,
                   default=[0.0, 0.0, 0.0, 0.0, 0.0])

    def __init__(self, env: CARLABaseEnvironment):
        print('Controls: (W, or UP) accelerate, (A or LEFT) steer left, (D or RIGHT) steer right, (S or DOWN) brake, '
              '(Q) toggle reverse, (SPACE) hand-brake, (ESC) quit.')
        self.env = env
        self._steer_cache = 0.0

        # Wrap environment's methods:
        self.env.actions_to_control = lambda actions: self.actions_to_control(self.env, actions)
        self.env.before_world_step = lambda: self.before_world_step(self.env)

    def reset(self) -> dict:
        self._steer_cache = 0.0
        return self.env.reset()

    def play(self):
        """Let's you control the vehicle with a keyboard."""
        state = self.reset()
        done = False

        try:
            with self.env.synchronous_context:
                while not done:
                    actions = self.get_action(state)
                    state, reward, done, info = self.env.step(actions)
                    self.observe(reward, done)
        finally:
            self.env.close()

    def get_action(self, state):
        return self._parse_events()

    def observe(self, reward, done):
        pass

    def _parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    raise Exception('closing...')

                elif event.key == K_q:
                    self.env.control.gear = 1 if self.env.control.reverse else -1

        return self._parse_vehicle_keys()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

    def _parse_vehicle_keys(self):
        keys = pygame.key.get_pressed()
        steer_increment = 5e-4 * self.env.clock.get_time()

        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment

        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(1.0, max(-1.0, self._steer_cache))
        self.env.control.reverse = self.env.control.gear < 0

        # actions
        throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer = round(self._steer_cache, 1)
        brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        reverse = 1.0 if self.env.control.reverse else 0.0
        hand_brake = keys[K_SPACE]

        return [throttle, steer, brake, reverse, hand_brake]

    @staticmethod
    def actions_to_control(env, actions):
        env.control.throttle = actions[0]
        env.control.steer = actions[1]
        env.control.brake = actions[2]
        env.control.reverse = bool(actions[3])
        env.control.hand_brake = bool(actions[4])

    @staticmethod
    def before_world_step(env):
        if env.should_debug:
            env.route.draw_route(env.world.debug, life_time=1.0 / env.fps)
            # env.route.draw_next_waypoint(env.world.debug, env.vehicle.get_location(), life_time=1.0 / env.fps)


class CARLACollectWrapper(CARLAWrapper):
    """Wraps a CARLA Environment, collects input observations and output actions that can be later
       used for pre-training or imitation learning purposes.
    """

    def __init__(self, env: CARLABaseEnvironment, ignore_traffic_light: bool, traces_dir='traces', name='carla',
                 behaviour='normal', unnormalize=True,):
        self.env = env
        self.agent = None
        self.agent_behaviour = behaviour  # 'normal', 'cautious', or 'aggressive'
        self.ignore_traffic_light = ignore_traffic_light

        # Saving & Buffers
        self.save_dir = rl_utils.makedir(traces_dir, name)
        print('save_dir:', self.save_dir)
        self.buffer = None
        self.timestep = 0

    def reset(self) -> dict:
        self.timestep = 0
        observation = self.env.reset()

        self.agent = BehaviorAgent(vehicle=self.env.vehicle, behavior=self.agent_behaviour,
                                   ignore_traffic_light=self.ignore_traffic_light)
        self.agent.set_destination(start_location=self.env.vehicle.get_location(), end_location=self.env.destination,
                                   clean=True)
        return observation

    def collect(self, episodes: int, timesteps: int, agent_debug=False, episode_reward_threshold=0.0):
        self.init_buffer(num_timesteps=timesteps)
        env = self.env

        for episode in range(episodes):
            state = self.reset()
            episode_reward = 0.0

            for t in range(timesteps):
                # act
                self.agent.update_information()
                control = self.agent.run_step(debug=agent_debug)
                action = env.control_to_actions(control)

                # step
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                # record
                self.store_transition(state=state, action=action, reward=reward, done=done)
                state = next_state

                if done or (t == timesteps - 1):
                    buffer = self.end_trajectory()
                    break

            if episode_reward > episode_reward_threshold:
                self.serialize(buffer, episode)
            else:
                print(f'Trace-{episode} discarded because reward={round(episode_reward, 2)} below threshold!')

        env.close()

    def init_buffer(self, num_timesteps: int):
        # partial buffer: misses 'state' and 'action'
        self.buffer = dict(reward=np.zeros(shape=num_timesteps),
                           done=np.zeros(shape=num_timesteps))

        obs_spec = rl_utils.space_to_spec(space=self.env.observation_space)
        act_spec = rl_utils.space_to_spec(space=self.env.action_space)

        print('obs_spec\n', obs_spec)
        print('action_spec\n', act_spec)

        # include in buffer 'state' and 'action'
        self._apply_space_spec(spec=obs_spec, size=num_timesteps, name='state')
        self._apply_space_spec(spec=act_spec, size=num_timesteps, name='action')
        self.timestep = 0

    def store_transition(self, **kwargs):
        """Collects one transition (s, a, r, d)"""
        for name, value in kwargs.items():
            self._store_item(item=value, index=self.timestep, name=name)

        self.timestep += 1

    def end_trajectory(self) -> dict:
        """Ends a sequence of transitions {(s, a, r, d)_t}"""
        # Add the reward for the terminal/final state:
        self.buffer['reward'] = np.concatenate([self.buffer['reward'], np.array([0.0])])

        # Duplicate the buffer and cut off the exceeding part (if any)
        buffer = dict()

        for key, value in self.buffer.items():
            buffer[key] = value[:self.timestep]

        buffer['reward'] = self.buffer['reward'][:self.timestep + 1]
        return buffer

    def serialize(self, buffer: dict, episode: int):
        """Writes to file (npz - numpy compressed format) all the transitions collected so far"""
        # Trace's file path:
        filename = f'trace-{episode}-{time.strftime("%Y%m%d-%H%M%S")}.npz'
        trace_path = os.path.join(self.save_dir, filename)

        # Save buffer
        np.savez_compressed(file=trace_path, **buffer)

    def _apply_space_spec(self, spec: Union[tuple, dict], size: int, name: str):
        if not isinstance(spec, dict):
            shape = (size,) + spec
            self.buffer[name] = np.zeros(shape=shape, dtype=np.float32)
            return

        # use recursion + names to handle arbitrary nested dicts and recognize them
        for spec_name, sub_spec in spec.items():
            self._apply_space_spec(spec=sub_spec, size=size, name=f'{name}_{spec_name}')

    def _store_item(self, item, index: int, name: str):
        if not isinstance(item, dict):
            self.buffer[name][index] = item
            return

        # recursion
        for key, value in item.items():
            self._store_item(item=value, index=index, name=f'{name}_{key}')


# TODO: 'CARLAObservationWrapper' takes (obs, r, d, info) in env.step() and wraps obs as a temporal
#  sequence with N frames skipping.
#  - change state space according to 'temporal horizon
class CARLAObservationWrapper(gym.ObservationWrapper):

    def __init__(self, env: CARLABaseEnvironment, temporal_horizon=1, consider_obs_every=1):
        super().__init__(env)

        self.temporal_horizon = temporal_horizon
        self.consider_obs_every = consider_obs_every

        # Expand observation space according to temporal horizon (i.e. add a temporal dimension):
        self._expand_space_shape(self.observation_space)

    def observation(self, observation):
        if isinstance(observation, dict):
            # Complex observation
            for name, obs in observation.items():
                new_obs = self.observation(obs)
                observation[name] = new_obs

            return observation
        else:
            if rl_utils.is_image(observation):
                arrays = (observation,) * self.temporal_horizon
                return rl_utils.depth_concat(*arrays)

            elif rl_utils.is_vector(observation):
                pass

    def _expand_space_shape(self, space: gym.Space):
        if isinstance(space, spaces.Box):
            if rl_utils.is_image(space):
                space.shape = space.shape[:2] + (space.shape[2] * self.temporal_horizon,)

            elif rl_utils.is_vector(space):
                space.shape = (self.temporal_horizon,) + space.shape
            else:
                raise ValueError(f'Unsupported space type: {type(space)}!')

        elif isinstance(space, spaces.Dict):
            for _, sub_space in space.spaces.items():
                self._expand_space_shape(space=sub_space)
        else:
            raise ValueError(f'Unsupported space type: {type(space)}!')


class CARLARecordWrapper:
    """Wraps a CARLA Environment in order to record input observations"""
    pass


# -------------------------------------------------------------------------------------------------
# -- Implemented CARLA Environments
# -------------------------------------------------------------------------------------------------

class OneCameraCARLAEnvironment(CARLABaseEnvironment):
    """One camera (front) CARLA Environment"""
    # Control: throttle or brake, steer, reverse
    ACTION = dict(space=spaces.Box(low=-1, high=1, shape=(3,)), default=np.zeros(shape=3, dtype=np.float32))
    CONTROL = dict(space=spaces.Box(low=-1, high=1, shape=(4,)), default=np.zeros(shape=4, dtype=np.float32))

    # Vehicle: speed, acceleration, angular velocity, similarity, distance to waypoint
    VEHICLE_FEATURES = dict(space=spaces.Box(low=np.array([0.0, -np.inf, 0.0, -1.0, -np.inf]),
                                             high=np.array([15.0, np.inf, np.inf, 1.0, np.inf])),
                            default=np.zeros(shape=5, dtype=np.float32))

    # Road: intersection (bool), junction (bool), speed_limit, traffic_light (presence + state), lane type and change,
    ROAD_FEATURES = dict(space=spaces.Box(low=np.zeros(shape=(9,)),
                                          high=np.array([1.0, 1.0, 15.0, 1.0, 4.0, 2.0, 10.0, 10.0, 3.0])),
                         default=np.zeros(shape=9, dtype=np.float32))

    # High-level routing command (aka RoadOption)
    COMMAND_SPACE = spaces.Box(low=0.0, high=1.0, shape=(len(RoadOption),))

    # Radar: velocity, altitude, azimuth, depth
    # RADAR_SPACE = dict(space=spaces.Box(low=-np.inf, high=np.inf, shape=(50, 4)), default=None)

    def __init__(self, *args, disable_reverse=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.disable_reverse = disable_reverse
        self.image_space = spaces.Box(low=-1.0, high=1.0, shape=self.image_shape)

        # reward computation
        self.collision_penalty = 0.0
        self.should_terminate = False
        self.similarity = 0.0
        self.forward_vector = None

        # TODO: use 'next_command' to compute "action penalty"?
        self.next_command = RoadOption.VOID

        self.last_actions = self.ACTION['default']
        self.last_location = None
        self.last_travelled_distance = 0.0
        self.total_travelled_distance = 0.0

        # Observations
        self.default_image = np.zeros(shape=self.image_shape, dtype=np.uint8)

    @property
    def action_space(self) -> spaces.Space:
        return self.ACTION['space']

    @property
    def observation_space(self) -> spaces.Space:
        return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
                           past_control=self.CONTROL['space'], command=self.COMMAND_SPACE, image=self.image_space)

    @property
    def reward_range(self) -> tuple:
        return -float('inf'), float('inf')

    def reward(self, actions, time_cost=-1, d=2.0, w=3.0, s=2.0, v_max=150.0, d_max=100.0, **kwargs) -> float:
        # Direction term: alignment of the vehicle's forward vector with the waypoint's forward vector
        speed = min(utils.speed(self.vehicle), v_max)

        if 0.8 <= self.similarity <= 1.0:
            direction_penalty = speed * self.similarity
        else:
            direction_penalty = (speed + 1.0) * abs(self.similarity) * -d

        # Distance from waypoint (and also lane center)
        waypoint_term = min(self.route.distance_to_next_waypoint(), d_max)
        waypoint_term = -waypoint_term if waypoint_term <= 5.0 else waypoint_term * -w

        # Speed-limit compliance:
        speed_limit = self.vehicle.get_speed_limit()
        speed_penalty = s * (speed_limit - speed) if speed > speed_limit else 0.0

        # Risk penalty discourages long action's horizon (validity)
        # risk_penalty = x**(self.last_actions['validity'] - 1)

        return time_cost - self.collision_penalty + waypoint_term + direction_penalty + speed_penalty

    def step(self, actions):
        state, reward, done, info = super().step(actions)
        self.collision_penalty = 0.0
        self.last_travelled_distance = 0.0

        return state, reward, done, info

    def reset(self) -> dict:
        self.last_actions = self.ACTION['default']
        self.should_terminate = False
        self.total_travelled_distance = 0.0
        self.last_travelled_distance = 0.0
        self.next_command = RoadOption.VOID

        # reset observations:
        observation = super().reset()

        self.last_location = self.vehicle.get_location()
        # self.last_location = self.origin.location
        return observation

    def terminal_condition(self, **kwargs) -> Union[bool, int]:
        if self.should_terminate:
            return 2

        return self.route.distance_to_destination(self.vehicle.get_location()) < 2.0

    def define_sensors(self) -> dict:
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    camera=SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                           image_size_x=self.image_size[0],
                                                           image_size_y=self.image_size[1],
                                                           sensor_tick=self.tick_time),
                    depth=SensorSpecs.depth_camera(position='on-top2', attachment_type='Rigid',
                                                   image_size_x=self.image_size[0],
                                                   image_size_y=self.image_size[1],
                                                   sensor_tick=self.tick_time),
                    # radar=SensorSpecs.radar(position='radar', points_per_second=1000,
                    #                         sensor_tick=self.tick_time)
                    )

    def on_collision(self, event: carla.CollisionEvent, penalty=1000.0):
        actor_type = event.other_actor.type_id
        print(f'Collision with actor={actor_type})')

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += penalty / 100.0
            self.should_terminate = False

    def render(self, mode='human'):
        assert self.render_data is not None
        image = self.render_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

        # radar_data = self.render_data['radar']
        # print('radar count:', radar_data.get_detection_count())
        # env_utils.draw_radar_measurement(debug_helper=self.world.debug, data=radar_data)

    def debug_text(self, actions):
        speed_limit = self.vehicle.get_speed_limit()
        speed = utils.speed(self.vehicle)
        distance = self.total_travelled_distance

        if speed > speed_limit:
            speed_text = dict(text='Speed %.1f km/h' % speed, color=(255, 0, 0))
        else:
            speed_text = 'Speed %.1f km/h' % speed

        return ['%d FPS' % self.clock.get_fps(),
                '',
                'Throttle: %.2f' % self.control.throttle,
                'Steer: %.2f' % self.control.steer,
                'Brake: %.2f' % self.control.brake,
                'Reverse: %s' % ('T' if self.control.reverse else 'F'),
                '',
                speed_text,
                'Speed limit %.1f km/h' % speed_limit,
                'Distance travelled %.2f %s' % ((distance / 1000.0, 'km') if distance > 1000.0 else (distance, 'm')),
                '',
                'Similarity %.2f' % self.similarity,
                'Waypoint\'s Distance %.2f' % self.route.distance_to_next_waypoint(),
                'Route Option: %s' % self.next_command.name,
                'OP: %s' % self.next_command.to_one_hot(),
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty]

    def control_to_actions(self, control: carla.VehicleControl):
        reverse = bool(control.reverse)

        if control.throttle > 0.0:
            return [control.throttle, control.steer, reverse]

        return [-control.brake, control.steer, reverse]

    def on_sensors_data(self, data: dict) -> dict:
        data['camera'] = self.sensors['camera'].convert_image(data['camera'])
        data['depth'] = self.sensors['depth'].convert_image(data['depth'])

        # include depth information in one image:
        data['camera_plus_depth'] = np.multiply(1 - data['depth'] / 255.0, data['camera'])

        if self.image_shape[2] == 1:
            data['camera'] = env_utils.cv2_grayscale(data['camera_plus_depth'])
        else:
            data['camera'] = data['camera_plus_depth']

        # data['radar_converted'] = self.sensors['radar'].convert(data['radar'])
        return data

    def after_world_step(self, sensors_data: dict):
        self._update_env_state()

    def actions_to_control(self, actions):
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.reverse = bool(actions[2] > 0)
        self.control.hand_brake = False

        if self.disable_reverse:
            self.control.reverse = False

    def get_observation(self, sensors_data: dict) -> dict:
        if len(sensors_data.keys()) == 0:
            # sensor_data is empty so, return a default observation
            return dict(image=self.default_image, vehicle=self.VEHICLE_FEATURES['default'],
                        road=self.ROAD_FEATURES['default'], past_control=self.CONTROL['default'],
                        command=RoadOption.VOID.to_one_hot())

        # resize image if necessary
        image = sensors_data['camera']

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # -1, +1 scaling
        image = (2 * image - 255.0) / 255.0

        # observations
        vehicle_obs = self._get_vehicle_features()
        control_obs = self._control_as_vector()
        road_obs = self._get_road_features()

        return dict(image=image, vehicle=vehicle_obs, road=road_obs, past_control=control_obs,
                    command=self.next_command.to_one_hot())

    def _control_as_vector(self) -> list:
        return [self.control.throttle, self.control.brake, self.control.steer, float(self.control.reverse)]

    def _get_road_features(self):
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit()
        is_at_traffic_light = self.vehicle.is_at_traffic_light()

        if is_at_traffic_light:
            traffic_light_state = self.vehicle.get_traffic_light_state()
        else:
            traffic_light_state = carla.TrafficLightState.Unknown

        # get current lane type: consider only road (driving) lanes
        if waypoint.lane_type is carla.LaneType.NONE:
            lane_type = 0
        elif waypoint.lane_type is carla.LaneType.Driving:
            lane_type = 1
        else:
            lane_type = 2  # other

        return [waypoint.is_intersection,
                waypoint.is_junction,
                round(speed_limit / 10.0),
                # Traffic light:
                is_at_traffic_light,
                WAYPOINT_DICT['traffic_light'][traffic_light_state],
                # Lanes:
                lane_type,
                WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type],
                WAYPOINT_DICT['lane_change'][waypoint.lane_change]]

    def _get_vehicle_features(self):
        imu_sensor = self.sensors['imu']

        # vehicle's acceleration (also considers direction)
        acceleration = env_utils.magnitude(imu_sensor.accelerometer) * env_utils.sign(self.similarity)

        # vehicle's angular velocity
        angular_velocity = env_utils.magnitude(imu_sensor.gyroscope)

        return [utils.speed(self.vehicle) / 10.0,
                acceleration,
                angular_velocity,
                # Target (next) waypoint's features:
                self.similarity,
                self.route.distance_to_next_waypoint()]

    # TODO: move to base class
    def _update_env_state(self):
        if self.use_planner:
            self._update_target_waypoint()
            self._update_waypoint_similarity()
            self.next_command = self.route.next.road_op

        self._update_travelled_distance()

    def _update_target_waypoint(self):
        self.route.update_next_waypoint(location=self.vehicle.get_location())

    def _update_waypoint_similarity(self):
        self.forward_vector = self.vehicle.get_transform().get_forward_vector()
        self.similarity = utils.cosine_similarity(self.forward_vector,
                                                  self.route.next.waypoint.transform.get_forward_vector())

    # TODO: move to base class
    def _update_travelled_distance(self):
        location1 = self.last_location
        location2 = self.vehicle.get_location()

        self.last_travelled_distance = misc.compute_distance(location1, location2)
        self.total_travelled_distance += abs(self.last_travelled_distance)
        self.last_location = location2


class ThreeCameraCARLAEnvironment(OneCameraCARLAEnvironment):
    """Three Camera (front, lateral left and right) CARLA Environment"""

    def __init__(self, *args, image_shape=(120, 160, 1), window_size=(600, 300), **kwargs):
        # Make the shape of the final image three times larger to account for the three cameras
        image_shape = (image_shape[0], image_shape[1] * 3, image_shape[2])

        super().__init__(*args, image_shape=image_shape, window_size=window_size, **kwargs)
        self.image_size = (image_shape[1] // 3, image_shape[0])

    # TODO: add 'depth camera' only to frontal camera, add 'radar' sensor
    def define_sensors(self) -> dict:
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    front_camera=SensorSpecs.segmentation_camera(position='on-top2', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time),
                    depth=SensorSpecs.depth_camera(position='on-top2', attachment_type='Rigid',
                                                   image_size_x=self.image_size[0],
                                                   image_size_y=self.image_size[1],
                                                   sensor_tick=self.tick_time),
                    left_camera=SensorSpecs.segmentation_camera(position='lateral-left', attachment_type='Rigid',
                                                                image_size_x=self.image_size[0],
                                                                image_size_y=self.image_size[1],
                                                                sensor_tick=self.tick_time),
                    right_camera=SensorSpecs.segmentation_camera(position='lateral-right', attachment_type='Rigid',
                                                                 image_size_x=self.image_size[0],
                                                                 image_size_y=self.image_size[1],
                                                                 sensor_tick=self.tick_time))

    def render(self, mode='human'):
        assert self.render_data is not None
        image = self.render_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def on_sensors_data(self, data: dict) -> dict:
        front_image = self.sensors['front_camera'].convert_image(data['front_camera'])
        left_image = self.sensors['left_camera'].convert_image(data['left_camera'])
        right_image = self.sensors['right_camera'].convert_image(data['right_camera'])

        # include depth information in one image:
        data['depth'] = self.sensors['depth'].convert_image(data['depth'])
        front_image = np.multiply(1 - data['depth'] / 255.0, front_image)

        # Concat images
        data['camera'] = np.concatenate((left_image, front_image, right_image), axis=1)

        if self.image_shape[2] == 1:
            data['camera'] = env_utils.cv2_grayscale(data['camera'])

        return data
