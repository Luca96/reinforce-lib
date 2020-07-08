"""Utility functions for CARLAEnvironment organized by subject:
    - PyGame-related,
    - CARLA-related,
    - various (graphics, file, ...)
    - math-related.

   @author: Luca Anzalone
"""

import os
import cv2
import math
import random
import numpy as np
import carla
import pygame
import threading
import datetime

from typing import Tuple, Union, List


# -------------------------------------------------------------------------------------------------
# -- PyGame
# -------------------------------------------------------------------------------------------------

def init_pygame():
    if not pygame.get_init():
        pygame.init()

    if not pygame.font.get_init():
        pygame.font.init()


def get_display(window_size, mode=pygame.HWSURFACE | pygame.DOUBLEBUF):
    """Returns a display used to render images and text.
        :param window_size: a tuple (width: int, height: int)
        :param mode: pygame rendering mode. Default: pygame.HWSURFACE | pygame.DOUBLEBUF
        :return: a pygame.display instance.
    """
    return pygame.display.set_mode(window_size, mode)


def get_font(size=14):
    return pygame.font.Font(pygame.font.get_default_font(), size)


def display_image(display, image, window_size=(800, 600), blend=False):
    """Displays the given image on a pygame window
    :param blend: whether to blend or not the given image.
    :param window_size: the size of the pygame's window. Default is (800, 600)
    :param display: pygame.display
    :param image: the image (numpy.array) to display/render on.
    """
    # Resize image if necessary
    if (image.shape[1], image.shape[0]) != window_size:
        image = resize(image, size=window_size)

    if len(image.shape) == 2:
        # duplicate image three times along depth if grayscale
        image = np.stack((image,) * 3, axis=-1)

    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

    if blend:
        image_surface.set_alpha(100)

    display.blit(image_surface, (0, 0))


def display_text(display, font, text: [str], color=(255, 255, 255), origin=(0, 0), offset=(0, 2)):
    position = origin

    for line in text:
        if isinstance(line, dict):
            display.blit(font.render(line.get('text'), True, line.get('color', color)), position)
        else:
            display.blit(font.render(line, True, color), position)

        position = (position[0] + offset[0], position[1] + offset[1])


def pygame_save(display, path: str, name: str = None):
    if name is None:
        name = 'image-' + str(datetime.datetime.now()) + '.jpg'

    thread = threading.Thread(target=lambda: pygame.image.save(display, os.path.join(path, name)))
    thread.start()


# -------------------------------------------------------------------------------------------------
# -- CARLA
# -------------------------------------------------------------------------------------------------

def get_client(address, port, timeout=2.0) -> carla.Client:
    """Connects to the simulator.
        @:returns a carla.Client instance if the CARLA simulator accepts the connection.
    """
    client: carla.Client = carla.Client(address, port)
    client.set_timeout(timeout)
    return client


def random_blueprint(world: carla.World, actor_filter='vehicle.*', role_name='agent') -> carla.ActorBlueprint:
    """Retrieves a random blueprint.
        :param world: a carla.World instance.
        :param actor_filter: a string used to filter (select) blueprints. Default: 'vehicle.*'
        :param role_name: blueprint's role_name, Default: 'agent'.
        :return: a carla.ActorBlueprint instance.
    """
    blueprints = world.get_blueprint_library().filter(actor_filter)
    blueprint: carla.ActorBlueprint = random.choice(blueprints)
    blueprint.set_attribute('role_name', role_name)

    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)

    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')

    # set the max speed
    if blueprint.has_attribute('speed'):
        float(blueprint.get_attribute('speed').recommended_values[1])
        float(blueprint.get_attribute('speed').recommended_values[2])
    else:
        print("No recommended values for 'speed' attribute")

    return blueprint


def random_spawn_point(world_map: carla.Map, different_from: carla.Location = None) -> carla.Transform:
    """Returns a random spawning location.
        :param world_map: a carla.Map instance obtained by calling world.get_map()
        :param different_from: ensures that the location of the random spawn point is different from the one specified here.
        :return: a carla.Transform instance.
    """
    available_spawn_points = world_map.get_spawn_points()

    if different_from is not None:
        while True:
            spawn_point = random.choice(available_spawn_points)

            if spawn_point.location != different_from:
                return spawn_point
    else:
        return random.choice(available_spawn_points)


def spawn_actor(world: carla.World, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform,
                attach_to: carla.Actor = None, attachment_type=carla.AttachmentType.Rigid) -> carla.Actor:
    """Tries to spawn an actor in a CARLA simulator.
        :param world: a carla.World instance.
        :param blueprint: specifies which actor has to be spawned.
        :param spawn_point: where to spawn the actor. A transform specifies the location and rotation.
        :param attach_to: whether the spawned actor has to be attached (linked) to another one.
        :param attachment_type: the kind of the attachment. Can be 'Rigid' or 'SpringArm'.
        :return: a carla.Actor instance.
    """
    actor = world.try_spawn_actor(blueprint, spawn_point, attach_to, attachment_type)

    if actor is None:
        raise ValueError(f'Cannot spawn actor. Try changing the spawn_point ({spawn_point}) to something else.')

    return actor


def get_blueprint(world: carla.World, actor_id: str) -> carla.ActorBlueprint:
    return world.get_blueprint_library().find(actor_id)


def global_to_local(point: carla.Location, reference: Union[carla.Transform, carla.Location, carla.Rotation]):
    """Translates a 3D point from global to local coordinates using the current transformation as reference"""
    if isinstance(reference, carla.Transform):
        reference.transform(point)
    elif isinstance(reference, carla.Location):
        carla.Transform(reference, carla.Rotation()).transform(point)
    elif isinstance(reference, carla.Rotation):
        carla.Transform(carla.Location(), reference).transform(point)
    else:
        raise ValueError('Argument "reference" is none of carla.Transform or carla.Location or carla.Rotation!')


def draw_radar_measurement(debug_helper: carla.DebugHelper, data: carla.RadarMeasurement, velocity_range=7.5,
                           size=0.075, life_time=0.06):
    """Code adapted from carla/PythonAPI/examples/manual_control.py:
        - White: means static points.
        - Red: indicates points moving towards the object.
        - Blue: denoted points moving away.
    """
    radar_rotation = data.transform.rotation
    for detection in data:
        azimuth = math.degrees(detection.azimuth) + radar_rotation.yaw
        altitude = math.degrees(detection.altitude) + radar_rotation.pitch

        # move to local coordinates:
        forward_vec = carla.Vector3D(x=detection.depth - 0.25)
        global_to_local(forward_vec,
                        reference=carla.Rotation(pitch=altitude, yaw=azimuth, roll=radar_rotation.roll))

        # compute color:
        norm_velocity = detection.velocity / velocity_range
        r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
        g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
        b = int(abs(clamp(-1.0, 0.0, -1.0 - norm_velocity)) * 255.0)

        # draw:
        debug_helper.draw_point(data.transform.location + forward_vec, size=size, life_time=life_time,
                                persistent_lines=False, color=carla.Color(r, g, b))


# -------------------------------------------------------------------------------------------------
# -- Other
# -------------------------------------------------------------------------------------------------

def resize(image, size: (int, int), interpolation=cv2.INTER_CUBIC):
    """Resize the given image.
        :param image: a numpy array with shape (height, width, channels).
        :param size: (width, height) to resize the image to.
        :param interpolation: Default: cv2.INTER_CUBIC.
        :return: the reshaped image.
    """
    return cv2.resize(image, dsize=size, interpolation=interpolation)


def scale(num, from_interval=(-1.0, +1.0), to_interval=(0.0, 7.0)) -> float:
    """Scales (interpolates) the given number to a given interval.
        :param num: a number
        :param from_interval: the interval the number is assumed to lie in.
        :param to_interval: the target interval.
        :return: the scaled/interpolated number.
    """
    x = np.interp(num, from_interval, to_interval)
    return float(round(x))


def get_record_path(base_dir: str, prefix='ep', pattern='-') -> str:
    """Recording directory is organized as follows:
        - A [base_dir], usually `data/recordings` is the main folder.
        - Each new recording is stored within a new folder, named [prefix][pattern][count] where [count] is a number.
        - By default: [prefix] is 'ep' and [pattern] is '-'. So the folders within [base_dir] will be named 'ep-0',
          'ep-1', and so on. The idea is to separate recordings by the episode number.
       :returns a path.
    """
    if not os.path.isdir(base_dir):
        # create base_dir if not exists
        os.mkdir(base_dir)
        count = 0
    else:
        dirs = sorted(os.listdir(base_dir))
        count = 0

        if len(dirs) > 0:
            count = 1 + int(dirs[-1].split(pattern)[1])

    record_path = os.path.join(base_dir, f'{prefix}{pattern}{count}')
    os.mkdir(record_path)

    return record_path


def clamp(value, min_value, max_value):
    """Clips the given [value] in the given interval [min_value, max_value]"""
    return max(min_value, min(value, max_value))


def cv2_grayscale(image: np.ndarray, is_bgr=True, depth=1):
    """Convert a RGB or BGR image to grayscale using OpenCV (cv2).
        :param image: input image, a numpy.ndarray.
        :param is_bgr: tells whether the image is in BGR format. If False, RGB format is assumed.
        :param depth: replicates the gray depth channel multiple times. E.g. useful to display grayscale images as rgb.
    """
    assert depth >= 1

    if image.dtype != np.uint8:
        image = np.rint(image)
        image = image.astype(dtype=np.uint8)

    if is_bgr:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if depth > 1:
        # depth concatenation (stack creates a new axis)
        return np.stack((grayscale,) * depth, axis=-1)
    else:
        # cv2 drops the channel axis in grayscale images
        return np.reshape(grayscale, newshape=grayscale.shape + (1,))


def replace_nans(data: dict, nan=0.0, pos_inf=0.0, neg_inf=0.0):
    """In-place replacement of non-numerical values, i.e. NaNs and +/- infinity"""
    for key, value in data.items():
        if np.isnan(value).any() or np.isinf(value).any():
            data[key] = np.nan_to_num(value, nan=nan, posinf=pos_inf, neginf=neg_inf)

    return data


def all_instances_of(iterable: list, kind: type) -> bool:
    """Returns true is all elements of the given list are instances of [kind]"""
    return all(isinstance(x, kind) for x in iterable)


# -------------------------------------------------------------------------------------------------
# -- Math
# -------------------------------------------------------------------------------------------------

def magnitude(vec3d: Union[carla.Vector3D, Tuple[float, float, float], List[float]]) -> float:
    """Returns the magnitude (norm) of the given 3D vector (tuple/list or carla.Vector3D)."""
    if isinstance(vec3d, tuple) or isinstance(vec3d, list):
        assert len(vec3d) == 3
        return math.sqrt(vec3d[0]**2 + vec3d[1]**2 + vec3d[2]**2)

    elif isinstance(vec3d, carla.Vector3D):
        return math.sqrt(vec3d.x**2 + vec3d.y**2 + vec3d.z**2)
    else:
        raise TypeError(f"Type for argument 'vec3d' must be 'list', 'tuple' or 'carla.Vector3D', not {type(vec3d)}.")


def sign(number: float) -> float:
    """Returns the sign (+1, -1) of the given number."""
    if number == 0.0:
        return +1.0

    return abs(number) / number


def clip_bound(value: float, v_max: float, v_min: float):
    return max(v_min, min(v_max, value))
