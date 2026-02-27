from sys import prefix

import mujoco
from mujoco import MjSpec
import os
from ml_collections import config_dict
from copy import deepcopy
import struct
import numpy as np
import yaml
from multiprocessing import Queue
from collections import deque

dirname = os.path.dirname(__file__)
DEFAULT_ENV_PATH = os.path.join(dirname, "./MPL/include_MPL.xml")



def default_config():
  return config_dict.create(
    shorten_thumb=True,
    file_path=DEFAULT_ENV_PATH,
    n_substeps=10,
    iterations=10,
    camera_transforms=({"pos":(0., -0.3, 1), "zaxis":(0, 0, 1), "fovy":45, "ipd":0.068},),
    table_offset = [0, -0.3, 0],
    hand_color = (0.3, 0.3, 0.3, 1),
    table_scale = (2, 2, 1),
    add_ref_hand = True,
  )


class VisuMPL:
  def __init__(self, config=default_config()):
    if config is str:
      self.config = config_dict.config_dict
    self.config = config
    self.spec: MjSpec = self.preprocess_spec(MjSpec.from_file(config.file_path))
    self.mj_model: mujoco.MjModel = self.spec.compile()
    self.mj_data = mujoco.MjData(self.mj_model)
    self.ctrl_queue: Queue = Queue(maxsize=1)
    self.v = None
    self.key_presses = deque(maxlen=20)
    pass

  def set_mocap(self, pos, quat):
    self.mj_data.mocap_pos[0] = pos
    self.mj_data.mocap_quat[1] = quat

  def step(self, actions):
    self.mj_data.ctrl = actions
    mujoco.mj_step(self.mj_model, self.mj_data, self.config.n_substeps)

  def set_ref_color(self, rgba):
    for g in [_g for _g in self.spec.geoms if "ref_" in _g.parent.name]:
      self.mj_model.geom_rgba[g.id] = rgba

  def preprocess_spec(self, spec: MjSpec):
    if self.config.shorten_thumb:
      spec_copy = self._clear_assets(spec.copy(), ["A_thumb_DIP"])
      spec.delete(spec.body("thumb2"))
      fr = spec.body("thumb1").add_frame()
      fr.attach_body(spec_copy.body("thumb3"))
    spec.option.iterations = self.config.iterations
    for g in spec.geoms[:2]:
      g.size *= np.array(self.config.table_scale)
    for c in self.config.camera_transforms:
      spec.worldbody.add_camera(**c)
    for g in spec.worldbody.geoms:
      if not np.isnan(g.fromto[0]):
        g.fromto += np.array(self.config.table_offset*2)
      else:
        g.pos += np.array(self.config.table_offset)
    spec.material('MatMesh').rgba = self.config.hand_color
    for t in spec.tendons:
      t.group = 4

    if self.config.add_ref_hand:
      ref_spec = spec.copy()
      for g in ref_spec.geoms:
        g.group = 4
        g.contype = 0
        g.conaffinity = 0


      ref_hand = ref_spec.body("forearm")
      ref_spec.material('MatMesh').rgba = (0.1, 0.7, 0.2, 0.2)
      fr:mujoco.MjsFrame = spec.body("forearm").parent.add_frame()
      fr.attach_body(ref_hand,prefix="ref_")
      mocap_spec = self._clear_assets(spec.copy())
      fr.attach_body(mocap_spec.body("mocap"), prefix="ref_")
      spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                  name1='ref_mocap', name2='ref_forearm', solref=(.0001, 1.4), solimp=(.99,.99,0.0001, 0.5, 2))


    return spec

  @staticmethod
  def _clear_assets(spec, excluded_names=()):
    [[spec.delete(e) for e in typ if e.name not in excluded_names] for typ in
     [spec.meshes, spec.textures, spec.materials, spec.tendons,
      spec.equalities, spec.actuators, spec.sensors]]
    return spec

  def start_visu(self):
    from mujoco import viewer
    from loop_rate_limiters import RateLimiter

    def key_callback(keycode):
      self.key_presses.append(chr(keycode))

    rate = RateLimiter(frequency=1 / self.mj_model.opt.timestep // self.config.n_substeps, warn=False)
    with viewer.launch_passive(self.mj_model, self.mj_data, show_left_ui=False, show_right_ui=False, key_callback=key_callback) as self.v:
      # setup v.user_scn here https://mujoco.readthedocs.io/en/latest/python.html#passive-viewer
      self.v.sync()
      most_recent_control = np.zeros(self.mj_model.nu)
      while self.v.is_running():
        self.v.sync()
        while not self.ctrl_queue.empty():
          most_recent_control =  self.ctrl_queue.get()
        self.step(most_recent_control)
        rate.sleep()


def load_config_from_yaml(file_path: str) -> config_dict.ConfigDict:
  with open(file_path, 'r') as f:
    config_dict = yaml.safe_load(f)
  return config_dict.ConfigDict(config_dict)

if __name__ == '__main__':
  visu = VisuMPL()
  visu.set_ref_color(np.array([0.8, 0.3, 0.1, 0.3]))
  from mujoco import viewer
  from loop_rate_limiters import RateLimiter

  rate = RateLimiter(frequency=1 / visu.mj_model.opt.timestep // visu.config.n_substeps, warn=True)
  with viewer.launch_passive(visu.mj_model, visu.mj_data) as v:
    # setup v.user_scn here https://mujoco.readthedocs.io/en/latest/python.html#passive-viewer
    v.sync()
    while v.is_running():
      v.sync()
      visu.step(np.ones(visu.mj_model.nu)*np.sin(visu.mj_data.time))
      # Example display text
      v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_50, mujoco.mjtGridPos.mjGRID_TOPLEFT, f'{visu.mj_data.time}', '')])
      rate.sleep()

  def dummy_ctrl():
    while True:
      visu.ctrl_queue.put(np.sin(visu.mj_data.time) * np.ones(visu.mj_model.nu))



  import threading
  t = threading.Thread(target=visu.start_visu)
  t2 = threading.Thread(target=dummy_ctrl)
  t2.daemon = True
  t.daemon = True
  t.start()
  t2.start()
  t.join()
  t2.join()
