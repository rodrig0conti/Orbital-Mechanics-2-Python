import os
import glob
import subprocess
from vispy import app, scene
from vispy.scene.visuals import Mesh
from vispy.scene import MatrixTransform as Mat4
from vispy.scene import STTransform as STT 
from vispy.util.quaternion import Quaternion as Quat
from vispy.io import imread, read_mesh
from vispy.visuals.filters import TextureFilter
import numpy as np

import threading
import queue
import time

import simutils as su
import orbit_lib as ol 
import sat_lib as sl

class SimCanvas(scene.SceneCanvas):
    def __init__(self,anim_queue,anim_dt,anim_close,scale_factor,scene_conf=None):
        _default_conf = {'satellite_model':'3DModels/satellite.obj','earth_model':'3DModels/earth.obj','earth_texture':'3DModels/earth.jpg'}
        scene.SceneCanvas.__init__(self,title='STE-3605 Simulator',keys='interactive',size=(800,600))
        self.unfreeze()
        self.anim_queue = anim_queue
        self.anim_dt = anim_dt
        self.anim_close = anim_close
        self.scale_factor = scale_factor
        self.view = self.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.distance = 30 
        self.light_dir = (-100, 0, 0, 0)
        self.scene_list = {}
        self.follow = 'earth'
        if scene_conf is None:
            self.make_default_scene(_default_conf)

        self._timer = app.Timer(self.anim_dt,connect=self.on_timer,start=True,app=self.app)
        self.show()
        self.freeze()

    def make_default_scene(self,conf):
        self.make_default_satellite(conf['satellite_model'])
        self.make_default_earth(conf['earth_model'],conf['earth_texture'])
        self.make_default_frames()
        for k in self.scene_list.keys():
            self.scene_list[k][0].transform = Mat4()
        

    def make_default_satellite(self,satellite_model):
        self.vertices, self.vertex_cols, self.faces = su.read_obj(satellite_model)
        if len(self.vertex_cols) > 0:
            self.satellite = Mesh(self.vertices, self.faces, vertex_colors = self.vertex_cols, shading='flat', color='white',parent=self.view.scene)
        else:
            self.satellite = Mesh(self.vertices, self.faces, shading='flat', color='white',parent=self.view.scene)
        self.satellite.shading_filter.light_dir = self.light_dir[:3]
        s = ol.R_E/(40*self.scale_factor*np.max(np.abs(self.vertices)))
        self.scene_list['satellite'] = (self.satellite,s)

    def make_default_earth(self,earth_model,earth_texture):
        self.earth_verts, self.earth_faces, self.earth_normals, self.earth_texcoords = read_mesh(earth_model)
        self.earth = Mesh(self.earth_verts, self.earth_faces, shading='smooth', color='white',parent=self.view.scene)
        self.earth.shading_filter.light_dir = self.light_dir[:3]
        self.earth_texture = np.flipud(imread(earth_texture))
        self.texture_filter = TextureFilter(self.earth_texture,self.earth_texcoords)
        self.earth.attach(self.texture_filter)
        s = ol.R_E/self.scale_factor
        self.scene_list['earth'] = (self.earth,s)

    def make_default_frames(self):
        self.body_frame = scene.visuals.XYZAxis(parent=self.view.scene)
        self.scene_list['body frame'] = (self.body_frame,self.scene_list['satellite'][1])
        self.ecef_frame = scene.visuals.XYZAxis(parent=self.view.scene)
        self.eci_frame = scene.visuals.XYZAxis(parent=self.view.scene)
        self.scene_list['ECI frame'] = (self.eci_frame,1.3*self.scene_list['earth'][1])
        self.scene_list['ECEF frame'] = (self.ecef_frame,1.3*self.scene_list['earth'][1])

    def update_scene(self,anim_data):
        for name,p,q in anim_data:
            if name in self.scene_list.keys():
                self.scene_list[name][0].transform = su.rotscaleloc_to_vispy(pos=p/self.scale_factor,quat=q,scale=self.scene_list[name][1])

    def on_timer(self,event):
        if not self.anim_queue.empty():
            anim_data = self.anim_queue.get_nowait()
            self.update_scene(anim_data)
            _,self.view.camera.center = su.H_to_Rp(self.scene_list[self.follow][0].transform)
            self.update()

    def on_close(self,event):
        self.anim_close.set()
        return super().on_close(event)

    def on_key_press(self,event):
        if event.text == 'h':
            keys = sorted(self.scene_list.keys())
            for i,k in enumerate(keys):
                if self.follow == k:
                    idx = (i+1)%len(keys)
                    self.follow = keys[idx]
                    break

class Simulator:
  def __init__(self,config,scenario,sim_queue,event):
    self.sim_thread = threading.Thread(target=self.sim_runner)
    self.sim_close = event
    self.sim_queue = sim_queue
    self.scenario = scenario
    self.t_0 = config['t_0']
    self.t_e = config['t_e']
    self.t_step = config['t_step']
    self.speed_factor = config['speed_factor']
    self.anim_dt = config['anim_dt']
    self.visualise = config['visualise']

  def start(self):
    self.sim_thread.start()

  def sim_runner(self):
    self.t = self.t_0
    self.t_anim = 0
    self.scenario.init(self.t_0)

    # Always send initial state
    states = self.scenario.get()
    self.sim_queue.put(states)

    while not self.sim_close.is_set() and self.t < self.t_e:

      # Always update physics
      self.scenario.update(self.t,self.t_step)

      # If visualising, throttle animation
      if self.visualise:
        if (self.t - self.t_anim) >= (self.t_step * self.speed_factor):

          self.t_anim = self.t
          states = self.scenario.get()
          self.sim_queue.put(states)
          time.sleep(self.anim_dt) #debería ser time.sleep(0.01) en caso de que se rompa o (self.anim_dt)

      else:
        # If NOT visualising, push every step
        states = self.scenario.get()
        self.sim_queue.put(states)

      self.t += self.t_step

    states = self.scenario.get()
    self.sim_queue.put(states)

    self.scenario.post_process(self.t,self.t_step)

  def wait(self):
    if self.sim_thread.is_alive():
      self.sim_thread.join()


class BaseScenario:
    def init(self,t):
        pass
    def update(self,t,dt):
        pass
    def get(self):
        return []
    def post_process(self,t,dt):
        pass

def create_and_start_simulation(sim_config, scenario):
    anim_queue = queue.SimpleQueue()
    anim_close = threading.Event()

    sim = Simulator(sim_config, scenario, anim_queue, anim_close)
    sim.start()

    if sim_config['visualise']:
        # Run the visual window (blocks until closed)
        canvas = SimCanvas(anim_queue, sim_config['anim_dt'], anim_close, sim_config['scale_factor'])
        app.run()

        # Do NOT wait for the simulation thread — it may still be running
        anim_close.set()   # Tell simulation to stop
    else:
        # Non-visual mode → wait normally
        sim.wait()

    # ---------------------------------------------------------
    # Automatically plot the latest data file (even if interrupted)
    # ---------------------------------------------------------
    data_folder = "data"
    list_of_files = glob.glob(os.path.join(data_folder, "*.txt"))

    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Plotting latest file: {latest_file}")

        subprocess.Popen(["python", "plotter.py", latest_file])
    else:
        print("No data files found to plot.")

