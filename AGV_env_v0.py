import pybullet as p 
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time as t 
import utils


class AGV_env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 24 }
    def __init__(
        self,
        max_length      = 1000,
        num_step        = 24,
        render_mode     = None,
        robot_file      = 'AGV_env//urdf//AGV.urdf',
        target_file     = 'AGV_env//urdf//target.urdf',
        seed            = 0,
        buffer_length   = 1,
    ):
        super().__init__()
        
        
        # Saving initial arguments
        if isinstance(render_mode, str):
            self.clientId = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0, physicsClientId=self.clientId)
        else:
            self.clientId = p.connect(p.DIRECT)
        self.max_length     = max_length
        self.num_step       = num_step
        self.robot_file     = robot_file
        self.target_file    = target_file
        self.seed           = seed
        self.buffer_length  = buffer_length
        self.jointId_list   = []
        self.jointName_list = []
        
        
        # Random variables
        self.action_filter  = 0.8
        self.initialMass    = [0, 1.5]
        
 
        # Constant DO NOT TOUCH
        self.planeId    = 0
        self.robotId    = 1
        self.targetId   = 2
        self.mode       = p.VELOCITY_CONTROL
        self.max_vel    = 10 # m/s
        self.max_to     = 10 # Nm
        self.action_sp  = 2
        self.sleep_time = 1./240.
        self.initHeight = 0.1
        np.random.seed(self.seed)
        self.g          = (0,0,-9.81) 
        self.time_steps_in_current_episode = [0]
        
        
        # Archives variables
        self.world_pos          = np.zeros((1,3))
        self.world_ori          = np.zeros((1,3))
        self.local_ori          = np.zeros((1,2))
        self.local_lin_vel      = np.zeros((1,2))
        self.target_dir_world   = np.zeros((1,3))
        self.target_dir_robot   = np.zeros((1,2))

        
        # Setup the env
        print('-'*60)
        print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.clientId)
        p.setGravity(*self.g, physicsClientId=self.clientId)
        self.planeId = p.loadURDF('plane.urdf',physicsClientId = self.clientId)
        self.robotId = p.loadURDF(self.robot_file,basePosition=[0,0,self.initHeight],baseOrientation=[0,0,0,1],physicsClientId = self.clientId)
        self.targetId= p.loadURDF(self.target_file,basePosition=[0,0,0],baseOrientation=[0,0,0,1],physicsClientId = self.clientId)
        ###
        self.sample_scene()
        self.sample_target()
        ###
        self.number_of_joints = p.getNumJoints(self.robotId,physicsClientId = self.clientId)
        for joint_index in range(self.number_of_joints):
            data = p.getJointInfo(self.robotId, joint_index,physicsClientId = self.clientId)
            self.jointId_list.append(data[0])
            self.jointName_list.append(str(data[1]))
            print(f'Id: {data[0]}, Name: {str(data[1])}')
        self.robotBaseMassandFriction = [p.getDynamicsInfo(self.robotId,-1,physicsClientId = self.clientId)[0], p.getDynamicsInfo(self.robotId,self.jointId_list[-1],physicsClientId = self.clientId)[1]]
        print(f'Robot mass: {self.robotBaseMassandFriction[0]} and friction on feet: {self.robotBaseMassandFriction[1]}')
        p.setJointMotorControlArray(self.robotId,self.jointId_list,self.mode, physicsClientId = self.clientId)
        for joints in self.jointId_list:
            p.enableJointForceTorqueSensor(self.robotId,joints,True,physicsClientId = self.clientId)
        self.previous_act       = np.zeros((1,self.action_sp))
        self.action_space_      = self.action_sp
        self.observation_space_ = len(self.get_all_obs())
        self.action_space       = spaces.Box(low = -1, high = 1, shape = (self.action_space_,),dtype = np.float32)
        self.observation_space  = spaces.Box(low = -3.4e+38, high = 3.4e+38, shape = (self.observation_space_,), dtype = np.float32)
        print(f'Action space:       {self.action_space}')
        print(f'Observation space:  {self.observation_space}')
        print('-'*60)
    
    
    def stopper(self,time):
        # Stopping function
        start   = t.time()
        stop    = start
        while(stop<(start+time)):
            stop = t.time()
       
        
    def sample_scene(self):
        # Sample the scene of moving human in robot env
        return

    
    def sample_target(self):
        # Sample target direction of robot
        return
    
    
    def get_all_obs(self):
        # Get all observation from sensors
        return [0 for i in range(10)]
    
    
    def get_obs(self):
        # Get all observation, reward and termination info
        return
    
    
    def act(self,action):
        # Perform action in the environment
        reshape_action = np.hstack([action[0],action[0]])
        p.setJointMotorControlArray(self.robotId,
                                    self.jointId_list,
                                    self.mode,
                                    targetVelocities = reshape_action, 
                                    forces = np.ones_like(reshape_action)*self.max_to, 
                                    physicsClientId = self.clientId)
        
    
    def step(self,action,real_time=False):
        # Step the simulation
        
        ## Feed action to our env through a filter
        action *= self.max_vel
        filtered_action = self.previous_act*self.action_filter + action*(1-self.action_filter)
        self.previous_act[0] = action 
        
        ## Step the simulation and act numstep times
        self.time_steps_in_current_episode = [self.time_steps_in_current_episode[0]+1]
        for _ in range(self.num_step):
            self.act(filtered_action)
            p.stepSimulation(physicsClientId=self.clientId)
            p.resetBasePositionAndOrientation(self.targetId,self.target_dir_world[0], [0,0,0,1], physicsClientId = self.clientId)
            if real_time:
                self.stopper(self.sleep_time)
        return self.get_obs()
    
    
env = AGV_env(render_mode='human')
while True:
    env.step(10*np.random.rand(2,),real_time=True)
