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
        debug           = False,
        robot_file      = 'AGV_env//urdf//AGV.urdf',
        target_file     = 'AGV_env//urdf//target.urdf',
        room_file       = 'AGV_env//urdf//room.urdf',
        seed            = 0,
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
        self.render_mode    = render_mode
        self.debug          = debug
        self.robot_file     = robot_file
        self.target_file    = target_file
        self.room_file      = room_file
        self.seed           = seed
        self.jointId_list   = []
        self.jointName_list = []
        self.rayId_list = []
        self.vizId_list = []
        
        
        # Random variables
        self.action_filter  = 0.8
        self.initialMass    = [0, 1.5]
        
 
        # Constant DO NOT TOUCH
        self.planeId    = 0
        self.robotId    = 1
        self.targetId   = 2
        self.mode       = p.VELOCITY_CONTROL
        self.max_vel    = 10    # rad/s
        self.max_to     = 10    # Nm
        self.numray     = 500   # From Hshop C1M1 RPLIDAR
        self.raylen     = 12    # From Hshop C1M1 RPLIDAR
        self.action_sp  = 2
        self.sleep_time = 1./240.
        self.initHeight = 0.1
        self.rayHfromB  = 0.2
        np.random.seed(self.seed)
        self.g          = (0,0,-9.81) 
        self.time_steps_in_current_episode = [0]
        self.x          = np.array([i/self.numray for i in range(self.numray)])
        self.rayend     = np.vstack([self.raylen*np.cos(2*np.pi*self.x), self.raylen*np.sin(2*np.pi*self.x),np.zeros_like(self.x),np.ones_like(self.x)])
        
        
        # Archives variables
        self.world_pos          = np.zeros((1,3))
        self.world_ori          = np.zeros((1,4))
        self.local_ori          = np.zeros((1,2))
        self.local_lin_vel      = np.zeros((1,2))
        self.ray_info           = [[]]          
        self.target_dir_world   = np.zeros((1,3))
        self.target_dir_robot   = np.zeros((1,2))

        
        # Setup the env
        print('-'*60)
        print(f'ENVIRONMENT STARTED WITH SEED {self.seed}')
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.clientId)
        p.setGravity(*self.g, physicsClientId=self.clientId)
        self.planeId = p.loadURDF(self.room_file,basePosition=[0,0,0],baseOrientation=[0,0,0,1],globalScaling=1,useFixedBase=True,physicsClientId = self.clientId)
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
        if self.debug:
            for i in range(self.numray):
                self.rayId_list += [p.addUserDebugLine([0,0,0], [0,0,0], [0,1,0], physicsClientId=self.clientId)]
            for _ in range(1):
                self.vizId_list += [p.addUserDebugLine([0,0,0], [0,0,0], [0,0,0], physicsClientId=self.clientId)]
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
        ## Reset robot
        p.resetBasePositionAndOrientation(self.robotId,[0.,0.,self.initHeight],[0,0,0,1])
        ## Reset target
        new_direction   = np.random.normal(0,5,2)
        new_direction   = np.hstack([30*new_direction/np.linalg.norm(new_direction),np.array([self.initHeight])])
        p.resetBasePositionAndOrientation(self.targetId, new_direction, [0,0,0,1], physicsClientId = self.clientId)
        self.target_dir_world[0] = new_direction
        return
    
    
    def get_velocity(self):
        # Get the velocity from robot local frame
        temp_obs_value          = []
        fac                     = np.array([1,0,0,0])
        base_pos, base_ori      = p.getBasePositionAndOrientation(self.robotId,physicsClientId=self.clientId)[:2]
        local_facing_dir        = utils.passive_rotation(np.array(base_ori),fac)[:2]
        self.world_pos[0,:]     = np.array(base_pos)
        self.world_ori[0,:]     = np.array(base_ori)     
        self.local_ori[0,:]     = np.array(local_facing_dir)
        linear_vel, _           = p.getBaseVelocity(self.robotId,physicsClientId=self.clientId)
        linear_vel              = np.array(list(linear_vel)+[1])
        quaternion1, quaternion2= np.array(base_ori),linear_vel
        self.local_lin_vel[0,:] = utils.quaternion_multiply(utils.quaternion_multiply(utils.quaternion_inverse(quaternion1),quaternion2),quaternion1)[:2]
        temp_obs_value         += [*self.local_lin_vel[0]]
        # Check weather global velocity equal local velocity
        # print(np.abs(np.linalg.norm(self.local_lin_vel)-np.linalg.norm(linear_vel[:2]))<1e-4)
        return temp_obs_value
    
    
    def get_lidar(self):
        # Get lidar sensor signals
        temp_obs_value      = []
        base_pos, base_ori  = self.world_pos[0], self.world_ori[0]
        start               = np.array([[base_pos[0],base_pos[1],self.rayHfromB] for i in range(self.numray)])
        quat1, quat2        = base_ori,self.rayend
        end                 = (utils.quaternion_multiply(utils.quaternion_multiply(quat1,quat2),utils.quaternion_inverse(quat1))[:3,:]).T+start
        ray_contact         = []
        ray_tio             = []
        ray_hit             = []
        for info in p.rayTestBatch(start,end,numThreads=0,physicsClientId =self.clientId):
            ray_contact    += [info[3]]
            ray_tio        += [info[2]]
            ray_hit        += [info[0]]
        self.ray_info[0]    = [start,end,np.array(ray_tio),np.array(ray_hit)]
        temp_obs_value     += [*ray_tio]
        return temp_obs_value
    
    
    def get_target_dir(self):
        # Get target direction vector
        temp_obs_value = []
        base_pos, base_ori = self.world_pos[0], self.world_ori[0]
        target_dir = self.target_dir_world[0] - base_pos
        target_dir = np.array(list(target_dir)+[0])
        qua1, qua2 = base_ori, target_dir
        target_dir = utils.quaternion_multiply(utils.quaternion_multiply(utils.quaternion_inverse(qua1),qua2),qua1)[:2]
        self.target_dir_robot[0] = target_dir
        temp_obs_value += [*(target_dir/np.linalg.norm(target_dir))]
        return temp_obs_value
    
    
    def get_all_obs(self):
        # Get all observation from sensors
        temp_obs_value  = []
        ## Target dir
        target_info     = self.get_target_dir()
        ## Velocity vector
        velo_info       = self.get_velocity()
        ## RPLIDAR 
        ray_info        = self.get_lidar()
        
        temp_obs_value += [ *target_info,
                            *velo_info,
                            *ray_info,
                          ]
        return temp_obs_value
    
    
    def get_obs(self):
        # Get all observation, reward and termination info
        ## GET OBSERVATION
        obs = np.array(self.get_all_obs()).astype('float32')
        ## GET REWARD
        rew = self.calculate_reward()
        ## GET INFO
        ray_inf = obs[-self.numray:]
        terminated = np.sum(ray_inf < 0.01*self.raylen) > 0
        truncated  = self.time_steps_in_current_episode[0]> self.max_length
        info = None
        return obs, rew, bool(terminated), bool(truncated), info
    
    
    def viz_dir(self):
        # Visualize velocity vector
        base_pos = self.world_pos[0]
        base_ori = np.array([*self.local_ori[0],0])
        p.addUserDebugLine(base_pos,base_pos+base_ori,lineWidth = 2, lifeTime =.5, lineColorRGB = [1,0,0],replaceItemUniqueId=self.vizId_list[0],physicsClientId = self.clientId)
        return
    
    
    def viz_ray(self):
        # Visualize raydetect from lidar
        start, end, ratio, hit = self.ray_info[0]
        for i,Id in enumerate(self.rayId_list):
            if hit[i] != -1:
                p.addUserDebugLine(start[i],start[i]+(ratio[i]*(end[i]-start[i])),[0,1,0],replaceItemUniqueId=Id,physicsClientId = self.clientId)
            else:
                p.addUserDebugLine([0,0,0],[0,0,0],[1,0,0],replaceItemUniqueId=Id,physicsClientId = self.clientId)
                # p.addUserDebugLine(start[i],end[i],[1,0,0],replaceItemUniqueId=Id,physicsClientId = self.clientId)
        return
    
    def viz(self):
        # Visualize all component
        self.viz_dir()
        self.viz_ray()
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
    
    
    def calculate_reward(self):
        # Calculate reward value
        return None
    
    
    def reset(self):
        self.sample_target()
        self.sample_scene()
        self.time_steps_in_current_episode[0] = 0
        self.previous_act[0]                  = np.zeros((self.action_sp))
        info = None
        return np.array(self.get_all_obs()).astype('float32'), info
    
    
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
            if self.debug:
                self.viz()
        return self.get_obs()
  
    
if __name__ == '__main__':
    env = AGV_env(render_mode='human',debug=True,num_step=1)
    p.loadURDF('cube.urdf',[-2,2,0.5])
    p.loadURDF('cube.urdf',[2,2,0.5])
    p.loadURDF('cube.urdf',[2,-2,0.5])
    p.loadURDF('cube.urdf',[-2,-2,0.5])
    while True:
        obs, reward, terminated, truncated, info = env.step(np.array([3,3]),real_time=True)
        if terminated or truncated:
            print(terminated,truncated)
            obs, inf = env.reset()