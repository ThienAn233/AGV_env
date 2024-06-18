import pybullet as p
import pybullet_data
import numpy as np
import time as t


PATH = 'AGV_env\\urdf\\AGV.urdf'
initial_height = 1
initial_ori = [0,0,0,1]
jointId_list = []
jointName_list = []
debugId_list = []
temp_debug_value = []
mode = p.VELOCITY_CONTROL
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setRealTimeSimulation(True)


# Constants
fixed = False
g = (0,0,-9.81) 
pi = np.pi
max_speed = 10 # m/s
max_torque= 10 # Nm


# Setup the environment
print('-'*100)
p.setGravity(*g)
robotId = p.loadURDF(PATH,[0.,0.,initial_height],initial_ori)
planeId = p.loadURDF('plane.urdf')
number_of_joints = p.getNumJoints(robotId)
print(f'Robot id: {robotId}')
print(f'number of robot joints: {number_of_joints}')
for joint_index in range(number_of_joints):
    data = p.getJointInfo(robotId, joint_index)
    jointId_list.append(data[0])                                                                                # Create list to store joint's Id
    jointName_list.append(str(data[1]))                                                                         # Create list to store joint's Name
    debugId_list.append(p.addUserDebugParameter(str(data[1]), rangeMin = -max_speed, rangeMax = max_speed, ))   # Add debug parameters to manually control joints
    p.enableJointForceTorqueSensor(robotId,joint_index,True)
    print(f'Id: {data[0]}, Name: {str(data[1])}, DebugId: {debugId_list[-1]}')
p.setJointMotorControlArray(robotId,jointId_list,mode)
print(f'Control mode is set to: {"Velocity" if mode==0 else "Position"}')
print('-'*100)


# Simulation loop
while True:
    p.stepSimulation()
    temp_debug_value = []
    for Id in debugId_list:
        temp_debug_value.append(p.readUserDebugParameter(Id))
    action = np.array(temp_debug_value)
    p.setJointMotorControlArray(robotId,
                                jointId_list,
                                mode,
                                targetVelocities = action,
                                forces = np.ones_like(action)*max_torque,        
                                )
    if fixed:
        p.resetBasePositionAndOrientation(robotId,[0.,0.,initial_height],initial_ori)
    # base_inf = p.getBaseVelocity(robotId)[1]
    # print(base_inf)