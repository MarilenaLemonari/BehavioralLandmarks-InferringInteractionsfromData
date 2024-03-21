#IMPORTS:
from re import S
from turtle import mode, shape
import numpy as np
import pickle
import time
import os
from numpy.core.numerictypes import nbytes
import os.path
import random
from tqdm import tqdm
import math

# Execute .\.venv\Scripts\activate
# Go to cd path_UMANS\DataDrivenInteractionFields\InteractionFieldsUMANS\examples
# Execute current script

#--------------------------------------------------------------------------------------------------------------
#SCENARIO SETUP:
path='C:/PROJECTS/DataDrivenInteractionFields/InteractionFieldsUMANS/examples/'
pathIF=f"{path}interactionFields/"
pathA=f"{path}agents/"
pathO=f"{path}obstacles/"
pathP=f"{path}policies/"
#IF.XML
def build_IFxml(desired_name,Ns):
  from xml.dom import minidom
  import os 
  root = minidom.Document()
  xml = root.createElement('InteractionFields') 
  root.appendChild(xml)
  dictionary=['IF0_approach','IF3_hide','IF2_circlearound','IF4_avoid']
  for i in range(Ns):
    for j in range(len(dictionary)):
      sourceIF=f'source{i+1}'
      sourceIF=root.createElement('InteractionField') 
      sourceIF.setAttribute('id', f'{i*10+j}')
      sourceIF.setAttribute('isVelocity', 'true')
      sourceIF.setAttribute('weight', '0')
      sourceIF.setAttribute('action_time', '0')
      sourceIF.setAttribute('inactive_time', '0')
      xml.appendChild(sourceIF)
      basis_name=dictionary[j]
      basisIF=f'matrix/{basis_name}.txt'
      basisMatrix = root.createElement('Matrix')
      basisMatrix.setAttribute('file', basisIF)
      sourceIF.appendChild(basisMatrix)
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
#AGENT.XML
def build_AGENTxml(desired_name,positions,s_positions,mode):
  from xml.dom import minidom
  import os 
  if mode == "t":
    speed='0.8'
  else:
    speed='1.0'
  root = minidom.Document()
  xml = root.createElement('Agents') 
  root.appendChild(xml)
  N=positions.shape[0]
  Ns=len(s_positions)
  for i in range(N):
    sourceIF=f'source{i+1}'
    sourceIF=root.createElement('Agent')     
    sourceIF.setAttribute('group', '0')
    sourceIF.setAttribute('max_speed', speed)
    sourceIF.setAttribute('pref_speed', speed)
    if i==0:
      sourceIF.setAttribute('max_speed', '0.0')
      sourceIF.setAttribute('pref_speed', '0.0')
    sourceIF.setAttribute('rad', '0.3')
    xml.appendChild(sourceIF)
    sourcePos = root.createElement('pos')
    sourcePos.setAttribute('x', f'{positions[i,0]}')
    sourcePos.setAttribute('y', f'{positions[i,1]}')
    sourceIF.appendChild(sourcePos)
    sourceGoal = root.createElement('goal')
    sourceGoal.setAttribute('x', '0')
    sourceGoal.setAttribute('y', '0')
    sourceIF.appendChild(sourceGoal)
    sourcePolicy = root.createElement('Policy')
    sourcePolicy.setAttribute('id', '0')
    sourceIF.appendChild(sourcePolicy)
    if (i in s_positions)==True:
      for j in range(4):
          sourcePolicy = root.createElement('InteractionField')
          sourcePolicy.setAttribute('id', f'{j}')
          sourceIF.appendChild(sourcePolicy)
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
  return Ns
#SCENARIO.XML
def build_SCENARIOxml(desired_name,desired_files, end_time):
  from xml.dom import minidom
  import os 
  root = minidom.Document()
  xml = root.createElement('Simulation')
  xml.setAttribute('delta_time','0.1')
  xml.setAttribute('end_time',end_time) # Difference!
  root.appendChild(xml)
  world=root.createElement('World') 
  world.setAttribute('type', 'Infinite')
  xml.appendChild(world)
  obs=root.createElement('Obstacles') 
  obs.setAttribute('file', f'{pathO}UserStudy1_obstacle.xml')
  world.appendChild(obs)
  agents=root.createElement('Agents') 
  agents.setAttribute('file', f'{pathA}Agent_{desired_files}.xml')
  xml.appendChild(agents)
  policies=root.createElement('Policies') 
  policies.setAttribute('file', f'{pathP}InteractionFieldsWithCollisionAvoidance.xml')
  xml.appendChild(policies)
  ifs=root.createElement('InteractionFields') 
  ifs.setAttribute('file', f'{pathIF}InteractionField_{desired_files}.xml')
  xml.appendChild(ifs)  
  xml_str = root.toprettyxml(indent ="\t") 
  save_path_file = desired_name 
  with open(save_path_file, "w") as f:
      f.write(xml_str)
def build_xml(mode,n,init_positions,end_time):
  if(mode=="s"):
    build_IFxml(f"{pathIF}InteractionField_s.xml",1)
    Ns=build_AGENTxml(f"{pathA}Agent_s.xml",init_positions,[0],mode)
    build_SCENARIOxml(f"{path}Scenario_s.xml","s",'11.1')
  elif(mode=="t"):
    build_IFxml(f"{pathIF}InteractionField_t.xml",1)
    Ns=build_AGENTxml(f"{pathA}Agent_t.xml",init_positions,[0],mode)
    build_SCENARIOxml(f"{path}Scenario_t.xml","t",'11.1')
  else:
    build_IFxml(f"{pathIF}InteractionField_no.xml",1)
    Ns=build_AGENTxml(f"{pathA}Agent_no.xml",init_positions,[0],mode)
    build_SCENARIOxml(f"{path}Scenario_no.xml","no",end_time)

def update_gtIFxml(IFxml_file,W,actionTimes,inactiveTimes):
  Ns=W.shape[0]
  import xml.etree.ElementTree as ET
  tree = ET.parse(IFxml_file)
  root=tree.getroot()
  for source in range(Ns):
    for basis_i in range(4):
      weight=W[source,basis_i]
      element = root[source+basis_i]
      element.set('weight', str(weight))
      actionTime=actionTimes[source,basis_i]
      element.set('action_time', str(actionTime))
      inactiveTime=inactiveTimes[source,basis_i]
      element.set('inactive_time', str(inactiveTime))
      tree.write(IFxml_file, encoding = "UTF-8", xml_declaration = True)
#--------------------------------------------------------------------------------------------------------------
#Make S_true:
def load_trajectories(file_name, resume_time):
  file = open(file_name)
  numpy_array = np.loadtxt(file, delimiter=",")
  numpy_array=numpy_array[resume_time:,:]
  return numpy_array
def make_trajectory(n,n_agents,mode,category):
  if category == "Training":
    if mode == "t":
      folder = "TemporalLandmarks"
    elif mode == "s":
      folder="SpatialLandmarks"
    else:
      folder="NoLandmarksExt"
  else:
      if mode == "t":
        folder = "TemporalLandmarks/TestData"
      elif mode == "s":
        folder="SpatialLandmarks/TestData"
      else:
        folder="NoLandmarksExt/TestData"

  for i in range(1,n_agents):
    file=f"C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\Trajectories\{folder}\{n}_a{i}.csv"
    file2=f"C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\Trajectories\{folder}\output_{mode}{i}.csv"
    if os.path.exists(file)==True:
      os.remove(file)
    if os.path.exists(file2)==True:
      os.remove(file2)
  os.getcwd()
  os.system(f"C:\\PROJECTS\\DataDrivenInteractionFields\\InteractionFieldsUMANS\\build\\Release\\UMANS-ConsoleApplication-Windows.exe -i Scenario_{mode}.xml -o C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\Trajectories\{folder}")
  S_true=[]
  for i in range(1,n_agents):
    os.rename(f"C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\Trajectories\{folder}\output_{i}.csv",f"C:\PROJECTS\BehavioralLandmarks\BehavioralLandmarks_Python\Data\Trajectories\{folder}\{n}_a{i}.csv")
    S_true.append(load_trajectories(f"C:/PROJECTS/BehavioralLandmarks/BehavioralLandmarks_Python/Data/Trajectories/{folder}/{n}_a{i}.csv", 0))
  return S_true
#---------------------------------------------------------------------------------------------------------------

def generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,mode,category,end_time):
  build_xml(mode,n,init_positions,end_time)
  n_agents=init_positions.shape[0]
  update_gtIFxml(f"{pathIF}InteractionField_{mode}.xml",weight,actionTimes,inactiveTimes)
  S_true=make_trajectory(n,n_agents,mode,category)

if __name__ ==  '__main__':
  category = "Testing" # TODO
  if category == "Training":
    repeat = 1000
    prefix = '_IF_'
  elif category == "Testing":
    repeat = 100
    prefix = '_test_IF_'
  mode = "no" 
  counter = 0 #TODO: change mode
  if mode == "t":
    for r in tqdm(range(repeat)):
      x0 = 0
      y0 = 0
      radius = 9
      angle = random.uniform(0, 2 * math.pi)
      x = x0 + radius * math.cos(angle)
      y = y0 + radius * math.sin(angle)
      init_positions=np.array([[x0,y0],[x,y]])
      # CHANGE ACTIVE INTERACTION FIELDS:
      weight=np.zeros((1,4))
      IF1,IF2=random.sample(range(4), 2)
      weight[0,IF1]=1
      weight[0,IF2]=1
      # CHANGE ACTIVE DURATION:
      T=random.randint(1,9)
      actionTimes=np.ones((1,4))*(-1)
      inactiveTimes=np.ones((1,4))*(-1)
      actionTimes[0,IF1]=0
      actionTimes[0,IF2]=T
      inactiveTimes[0,IF1]=T
      inactiveTimes[0,IF2]=10
      
      n=str(counter)+prefix+str(IF1)+str(IF2)+'_T'+str(T)
      counter += 1

      generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,mode,category,end_time='11.1')
  elif mode == "s":
    for r in tqdm(range(repeat)):
      x0 = 0
      y0 = 0
      radius = 9
      angle = random.uniform(0, 2 * math.pi)
      x = x0 + radius * math.cos(angle)
      y = y0 + radius * math.sin(angle)
      init_positions=np.array([[x0,y0],[x,y]])
      # CHANGE ACTIVE INTERACTION FIELDS:
      weight=np.zeros((1,4))
      IF1,IF2=random.sample(range(4), 2)
      # CHANGE BLENIDN VALUE: FIX BLENDING VALUE
      b=random.randint(1,5)
      weight[0,IF1]=1
      weight[0,IF2]=b
      T=10
      actionTimes=np.ones((1,4))*(-1)
      inactiveTimes=np.ones((1,4))*(-1)
      actionTimes[0,IF1]=0
      actionTimes[0,IF2]=0
      inactiveTimes[0,IF1]=T
      inactiveTimes[0,IF2]=T
      
      n=str(counter)+prefix+str(IF1)+str(IF2)+'_b'+str(b)
      counter += 1

      generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,mode,category,end_time='11.1')
  elif mode == "no":
    for r in tqdm(range(repeat)):

      weight=np.zeros((1,4))
      IF,=random.sample(range(4), 1)
      weight[0,IF]=1
      actionTimes=np.ones((1,4))*(-1)
      inactiveTimes=np.ones((1,4))*(-1)
      actionTimes[0,IF]=0
      end_time, = random.sample(range(5,20),1)
      inactiveTimes[0,IF]=end_time

      x0 = 0
      y0 = 0
      radius = 9
      angle = random.uniform(0, 2 * math.pi)
      x = x0 + radius * math.cos(angle)
      y = y0 + radius * math.sin(angle)
      init_positions=np.array([[x0,y0],[x,y]])
      
      n=str(counter)+prefix+str(IF)+"_dur"+str(end_time)
      counter += 1 
      generate_instance(n,init_positions,weight,actionTimes,inactiveTimes,mode,category,str(end_time))
  else:
      print("ERROR! Wrong mode")