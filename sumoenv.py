# region 库导入
from gym import spaces, core
import traci
import numpy as np
# endregion

# region 常量
SUMOPATH:str = "D:/Sumo/bin/sumo-gui"
"""sumo 可执行文件路径"""
SIMFILEPATH:str = "sumofiles/run.sumocfg"
"""sumo仿真文件 (*.sumocfg) 路径"""
SIMTIME:int = 3600
"""仿真时间长度"""
DELTADUR:int = 20
DELTATIME:int = 10
"""决策间隔时间
>>> def SumoEnv.step():
        ...
        take_action()
        for i in range(DELTATIME):
            ...
            traci.simulationStep()
        ...
"""
MAXPHASETIME:float = 60.0
"""最长绿灯时间, default to 60.0"""
INPUTEDGE:list[str] = []
"""路网文件中交叉口的进口边 `edge`"""
INPUTLANE:list[str] = [] 
"""路网文件中交叉口的进口车道 `lane`"""
OUTPUTLANE:list[str] = []
"""路网文件中交叉口对应进口车道的出口车道 `lane`"""
LANEAREALENGTH:list[float] = []
"""E2 检测器的长度"""
FREEFLOWSPEED:float = 11.11 # 期望速度
"""期望速度, default to `11.11m/s` (40km/h)"""
DETECTORS:list[str] = []
"""E2 检测器列表"""
INT:str = "0"
"""交叉口ID, default to '0'"""
VEHCILELENGTH:float = 5.0
"""车辆长度, default to 5.0"""
# endregion

# region 临时变量
# 用于 edge, lane, e2 detector 的排序
# 逆时针排序
edge_idx = ["n","e","s","w"]
lane_idx = ["r","s","l"]
# endregion


def initconstants()->None:
    """初始化全局变量
    """
    clearconstants()
    global INT 
    INT = traci.trafficlight.getIDList()[0]
    detectors:tuple = traci.lanearea.getIDList()
    detectors = sorted(detectors, key=lambda x: (edge_idx.index(x[3]),lane_idx.index(x[-1])))
    for detector in detectors:
        if "l" in detector:
            detectors.remove(detector)

    for detector in detectors:
        DETECTORS.append(detector)
        LANEAREALENGTH.append(traci.lanearea.getLength(detector))
        INPUTLANE.append(traci.lanearea.getLaneID(detector))
    
    edges = []
    for lane in INPUTLANE:
        edges.append(traci.lane.getEdgeID(lane))
        edges = sorted(list(set(edges)),key=lambda x: edge_idx.index(x[0]))
        OUTPUTLANE.append(traci.lane.getLinks(lane)[0][0])
    
    for edge in edges:
        INPUTEDGE.append(edge)
        global VEHCILELENGTH 
        VEHCILELENGTH=  traci.vehicletype.getLength("DEFAULT_VEHTYPE")
    pass

def clearconstants()->None:
    """全局变量置空
    """
    global INPUTEDGE 
    INPUTEDGE = []
    global INPUTLANE
    INPUTLANE = [] 
    global OUTPUTLANE 
    OUTPUTLANE = []
    global LANEAREALENGTH
    LANEAREALENGTH = []
    global DETECTORS 
    DETECTORS = []
    global INT 
    INT = "0"

    pass
class SumoEnv(core.Env):
    """`sumo` 仿真环境

    Args:
        core (gym.core): 继承 gym.core , 可使用 gym 的 api
    """
    def __init__(self):

        self.action_space = spaces.Discrete(4) # 要切换到的相位
        """动作空间 `(相位索引, 相位持续时间)`\n
            注 : \n
            相位索引仅包括绿灯相位索引, [0,3,6,9]\n
        """
        self.observation_space =  spaces.Dict({"meanspeed":spaces.Box(0,11.11),
                                               "veh_occ":spaces.Box(0,1.0),
                                               "halt_occ":spaces.Box(0,1.0),
                                               "in_occ":spaces.Box(0,1.0),
                                               "out_occ":spaces.Box(0,1.0),
                                               "phase":spaces.Discrete(12),
                                               "sus":spaces.Discrete(60),
                                               "re":spaces.Discrete(60)})
        """状态空间"""
        self.detectors:list[str] = []
        """
        E2检测器列表, 初始化为空, reset()获取
        """
        self.signalstate = SignalState()
        """
        记录当前仿真时间刻 `self.timestep` 的信号灯状态
        >>> state_index:int = self.signalstate.state # 当前的相位索引
        begin_timestep:int = self.begin_time # 当前相位的开始仿真时间刻
        """
        self.timestep = 0
        """当前仿真时间刻
        """
        self.ob_data = {}

        pass
    
    def reset(self)->dict:
        """重置 `sumo` 仿真, 并返回初始状态

        Returns:
            dict: 初始状态
        """
        try:
            traci.close()
        except traci.FatalTraCIError as err:
            pass
        finally:
            traci.start([SUMOPATH,"-c",SIMFILEPATH]) # 开始一次仿真
            initconstants()
            self.detectors:list = DETECTORS 
            self.timestep = 0
            self.signalstate.reset()
            obs = self.get_observation()
            return obs
    
    def step(self, state_idx:int)->tuple[dict, float, bool, bool, dict]:
        """将 `sumo` 环境中的信号相位强制切换为 `state_idx`, 在 `sumo` 中进行 `DELTATIME` 步仿真, 返回仿真后的状态, 奖励, 仿真终止标志位, 打断标志位, 附加信息\n
        注: 若输入为 `None`, 将不作强制切换并保持原有固定相位执行

        Args:
            state_idx (int):相位索引

        Returns:
            dict[dict, float, bool, bool, dict]: `sumo` 仿真返回的状态, 奖励, 仿真终止标志位, 打断标志位, 附加信息
        """
        step_data = {
            "meanspeed":[],
            "traveltime":[],
            "veh_occ":[],
            "halt_occ":[],
            "in_occ":[],
            "out_occ":[],
            "phase":[],
            "sus":[],
            "re":[]
            }
        
        self.take_action(state_idx)

        for i in range(10):
            now_state = traci.trafficlight.getPhase(INT)
            now_time = traci.simulation.getTime()
            self.signalstate.update(now_state,now_time)
            ob_data:dict = self.get_observation()

            for key in step_data.keys():
                step_data[key].append(ob_data[key])
            
            traci.simulationStep()
            self.timestep += 1
        
        reward:float = self.get_reward(step_data)
        done:bool = self.get_done()
        truncated:bool = False # 是否因为意外没有到达终点
        obs:dict = self.get_observation()
        info = {} # 用于记录训练过程中的环境信息,便于观察训练状态
        return obs, reward, done, truncated, info
    
    def render(self) -> None:
        """可视化方法, `sumo-gui` 即可, 无需实现
        """
        pass


    
    def get_observation(self)->dict[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,int,int,int]:
        """从 `sumo` 仿真中获取状态

        Returns:
            dict: 状态空间\n
            `key`:
                检测器信息\n
                - "meanspeed" : 平均速度, np.ndarray,shape = (检测器个数,)
                - "traveltime": 行程时间, np.ndarray,shape = (检测器个数,)
                - "veh_occ" : 占有率, np.ndarray,shape = (检测器个数,)
                - "halt_occ" : 停车占有率, np.ndarray,shape = (检测器个数,)
                路侧信息\n
                - "in_occ" : 输入占有率, np.ndarray,shape = (入口车道数,)
                - "out_occ" : 输出占有率, np.ndarray,shape = (出口车道数,)
                交叉口信息\n
                - "phase" : 当前相位索引, int
                - "sus" : 当前相位已进行时间, int
                - "re" : 当前相位计划剩余时间, int
        """
        # 检测器信息
        ob_MeanSpeed = [] # 平均速度
        ob_VehicleNumber = [] # 车辆数
        ob_HaltVehicle = [] # 停车数
        # 车道信息
        ob_Input = [] # 流入
        ob_Output = [] # 流出
        ob_Traveltime = [] # 行程时间
        ob_Waittime = [] # 等待时间
        # 信号机信息
        ob_State = None # 当前相位
        ob_Remain = None # 计划剩余时间
        ob_Sustain = None # 持续时间

        for detector in self.detectors:
            ob_MeanSpeed.append(max(traci.lanearea.getLastStepMeanSpeed(detector),0.0))
            ob_VehicleNumber.append(traci.lanearea.getLastStepOccupancy(detector))
            ob_HaltVehicle.append(traci.lanearea.getLastStepHaltingNumber(detector))
        
        for inlaneid, outlaneid in zip(INPUTLANE, OUTPUTLANE):
            ob_Input.append(traci.lane.getLastStepOccupancy(inlaneid))
            ob_Output.append(traci.lane.getLastStepOccupancy(outlaneid))
            ob_Traveltime.append(traci.lane.getTraveltime(inlaneid))
            ob_Waittime.append(traci.lane.getWaitingTime(inlaneid))
        
        
        ob_State = traci.trafficlight.getPhase(tlsID=INT)
        ob_Sustain = self.timestep - self.signalstate.begin_time
        ob_Remain = traci.trafficlight.getNextSwitch(tlsID=INT) - self.timestep



        return {"meanspeed":np.array(ob_MeanSpeed),
                "traveltime":np.array(ob_Traveltime),
                "veh_occ":np.array(ob_VehicleNumber) ,
                "halt_occ":np.array(ob_HaltVehicle)* VEHCILELENGTH / np.array(LANEAREALENGTH),
                "in_occ":np.array(ob_Input),
                "out_occ":np.array(ob_Output),
                "phase":ob_State,
                "sus":ob_Sustain,
                "re":ob_Remain}
    
    
    def get_reward(self,step_data:dict)->float:
        """根据输入状态返回奖励

        Args:
            step_data (dict): 状态

        Returns:
            float: 奖励
        """
        # REMIND 数值计算需要注意
        # REMIND 应统一单位
        # REMIND nan值处理, 替换为0

        # TODO 只有惩罚没有奖励

        # region # NOTE 1. 延迟 = 行程时间 - 时间 （单位 s ）
        # BUG 延迟计算过大
        # delay:np.ndarray = step_data["traveltime"][-1]
        # endregion

        # region # NOTE 2. 排队 = 平均停车率
        queue:np.ndarray = np.zeros_like(step_data["veh_occ"][0])
        for i in step_data["halt_occ"]:
            queue += i
            np.nan_to_num(queue, copy=False)
        queue = queue/10.0
        # endregion

        # region # NOTE 3. 通行压力 = 当前平均车道占有率-目标平均车道占有率
        pressure:np.ndarray = np.zeros_like(step_data["in_occ"][0])
        for i,j in zip(step_data["in_occ"],step_data["out_occ"]):
            pressure += (i - j)
            
            np.nan_to_num(pressure,copy=False)
        pressure = pressure/10.0
        # endregion

        # region # NOTE 4. 李雅普诺夫函数 = 各方向停车率的标准差 
        lyapunov = 0
        v = np.array_split(queue,4)
        lyapunov = np.std(np.sum(v,axis=1))
        # endregion

        # region # NOTE 5. 切换惩罚
        phase_0,sus_0,remain_time = step_data["phase"][0],step_data["sus"][0],step_data["re"][0]
        phase_1= step_data["phase"][-1] # 动作执行后 (相位索引)

        phase_0 = round((phase_0-1)/3) * 3 # 转为[0,3,6,9]
        phase_1 = round((phase_1-1)/3) * 3
        pram_1 = np.abs(phase_1 - phase_0)/9.0 # 相位切换惩罚 0~1

        if remain_time < 10.0 and phase_1-phase_0 ==0: # 强制保持相位
            pram_2 =sus_0/MAXPHASETIME
            swtich_p = - pram_2 #-1~0
        elif remain_time < 10.0 and phase_1-phase_0 !=1: # 发生了强制切换相位
            swtich_p = -pram_2 * pram_1 #-1~0
        else: # 正常相位切换
            swtich_p = 1.0
        # endregion

        return - np.sum(queue) - np.sum(pressure) - lyapunov + swtich_p

    def get_done(self)->bool:
        """返回 `sumo` 仿真是否到达终点

        Returns:
            bool: 仿真完成标志位
        """
        if self.timestep == SIMTIME:
            return True
        else:
            return False
    
    def take_action(self, action:int)->None:
        """执行动作\n
        注: 若输入为 `None`, 则不执行任何动作
        
        >>> if action = None:
                pass # 保持固定相位
            else:
                action

        Args:
            action (int): 动作, 相位索引
        """
        if action is None:
            return None
        else:
            traci.trafficlight.setPhase(INT, action)
            traci.trafficlight.setPhaseDuration(INT, DELTADUR)
        return None

class SignalState():
    """当前相位信息\n
    
    ---

    attribute:
        `state`:int 当前相位索引\n
        `begin_time`:int 当前相位已持续时间\n
    
    ---

    method:
        `reset()`: 置零 `attribute`
        `update(state:int,time:int)` 更新 `attribute`

    """
    def __init__(self):
        self.state:int = 0
        """当前相位索引
        """
        self.begin_time:int = 0
        """当前相位开始时间刻
        """

    def reset(self):
        """置零 `attribute`
        """
        self.state = 0
        self.begin_time = 0
    
    def update(self, state:int,time:int):
        """根据输入的 相位和时间刻更新 `attribute`

        Args:
            state (int): _description_
            time (int): _description_
        """
        if state!=self.state:
            self.state = state
            self.begin_time = time

def GetFlattenedState(states_dict:dict)->np.ndarray:
    """将 `Env` 返回的状态提取需要的信息并展平成一维向量
    Args:
        states_dict (dict): 状态

    Returns:
        np.ndarray: 展平的状态信息
    """
    veh_occ:np.ndarray = states_dict["veh_occ"]
    halt_occ:np.ndarray = states_dict["halt_occ"]
    in_occ:np.ndarray = states_dict["in_occ"]
    out_occ:np.ndarray = states_dict["out_occ"]
    phase:float = states_dict["phase"]/12
    sus:float = states_dict["sus"]/MAXPHASETIME
    re:float = states_dict["re"]/MAXPHASETIME
    return np.concatenate((veh_occ, halt_occ, in_occ, out_occ,np.array([phase,sus,re])), axis=0)