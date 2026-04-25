from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import BimanualTask
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from collections import defaultdict
import random
from rlbench.backend.conditions import Condition
from pyrep.objects.dummy import Dummy
from pyrep.robots.arms.arm import Arm
import time

class LiftedCondition(Condition):

    def __init__(self, item: Shape, min_height: float, sustained_for_steps: int = 5):
        self.item = item
        self.min_height = min_height
        self._sustained_for_steps = sustained_for_steps
        self._is_met = False
        self._success_counter = 0

    def reset(self):
        """重置条件状态，在每个新任务开始时调用"""
        self._is_met = False
        self._success_counter = 0
        print("LiftedCondition has been reset.")
        return self

    def condition_met(self):
        # 如果条件已经满足过，就一直返回成功
        if self._is_met:
            return True, False
        
        try:
            pos = self.item.get_position()
            if random.random() < 0.2:
                print(f"Current fork position: {pos}, Success counter: {self._success_counter}/{self._sustained_for_steps}")
            
            if pos[2] >= self.min_height:
                self._success_counter += 1
            else:
                self._success_counter = 0
            
            if self._success_counter >= self._sustained_for_steps:
                self._is_met = True
                print("LiftedCondition has been met and is now permanently successful.")

            return self._is_met, False
        
        except Exception as e:
            print(f"Error in LiftedCondition: {e}")
            # 如果出现异常（例如对象不存在），返回未满足
            return False, False


class BimanualPickFork(BimanualTask):

    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        self.fork = Shape('Fork_phy')
      
        # 保存对条件对象的引用，以便后续重置
        self.lifted_condition = LiftedCondition(self.fork, 1.0, sustained_for_steps=10)
        self.register_success_conditions([self.lifted_condition])
        self.register_graspable_objects([self.fork])
        self.waypoint_mapping = defaultdict(lambda: 'right')
        self.waypoint_mapping.update({'waypoint0': 'left', 'waypoint2': 'left', 'waypoint6': 'left'})

        # 注册路径点能力 - 使用不同的回调函数处理不同阶段
        self.register_waypoint_ability_start(0, self._init_task_states)  # 任务开始时初始化状态
        self.register_waypoint_ability_end(2, self._left_arm_at_press_position)  # 左臂到达按压位置
        self.register_waypoint_ability_end(5, self._right_arm_at_grasp_position)  # 右臂到达抓取位置
        self.register_waypoint_ability_start(6, self._left_arm_release_pressure)  # 左臂释放压力
        
        # 初始化状态变量
        self.right_arm_at_grasp_pos = False
        self.left_arm_at_press_pos = False
        self.left_arm_released = False

    def _init_task_states(self, waypoint):
        """初始化任务状态变量"""
        self.right_arm_at_grasp_pos = False
        self.left_arm_at_press_pos = False
        print("Task initialized, starting bimanual fork pick operation")

    def _left_arm_at_press_position(self, waypoint):
        """左臂到达按压位置的回调"""
        self.left_arm_at_press_pos = True
        print("Left arm in position at waypoint2, pressing on fork handle")
        # 添加短暂延时，确保按压稳定
        time.sleep(0.2)  # 200毫秒的稳定按压时间
        
        # 设置标志，允许右臂抓取
        self.fork_ready_for_grasp = True

    def _right_arm_at_grasp_position(self, waypoint):
        """右臂到达抓取位置的回调"""
        if not self.left_arm_at_press_pos:
            print("Waiting for left arm to press fork first...")
            time.sleep(0.1)  # 短暂延时
            return
        
        self.right_arm_at_grasp_pos = True
        # 确保右臂抓住叉子
        grasp_success = self.robot.grasp(self.fork, name='right')
        if grasp_success:
            print("Right arm successfully grasped fork head at waypoint5")
        else:
            print("Right arm attempted to grasp fork at waypoint5")
            
        # 信号左臂可以释放压力并移动到waypoint6
        self.left_arm_released = True

    def _left_arm_release_pressure(self, waypoint):
        """左臂释放压力准备移动到waypoint6"""
        if self.right_arm_at_grasp_pos:
            print("Left arm releasing pressure, right arm has grasped fork")
        else:
            print("Left arm releasing pressure, waiting briefly for right arm")
            time.sleep(0.2)  # 短暂等待右臂
    
        # 不阻塞左臂的移动
        print("Left arm proceeding to waypoint6")

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        # 重置条件对象
        self.lifted_condition.reset()
        
        # 重新注册成功条件（双重保险）
        self.register_success_conditions([self.lifted_condition])
        
        # 重置任务状态变量
        self.right_arm_at_grasp_pos = False
        self.left_arm_at_press_pos = False
        self.left_arm_released = False
        
        print(f"Starting episode {index} of bimanual fork picking task")
        return ['pick up the fork']

    def variation_count(self) -> int:
        # TODO: The number of variations for this task.
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
