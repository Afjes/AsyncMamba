from datetime import datetime
from typing import Optional, Dict


class ExperimentTimer:
    """
    通用实验计时器，支持全局总计时和分阶段计时
    自动处理时间格式转换，所有输出强制刷新，解决tee重定向缓冲问题
    """
    def __init__(self, experiment_name: str = "实验"):
        self.experiment_name = experiment_name
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.stages: Dict[str, float] = {}  # 阶段名称 -> 耗时(秒)
        self.current_stage_start: Optional[datetime] = None

    def start(self, mode: str = "TRAIN") -> None:
        """
        启动全局计时器
        Args:
            mode: 实验模式，如TRAIN/ABLATION/TEST
        """
        self.start_time = datetime.now()
        print("\n" + "="*80, flush=True)
        print(f"🚀 【实验启动】{self.experiment_name} | 模式: {mode.upper()}", flush=True)
        print(f"🕒 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print("="*80 + "\n", flush=True)

    def start_stage(self, stage_name: str) -> None:
        """
        开始一个新的阶段计时
        Args:
            stage_name: 阶段名称，如"数据加载"、"模型训练"、"测试评估"
        """
        self.current_stage_start = datetime.now()
        print(f"\n⏳ 开始阶段: {stage_name}", flush=True)
        print(f"🕒 阶段开始时间: {self.current_stage_start.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    def end_stage(self, stage_name: Optional[str] = None) -> float:
        """
        结束当前阶段计时
        Args:
            stage_name: 阶段名称，可选，不填则使用上一个阶段的名称
        Returns:
            阶段耗时(秒)
        """
        if self.current_stage_start is None:
            print("⚠️  警告: 没有正在进行的阶段计时", flush=True)
            return 0.0
        
        end_time = datetime.now()
        duration = (end_time - self.current_stage_start).total_seconds()
        
        if stage_name is None:
            stage_name = f"阶段_{len(self.stages)+1}"
        
        self.stages[stage_name] = duration
        hours, minutes, seconds = self._format_duration(duration)
        print(f"✅ 阶段完成: {stage_name}", flush=True)
        print(f"⏱️  阶段耗时: {hours}小时{minutes}分{seconds}秒", flush=True)
        
        self.current_stage_start = None
        return duration

    def end(self) -> float:
        """
        结束全局计时，打印总耗时和各阶段耗时统计
        Returns:
            总耗时(秒)
        """
        if self.start_time is None:
            print("⚠️  警告: 计时器尚未启动", flush=True)
            return 0.0
        
        self.end_time = datetime.now()
        total_duration = (self.end_time - self.start_time).total_seconds()
        total_h, total_m, total_s = self._format_duration(total_duration)

        print("\n" + "="*80, flush=True)
        print(f"✅ 【实验完成】{self.experiment_name} 所有任务执行完毕", flush=True)
        print(f"🕒 开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"🕒 结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        print(f"⏱️  总耗时: {total_h}小时{total_m}分{total_s}秒", flush=True)
        
        if self.stages:
            print("\n📊 各阶段耗时统计:", flush=True)
            print("-"*40, flush=True)
            for stage, duration in self.stages.items():
                h, m, s = self._format_duration(duration)
                percentage = (duration / total_duration) * 100
                print(f"  {stage:<20} | {h:2d}h {m:2d}m {s:2d}s | {percentage:5.1f}%", flush=True)
            print("-"*40, flush=True)
        
        print("="*80 + "\n", flush=True)
        return total_duration

    @staticmethod
    def _format_duration(seconds: float) -> tuple[int, int, int]:
        """将秒数转换为(小时, 分钟, 秒)格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return hours, minutes, seconds


# 便捷函数：创建并启动计时器
def start_timer(experiment_name: str = "实验", mode: str = "TRAIN") -> ExperimentTimer:
    timer = ExperimentTimer(experiment_name)
    timer.start(mode)
    return timer