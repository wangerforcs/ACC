import pynvml
import time
import threading

class GPUMonitor:
    def __init__(self, gpu_id=0, interval=0.1):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        self.interval = interval
        self.power_samples = []
        self.keep_running = False
        self.thread = None
        self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)

    def _monitor(self):
        while self.keep_running:
            try:
                # 获取功率 (单位: mW)
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle)
                self.power_samples.append(power / 1000.0)  # 转为 Watts
            except pynvml.NVMLError:
                pass
            time.sleep(self.interval)

    def start(self):
        """开始或恢复监控，不会清空之前的数据"""
        if not self.keep_running:
            self.keep_running = True
            self.thread = threading.Thread(target=self._monitor)
            self.thread.start()
            print(">>> 监控已启动/恢复...")

    def stop(self):
        """暂停监控"""
        if self.keep_running:
            self.keep_running = False
            if self.thread:
                self.thread.join()
            print(">>> 监控已暂停。")

    def clear(self):
        """清空所有已采样的统计数据"""
        self.power_samples = []
        print(">>> 统计数据已重置。")

    def get_report(self):
        """获取当前所有累积样本的统计报告"""
        if not self.power_samples:
            return None
        
        avg_power = sum(self.power_samples) / len(self.power_samples)
        max_power = max(self.power_samples)
        # 能耗 = 平均功率 * 总采样时长
        total_duration = len(self.power_samples) * self.interval
        energy_j = avg_power * total_duration
        
        return {
            "gpu_name": self.gpu_name,
            "avg_power_w": round(avg_power, 2),
            "max_power_w": round(max_power, 2),
            "total_energy_j": round(energy_j, 2),
            "total_time_s": round(total_duration, 2),
            "samples_count": len(self.power_samples)
        }

# --- 使用示例 ---
if __name__ == "__main__":
    monitor = GPUMonitor(gpu_id=0, interval=0.1)

    # 场景：测试第一轮推理
    monitor.start()
    print("运行第一轮推理...")
    time.sleep(3) 
    monitor.stop()

    # 中途可以做一些不计入功耗的操作，比如数据预处理
    print("执行非计算密集型操作（不记录功耗）...")
    time.sleep(2)

    # 场景：恢复监控，测试第二轮推理
    monitor.start()
    print("运行第二轮推理...")
    time.sleep(3)
    monitor.stop()

    # 获取最终汇总结果
    stats = monitor.get_report()
    
    if stats:
        print("\n" + "="*30)
        print(f"显卡型号: {stats['gpu_name']}")
        print(f"累计采样时间: {stats['total_time_s']} s")
        print(f"全程平均功率: {stats['avg_power_w']} W")
        print(f"全程峰值功率: {stats['max_power_w']} W")
        print(f"累计消耗能量: {stats['total_energy_j']} J")
        print("="*30)