"""
Compute FLOPs for PI05 model inference using DeepSpeed Profiler
"""
import torch
torch._dynamo.disable()
import numpy as np
from pathlib import Path

from lerobot.configs import parser
from lerobot.envs.factory import make_env
from lerobot.envs.configs import LiberoEnv
from lerobot.policies.pretrained import PreTrainedPolicy, PreTrainedConfig
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.random_utils import set_seed

# 导入 DeepSpeed Profiler
from deepspeed.profiling.flops_profiler import get_model_profile, FlopsProfiler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_observation(obs, device):
    """准备观测数据用于策略推理 - VectorEnv格式"""
    batch_input = {}
    if "agent_pos" in obs:
        batch_input["observation.state"] = torch.from_numpy(obs["agent_pos"]).float().to(device)
    if "pixels" in obs and isinstance(obs["pixels"], dict):
        for k, v in obs["pixels"].items():
            img_tensor = torch.from_numpy(v).float().permute(0, 3, 1, 2).to(device) / 255.0
            batch_input[f"observation.images.{k}"] = img_tensor
    return batch_input

from vlash.configs.train_config import VLASHTrainConfig
from dataclasses import dataclass, field

@dataclass
class SpikeTestConfig:
    policy_name: str = "pi05"
    env_task: str = "libero_object"
    n_envs: int = 1
    max_steps: int = 100
    policy_path: str = "/data/datasets/models-hf/pi05_libero_finetuned"
    output_dir: str = "spike_test_results"
    seed: int = 42
    train_cfg: VLASHTrainConfig = field(default_factory=VLASHTrainConfig)

@parser.wrap()
def main(cfg: SpikeTestConfig):
    init_logging()
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    from vlash.policies.factory import get_policy_class
    from vlash.train import make_vlash_dataset
    
    logger.info(f"Loading policy from: {cfg.policy_path}")
    dataset = make_vlash_dataset(cfg.train_cfg)
    
    policy_cls = get_policy_class(cfg.policy_name)
    config = PreTrainedConfig.from_pretrained(cfg.policy_path)
    config.compile_model = False

    policy = policy_cls.from_pretrained(
        cfg.policy_path,
        config=config,
        dataset_stats=dataset.meta.stats
    )
    policy = policy.to(device)
    policy.eval()
    
    env_cfg = LiberoEnv(task=cfg.env_task, task_ids=[0])
    all_envs_dict = make_env(env_cfg, n_envs=1)
    
    first_suite = list(all_envs_dict.keys())[0]
    first_task_id = list(all_envs_dict[first_suite].keys())[0]
    env = all_envs_dict[first_suite][first_task_id]
    
    task_description = env.get_attr("task_description")[0]
    logger.info(f"Task: {task_description}")
    
    obs, info = env.reset()
    batch_obs = prepare_observation(obs, device)
    batch_obs["task"] = [task_description]
    
    def prepare_for_inference(batch, device):
        batch = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens, lang_masks = policy.prepare_language(batch, pad_to_max_length=False)
        return images, img_masks, state, lang_tokens, lang_masks
    
    import torch.nn as nn

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images, img_masks, lang_tokens, lang_masks, state):
            # 注意：DeepSpeed Profiler 需要 forward 返回 tensor 或由 tensor 组成的 tuple
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state
            )
    
    # --- DeepSpeed Profiler 核心部分 ---
    with torch.inference_mode():
        # 准备输入数据
        inputs = prepare_for_inference(batch_obs, device)
        images, img_masks, state, lang_tokens, lang_masks = inputs
        
        wrapper_model = ModelWrapper(policy.model).to(device)

        logger.info("Starting DeepSpeed Profiler...")
        
        # 使用 get_model_profile 进行分析
        # args 必须是 list/tuple 格式，对应 ModelWrapper.forward 的输入
        # flops, macs, params = get_model_profile(
        #     model=wrapper_model,
        #     args=[images, img_masks, lang_tokens, lang_masks, state],
        #     print_profile=True,      # 打印每一层的详细报告
        #     detailed=True,           # 显示算子级别的细节
        #     module_depth=-1,         # 递归深度，-1 表示显示所有层
        #     top_modules=1,           # 显示消耗最大的模块
        #     warm_up=1,               # 预热次数
        #     as_string=False,         # 返回数值以便后续计算
        # )

        prof = FlopsProfiler(wrapper_model)
        
        with torch.no_grad():
            wrapper_model(images, img_masks, lang_tokens, lang_masks, state)  # 预热
        
        prof.start_profile()
        with torch.no_grad():
            wrapper_model(images, img_masks, lang_tokens, lang_masks, state)
            
        flops = prof.get_total_flops()
        macs = prof.get_total_macs()
        params = prof.get_total_params()
        prof.end_profile()
        
        logger.info("-" * 30)
        logger.info(f"Total Params: {params / 1e6:.2f} M")
        logger.info(f"Total MACs:   {macs / 1e9:.2f} GMACs")
        logger.info(f"Total FLOPs:  {flops / 1e12:.2f} TFLOPs")
        logger.info("-" * 30)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()