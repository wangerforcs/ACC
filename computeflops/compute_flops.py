"""
Compute FLOPs for PI05 model inference
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

# Sentinel-VLA imports
from dataclasses import dataclass, field
import traceback

@dataclass
class SpikeTestConfig:
    """Spike Test 配置"""
    # 环境配置
    policy_name: str = "pi05"
    env_task: str = "libero_object"
    n_envs: int = 1
    max_steps: int = 100
    
    # Policy配置
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
    from vlash.policies.pi05.configuration_pi05 import PI05Config
    # 应该要在加载之前禁止compile, 注意这里是PreTrainedConfig, 不是PI05Config和PreTrainedPolicy
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
        """准备推理输入"""
        from vlash.policies.pi05.modeling_pi05 import PI05Policy
        
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
            return self.model.sample_actions(
                images, img_masks, lang_tokens, lang_masks, state
            )
    
    with torch.inference_mode():
        images, img_masks, state, lang_tokens, lang_masks = prepare_for_inference(batch_obs, device)
        
        from fvcore.nn import FlopCountAnalysis
        wrapper_model = ModelWrapper(policy.model).to(device)
        
        flops = FlopCountAnalysis(wrapper_model, (images, img_masks, lang_tokens, lang_masks, state))
        total_flops = flops.total()
        
        logger.info(f"Total FLOPs per inference: {total_flops / 1e12:.2f} TFLOPs")
        logger.info(f"Total FLOPs per inference: {total_flops / 1e9:.2f} GFLOPs")
        print(flops.by_module())

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()