from mmpose.apis import init_model as init_pose_estimator
from mmpose.apis import inference_topdown
from mmpose.evaluation.metrics import CocoMetric
from mmpose.registry import METRICS
from mmpose.datasets import build_dataset
from mmengine.registry import FUNCTIONS


from mmengine.config import Config
from mmpose.models.data_preprocessors import PoseDataPreprocessor

import torch
from torch.utils.data import DataLoader
import tqdm
import time

if __name__ == "__main__":
    device = 'cuda'

    mmpose_cfg = "./configs/body_2d_keypoint/topdown_heatmap/coco/dy_router_hm_res50_hrnet32_256x192.py"
    mmpose_ckpt = './work_dirs/dy_hm_res50_hrnet32_256x192_woRGB_sge_Pconv_imgrouterV10/epoch_1.pth'
    warm_up_iters = 10
    batch_size = 64
    num_workers = 2

    config = Config.fromfile(mmpose_cfg)
    pose_estimator = None
    pose_estimator = init_pose_estimator(
        config,
        mmpose_ckpt,
        device=device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False)))
    )
    pose_estimator.eval()

    config.val_dataloader.dataset.bbox_file = None
    config.val_dataloader.dataset.test_mode = False
    
    dataset = build_dataset(config.val_dataloader.dataset)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=FUNCTIONS.get('pseudo_collate'),
        persistent_workers=True,
        drop_last=False,
    )
    
    evaluator: CocoMetric = METRICS.build(config.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    
    start = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(dataloader)):
            if i == warm_up_iters - 1:
                start = time.time()
            pose_estimator.val_step(data)
        
    end = time.time()
    
    print(f"BenchMark on validation: {(len(dataloader) - warm_up_iters) * batch_size / (end - start)}")
