# Copyright (c) 2026 Weirong Chen
"""Evaluation script for Nova3r models. Computes Chamfer Distance and F-Score metrics on SCRREAM benchmarks."""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from nova3r.models.nova3r_img_cond import Nova3rImgCond

import time

from dust3r.datasets import get_data_loader
import json
import torch
import numpy as np
import os
import open3d as o3d
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
import torchvision.transforms as transforms

from eval.mv_recon.metric import preprocess_data, accuracy, completion, SSI3DScore_Scene, SSI3DScore_Scene_Multi
from eval.mv_recon.utils import save_point_cloud_with_outlier_removal
import croco.utils.misc as misc  # noqa
from collections import defaultdict

import csv

class BSAgonisticMetricLogger(misc.MetricLogger):
    '''
    a metric logger whose averaged metric is not influenced by 
    the evaluation batchsize used only for evaluation
    '''
    def __init__(self, delimiter="\t"):
        super().__init__(delimiter=delimiter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue

            assert isinstance(v, tuple)

            avg_loss, count = v
            if isinstance(avg_loss, torch.Tensor):
                avg_loss = avg_loss.item()
            if isinstance(count, torch.Tensor):
                count = count.item()
                assert isinstance(count, int)

            self.meters[k].update(avg_loss, count)
            


def build_dataset(args, dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(args, dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    print("batch size:", batch_size, 'num_workers:', num_workers)
    return loader


def _apply_eval_defaults(args):
    """Apply default values for evaluation parameters not set via Hydra config."""
    OmegaConf.set_struct(args, False)
    defaults = {
        'eval_vis': False,
        'eval_vis_fm': False,
        'save_dir': 'eval_results',
        'scale_inv': True,
        'outlier_filtering': True,
        'save_3dpts_per_n_batch': 1,
        'eval_stride': 1,
        'fm_step_size': 0.04,
        'fm_sampling': 'euler',
        'num_queries': 50000,
        'num_eval_pts': 100000,
        'cfg_scale': 1.0,
        'batch_size': 1,
        'print_freq': 20,
    }
    for key, value in defaults.items():
        if key not in args:
            OmegaConf.update(args, key, value)

    # Per-dataset defaults
    dataset_defaults = {
        'scrream_n1': {'num_queries': 110000, 'num_eval_pts': 100000},
        'scrream_n2': {'num_queries': 110000, 'num_eval_pts': 100000},
    }
    if args.test_dataset_name in dataset_defaults:
        for key, value in dataset_defaults[args.test_dataset_name].items():
            if key not in args or args[key] == defaults.get(key):
                OmegaConf.update(args, key, value)

    # Set source/target defaults for all datasets
    head_params = args.model.params.cfg.pts3d_head.params
    if 'target_source' not in head_params:
        OmegaConf.update(args, 'model.params.cfg.pts3d_head.params.target_source', 'src_complete')
    if 'query_source' not in head_params:
        OmegaConf.update(args, 'model.params.cfg.pts3d_head.params.query_source', 'src_complete')
    if 'target_sampling' not in head_params:
        OmegaConf.update(args, 'model.params.cfg.pts3d_head.params.target_sampling', 'none')
    if 'source_sampling' not in head_params:
        OmegaConf.update(args, 'model.params.cfg.pts3d_head.params.source_sampling', 'none')


def test_svd(args, show_gt=True):
    """Evaluate a Nova3r model on SCRREAM benchmarks.

    Loads the model from a checkpoint, runs inference on the specified test
    dataset, computes Chamfer Distance and F-Score metrics, and saves per-sample
    results (point clouds, images, CSV/JSON logs) to the configured save directory.
    """
    _apply_eval_defaults(args)

    device = 'cuda'
    data_root = args.data_root
    batch_size = args.batch_size
    num_workers = 4
    print('Loading model: {:s}'.format(args.model.name))
    model_config = args.model
    model = eval(model_config['name'])(**model_config['params'])

    model.to(device)

    ckpt_path = args.ckpt_path
    print("ckpt_path", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'model' in ckpt:
        print("Loading model from checkpoint")
        print(model.load_state_dict(ckpt['model'], strict=False))
    else:
        print(model.load_state_dict(ckpt, strict=False))

    del ckpt

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params / 1e6}M")

    # prepare data loader
    from nova3r.inference import loss_of_one_batch_lari, to_cpu, collate_with_cat

    test_dataset_dict = {
        "scrream_n1": f"SCRREAM(split='train', ROOT='{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n1_list.json', test_list_path='data/scrream/scrream_n1_list.json', resolution=518, aug_crop=0, input_n=1, n_ldi_layers=0, enforce_img_reso_for_eval=[518,392], max_pts=100000)",
        "scrream_n2": f"SCRREAM_MULTI(split='train', ROOT='{data_root}/eval_scrream', train_list_path='data/scrream/scrream_n2_list.json', test_list_path='data/scrream/scrream_n2_list.json', resolution=518, aug_crop=0, input_n=2, n_ldi_layers=1, enforce_img_reso_for_eval=[518,392], max_pts=100000)",
    }

    data_loader = build_dataset(args, test_dataset_dict[args.test_dataset_name], batch_size, num_workers, test=True)

    os.makedirs(args.save_dir, exist_ok=True)
    
    test_name = args.test_dataset_name

    if 'fm_sampling' in args:
        fm_sampling = args.fm_sampling
        test_name += f"_fm_{fm_sampling}"
        
    save_dir = os.path.join(args.save_dir, test_name)
    os.makedirs(save_dir, exist_ok=True)

    save_samples_dir = os.path.join(save_dir, 'examples')
    os.makedirs(save_samples_dir, exist_ok=True)


    if args.test_dataset_name == 'scrream_n2':
        test_criterion = SSI3DScore_Scene_Multi(num_eval_pts=args.num_eval_pts, fs_thres=[0.1, 0.05, 0.02], pts_sampling_mode='uniform', alignment='none', use_cd_align=True).to(device)
    elif args.test_dataset_name == 'scrream_n1':
        test_criterion = SSI3DScore_Scene(num_eval_pts=args.num_eval_pts, fs_thres=[0.1, 0.05, 0.02], pts_sampling_mode='uniform', alignment='none', use_cd_align=True).to(device)
    else:
        raise NotImplementedError(f"Unknown test dataset: {args.test_dataset_name}")

    epoch = 0
    model.eval()

    wrapped_vf = None


    metric_logger = BSAgonisticMetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = f'Test Epoch: [{epoch}]'
    
    # Set epoch for distributed sampling (if applicable)
    if hasattr(data_loader, 'dataset') and hasattr(data_loader.dataset, 'set_epoch'):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler, 'set_epoch'):
        data_loader.sampler.set_epoch(epoch)
    

    csv_log_file_path = os.path.join(save_dir, 'eval_log.csv')
    log_file_path = os.path.join(save_dir, 'eval_log.txt')
    # clear existing log file
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    if os.path.exists(csv_log_file_path):
        os.remove(csv_log_file_path)

    
    for i, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):

        if args.eval_stride > 0 and i % args.eval_stride != 0 and i > 0:
            continue

        start_time = time.time()
        result = loss_of_one_batch_lari(args, batch, model, test_criterion, device, symmetrize_batch=True, mode='test', num_queries=args.num_queries, model_wrapper=wrapped_vf, use_amp=args.amp)
        output = to_cpu(result)
        end_time = time.time()

        elapsed_time = (end_time - start_time)

        pts3d_data = output['data']
        pts3d_pred_eval, pts3d_uniform_gt, pts3d_uniform_gt_vis, pts3d_pred_ori = pts3d_data['pts3d_pred_eval'], pts3d_data['pts3d_uniform_gt'], pts3d_data['pts3d_uniform_gt_vis'], pts3d_data['pts3d_pred_ori']

        loss_details = output['loss']
        view = output['view']
        pred = output['pred']

        # Visualize flow matching steps
        if hasattr(args, 'eval_vis_fm') and args.eval_vis_fm:
            fm_save_dir = os.path.join(save_dir, 'fm_steps_glb') if save_dir else None
            visualize_fm_steps(pred, save_dir=fm_save_dir, batch_id=i * args.batch_size)

        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(csv_log_file_path):
            with open(csv_log_file_path, 'w', newline='') as csvfile:
                # Get all possible keys from loss_details to create headers
                headers = ['batch_id'] + list(loss_details.keys())
                writer = csv.writer(csvfile)
                writer.writerow(headers)
        
        # Prepare row data for CSV
        row_data = [i * args.batch_size]  # batch_id

        for key, value in loss_details.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value) or torch.isinf(value):
                    message = f"WARNING: {key} contains NaN or Inf: {value}"
                    print(message)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(message + '\n')
                        row_data.append(f"NaN/Inf")
                else:
                    tensor_value = value.item() if value.numel() == 1 else str(value)
                    message = f"{key}: {tensor_value}"
                    print(message)
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(message + '\n')
                        row_data.append(tensor_value)
            else:
                message = f"{key}: {value[0]}"
                print(message)
                with open(log_file_path, 'a') as log_file:
                    log_file.write(message + '\n')

                value_format = f"{value[0]:.4f}"
                row_data.append(value_format)

        # Write row to CSV
        with open(csv_log_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        metric_logger.update(**loss_details)


        
        if i % args.save_3dpts_per_n_batch == 0:
            if save_dir:
                name = batch[0]["instance"][0].replace("/","_").replace(" ","_")
                name = f"{i * args.batch_size:05d}_{name}"

                pts3d_pred_eval, pts3d_uniform_gt, pts3d_uniform_gt_vis, pts3d_pred_ori = pts3d_data['pts3d_pred_eval'], pts3d_data['pts3d_uniform_gt'], pts3d_data['pts3d_uniform_gt_vis'], pts3d_data['pts3d_pred_ori']
                
                if 'pts3d_uniform_gt_rgb' in pts3d_data:
                    pts3d_uniform_gt_vis_rgb = pts3d_data['pts3d_uniform_gt_rgb']
                else:
                    pts3d_uniform_gt_vis_rgb = None

                gt_save_dir = os.path.join(save_dir,'gt_pts')
                pred_save_dir = os.path.join(save_dir,f'pred_pts_{args.num_queries}')
                img_save_dir = os.path.join(save_dir,'img')

                os.makedirs(gt_save_dir, exist_ok=True)
                os.makedirs(img_save_dir, exist_ok=True)
                os.makedirs(pred_save_dir, exist_ok=True)

                save_point_cloud_with_outlier_removal(
                    filename=os.path.join(pred_save_dir, f"{name}_pred.ply"),
                    xyz=pts3d_pred_eval[0].numpy(), 
                    rgb=None, 
                    remove_outliers=True, 
                )
                print(f"Pred saved to { os.path.join(pred_save_dir, f'{name}_pred.ply')}")

                # save gt
                save_point_cloud_with_outlier_removal(
                    filename=os.path.join(gt_save_dir, f"{name}_pseudo_gt.ply"),
                    xyz=pts3d_uniform_gt[0].numpy(), 
                    rgb=None, 
                    remove_outliers=False, 
                )
                print(f"Pseudo GT saved to {os.path.join(gt_save_dir, f'{name}_pseudo_gt.ply')}")

                # save img
                img_filename = os.path.join(img_save_dir, f"{name}_rgb.jpg")
                # Save image - convert tensor to PIL and save
                img_tensor = batch[0]["img"][0]  # Get first image from batch
                img_tensor = torch.clamp(img_tensor, 0, 1)
                img_pil = transforms.ToPILImage()(img_tensor)
                img_pil.save(img_filename)
                print(f"Image saved to {img_filename}")

            
        # Clear cache after each batch to prevent memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Gather and print stats across processes (if using distributed evaluation)
    metric_logger.synchronize_between_processes()
    print("Averaged testing stats:", metric_logger)
    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {f'{k}_{tag}': getattr(meter, attr)
               for k, meter in metric_logger.meters.items()
               for tag, attr in aggs}

    results['exp_name'] = args.exp_name
    print("exp_name", args.exp_name)

    try:
        results_path = os.path.join(save_dir, "test_metrics.json")
        with open(results_path, "a") as f:
            json.dump(results, f, indent=4)
            f.write('\n')  # Add newline after each JSON object
    except Exception as e:
        import traceback
        traceback.print_exc()
    
    


    return results





def color_by_depth(pts, color_idx=2, colormap_name='viridis'):
    """
    Color a point cloud based on depth (z-axis).
    
    Args:
        pts: (N, 3) numpy array of point cloud
        color_idx: index of the axis to use for coloring (default is 2 for z-axis)
        colormap_name: name of the matplotlib colormap to use
        
    Returns:
        colors: (N, 3) numpy array of RGB colors (0-255)
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    # Convert to numpy if tensor
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    
    z_values = pts[:, color_idx]
    z_min, z_max = np.percentile(z_values, 2), np.percentile(z_values, 98)
    normalized_z = (z_values - z_min) / (z_max - z_min)
    
    # Create color map (blue for low z, red for high z)
    colormap = cm.get_cmap(colormap_name)
    colors = colormap(normalized_z)[:, :3]  # Remove alpha channel
    colors = (colors * 255).astype(np.uint8)

    return colors


def save_fm_steps_glb(pts3d_xyz_list, save_path, batch_idx=0):
    """
    Save flow matching steps as GLB file with all timesteps.
    
    Args:
        pts3d_xyz_list: Tensor with shape (steps, batch, N, 3)
        save_path: Path to save the GLB file
        batch_idx: Which batch element to save (default: 0)
    """
    import trimesh
    
    # Convert to numpy if needed
    if isinstance(pts3d_xyz_list, torch.Tensor):
        pts3d_xyz_list = pts3d_xyz_list.cpu().numpy()
    
    num_steps = pts3d_xyz_list.shape[0]
    
    # Create a scene to hold all timesteps
    scene = trimesh.Scene()
    
    for step in range(num_steps):
        xyz = pts3d_xyz_list[step, batch_idx]  # N,3
        

        # center the point cloud for better visualization

        # Color by depth using viridis colormap
        colors = color_by_depth(xyz, color_idx=2, colormap_name='viridis')
        
        # Create point cloud for this step
        point_cloud = trimesh.PointCloud(vertices=xyz, colors=colors)
        
        # Add to scene with a name indicating the timestep
        scene.add_geometry(point_cloud, node_name=f"fm_step_{step:03d}")
    
    # Export as GLB
    scene.export(save_path, file_type='glb')
    print(f"Saved 4D point cloud (FM steps) to {save_path}")

def visualize_fm_steps(pred, save_dir=None, batch_id=None):
    """
    Save flow matching steps as GLB file.

    Args:
        pred: Dictionary containing 'pts3d_xyz_list' with shape (steps, batch, N, 3)
        save_dir: Optional directory to save GLB file
        batch_id: Batch ID for naming the saved file
    """
    pts3d_xyz_list = pred['pts3d_xyz_list']

    # Save as GLB if save_dir is provided
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        glb_filename = f"fm_steps_{batch_id:05d}.glb" if batch_id is not None else "fm_steps.glb"
        glb_path = os.path.join(save_dir, glb_filename)
        save_fm_steps_glb(pts3d_xyz_list, glb_path, batch_idx=0)


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    cfg = cfg.experiment
    print(OmegaConf.to_yaml(cfg))
    test_svd(cfg, show_gt=True)

if __name__ == "__main__":
    main()