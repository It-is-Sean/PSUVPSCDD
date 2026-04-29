import os.path as osp
import cv2
import numpy as np
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class ScanNet_Multi(BaseMultiViewDataset):
    def __init__(
        self,
        *args,
        ROOT,
        complete_gt_filename="mesh_complete_reservoir_vh_clean.npz",
        complete_gt_points=10000,
        complete_gt_target_mode="complete_zpos",
        complete_gt_frustum_margin=1.0,
        complete_gt_min_views=2,
        max_interval=1,
        **kwargs,
    ):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = int(max_interval)
        self.complete_gt_filename = complete_gt_filename
        self.complete_gt_points = int(complete_gt_points)
        self.complete_gt_target_mode = str(complete_gt_target_mode)
        self.complete_gt_frustum_margin = float(complete_gt_frustum_margin)
        self.complete_gt_min_views = int(complete_gt_min_views)
        self._complete_cache = {}
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data(self.split)

    def _load_data(self, split):
        preferred_root = osp.join(self.ROOT, f"scans_{split}")
        if os.path.isdir(preferred_root):
            self.scene_root = preferred_root
        else:
            # Backward-compatible fallback for older layouts that only expose
            # train/test directories.
            self.scene_root = osp.join(
                self.ROOT, "scans_train" if split == "train" else "scans_test"
            )
        self.scenes = sorted(
            scene for scene in os.listdir(self.scene_root) if scene.startswith("scene")
        )

        offset = 0
        scenes = []
        sceneids = []
        scene_img_list = []
        images = []
        start_img_ids = []

        j = 0
        for scene in tqdm(self.scenes):
            scene_dir = osp.join(self.scene_root, scene)
            with np.load(
                osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True
            ) as data:
                basenames = data["images"]
                num_imgs = len(basenames)
                img_ids = list(np.arange(num_imgs) + offset)
                cut_off = (
                    self.num_views
                    if not self.allow_repeat
                    else max(self.num_views // 3, 3)
                )
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue

                start_img_ids.extend(start_img_ids_)
                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
                scenes.append(scene)
                scene_img_list.append(img_ids)

                # offset groups
                offset += num_imgs
                j += 1

        self.scenes = scenes
        self.sceneids = sceneids
        self.images = images
        self.start_img_ids = start_img_ids
        self.scene_img_list = scene_img_list

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _load_complete_points(self, scene_dir):
        cache_key = osp.realpath(scene_dir)
        if cache_key in self._complete_cache:
            return self._complete_cache[cache_key]
        cache_path = osp.join(scene_dir, self.complete_gt_filename)
        if not osp.isfile(cache_path):
            self._complete_cache[cache_key] = None
            return None
        with np.load(cache_path) as data:
            points = data["points_world"].astype(np.float32)
        self._complete_cache[cache_key] = points
        return points

    @staticmethod
    def _points_in_view(points_world, camera_pose, intrinsics, height, width, margin=1.0):
        world_to_cam = np.linalg.inv(camera_pose).astype(np.float32)
        pts_cam = (world_to_cam[:3, :3] @ points_world.T).T + world_to_cam[:3, 3]
        z = pts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return valid

        K = intrinsics[:3, :3] if intrinsics.shape[0] >= 3 else intrinsics
        proj = (K @ pts_cam[valid].T).T
        u = proj[:, 0] / proj[:, 2]
        v = proj[:, 1] / proj[:, 2]
        margin = float(margin)
        if margin <= 0:
            raise ValueError(f"complete_gt_frustum_margin must be positive, got {margin}")
        cx = float(width - 1) * 0.5
        cy = float(height - 1) * 0.5
        half_w = float(width - 1) * 0.5 * margin
        half_h = float(height - 1) * 0.5 * margin
        inside = (u >= cx - half_w) & (u <= cx + half_w) & (v >= cy - half_h) & (v <= cy + half_h)
        out = np.zeros(points_world.shape[0], dtype=bool)
        valid_idx = np.where(valid)[0]
        out[valid_idx[inside]] = True
        return out

    @staticmethod
    def _projected_ldi_points(points_world, view, grid=224, layers=4, min_depth_sep=0.05):
        """Approximate SCRREAM/NOVA LDI-style complete targets from a mesh reservoir."""
        if points_world.shape[0] == 0:
            return points_world
        world_to_cam = np.linalg.inv(view["camera_pose"]).astype(np.float32)
        pts_cam = (world_to_cam[:3, :3] @ points_world.T).T + world_to_cam[:3, 3]
        z = pts_cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            return points_world[:0]
        K = view["camera_intrinsics"][:3, :3]
        proj = (K @ pts_cam[valid].T).T
        u = proj[:, 0] / np.maximum(proj[:, 2], 1e-8)
        v = proj[:, 1] / np.maximum(proj[:, 2], 1e-8)
        height, width = view["depthmap"].shape
        inside = (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
        valid_idx = np.where(valid)[0][inside]
        if valid_idx.size == 0:
            return points_world[:0]
        u = u[inside]
        v = v[inside]
        z = z[valid_idx]
        grid_w = int(grid)
        grid_h = max(1, int(round(grid * float(height) / max(float(width), 1.0))))
        uu = np.minimum((u / max(float(width), 1.0) * grid_w).astype(np.int64), grid_w - 1)
        vv = np.minimum((v / max(float(height), 1.0) * grid_h).astype(np.int64), grid_h - 1)
        cell = vv * grid_w + uu
        order = np.lexsort((z, cell))
        cell_sorted = cell[order]
        z_sorted = z[order]
        idx_sorted = valid_idx[order]
        selected = []
        start = 0
        n = len(order)
        layers = max(1, int(layers))
        while start < n:
            end = start + 1
            c = cell_sorted[start]
            while end < n and cell_sorted[end] == c:
                end += 1
            chosen = 0
            last_z = -np.inf
            for j in range(start, end):
                zj = z_sorted[j]
                if chosen == 0 or zj - last_z >= min_depth_sep:
                    selected.append(idx_sorted[j])
                    last_z = zj
                    chosen += 1
                    if chosen >= layers:
                        break
            start = end
        if not selected:
            return points_world[:0]
        return points_world[np.asarray(selected, dtype=np.int64)]

    def _attach_complete_gt(self, scene_dir, views, rng):
        points_world = self._load_complete_points(scene_dir)
        if points_world is None or points_world.shape[0] == 0:
            return

        mode = self.complete_gt_target_mode
        masks = []
        for view in views:
            height, width = view["depthmap"].shape
            margin = self.complete_gt_frustum_margin if mode in {"anchor_frustum_margin", "anchor_frustum_expanded", "nova_anchor_frustum_margin"} else 1.0
            masks.append(
                self._points_in_view(
                    points_world,
                    view["camera_pose"],
                    view["camera_intrinsics"],
                    height,
                    width,
                    margin=margin,
                )
            )

        # LDI-style per-view complete targets: each input view contributes
        # depth-layered image-plane samples instead of a surface-uniform frustum
        # crop. Mode suffix controls the number of depth layers, e.g.
        # nova_per_view_ldi4.
        if mode.startswith("nova_per_view_ldi"):
            suffix = mode.replace("nova_per_view_ldi", "")
            digits = "".join(ch for ch in suffix if ch.isdigit())
            layers = int(digits) if digits else 4
            world_to_anchor = np.linalg.inv(views[0]["camera_pose"]).astype(np.float32)
            for view in views:
                selected_i = self._projected_ldi_points(points_world, view, grid=224, layers=layers)
                if selected_i.shape[0] > 0:
                    selected_anchor_i = (world_to_anchor[:3, :3] @ selected_i.T).T + world_to_anchor[:3, 3]
                    selected_i = selected_i[selected_anchor_i[:, 2] > 1e-6]
                if selected_i.shape[0] > self.complete_gt_points:
                    choice = rng.choice(selected_i.shape[0], size=self.complete_gt_points, replace=False)
                    selected_i = selected_i[choice]
                valid_num_i = int(selected_i.shape[0])
                padded_i = np.zeros((self.complete_gt_points, 3), dtype=np.float32)
                if valid_num_i > 0:
                    padded_i[:valid_num_i] = selected_i.astype(np.float32)
                view["pts3d_complete"] = padded_i
                view["pts3d_complete_valid_num"] = np.array(valid_num_i, dtype=np.int64)
            return

        # More literal NOVA-style per-view complete targets: each input view
        # contributes its own complete/amodal frustum crop. NOVA's
        # get_complete_pts3d() stacks per-view pts3d_complete tensors, so this
        # mode preserves the intended per-view target structure instead of
        # collapsing all selected frusta into view 0.
        if mode in {"nova_per_view_frustum", "input_frustum_per_view", "nova_per_view_frustum_anchor_zpos"}:
            world_to_anchor = np.linalg.inv(views[0]["camera_pose"]).astype(np.float32)
            for view_idx, view in enumerate(views):
                selected_i = points_world[masks[view_idx]]
                if selected_i.shape[0] > 0 and mode.endswith("anchor_zpos"):
                    selected_anchor_i = (world_to_anchor[:3, :3] @ selected_i.T).T + world_to_anchor[:3, 3]
                    selected_i = selected_i[selected_anchor_i[:, 2] > 1e-6]
                if selected_i.shape[0] > self.complete_gt_points:
                    choice = rng.choice(selected_i.shape[0], size=self.complete_gt_points, replace=False)
                    selected_i = selected_i[choice]
                valid_num_i = int(selected_i.shape[0])
                padded_i = np.zeros((self.complete_gt_points, 3), dtype=np.float32)
                if valid_num_i > 0:
                    padded_i[:valid_num_i] = selected_i.astype(np.float32)
                view["pts3d_complete"] = padded_i
                view["pts3d_complete_valid_num"] = np.array(valid_num_i, dtype=np.int64)
            return

        # NOVA3R-style complete target support: keep mesh/depth-derived scene
        # points that lie inside the frustum of the selected input view(s),
        # including occluded points. This is a frustum test only, not a
        # visibility/depth-buffer test.
        if mode in {
            "complete_zpos",
            "union_zpos",
            "union_frustum_zpos",
            "nova_input_frustum",
            "input_frustum",
            "input_frustum_union",
        }:
            keep = np.logical_or.reduce(masks)
        elif mode in {"anchor_frustum", "nova_anchor_frustum"}:
            keep = masks[0]
        elif mode in {"anchor_frustum_margin", "anchor_frustum_expanded", "nova_anchor_frustum_margin"}:
            keep = masks[0]
        elif mode in {"covered_by_ge2", "multi_view_ge2", "covered_by_ge2_anchorfb"}:
            keep = np.stack(masks, axis=0).sum(axis=0) >= max(2, self.complete_gt_min_views)
        elif mode.startswith("covered_by_ge"):
            min_views = int(mode.replace("covered_by_ge", ""))
            keep = np.stack(masks, axis=0).sum(axis=0) >= min_views
        else:
            raise ValueError(f"Unsupported complete_gt_target_mode={mode!r}")

        selected = points_world[keep]
        if selected.shape[0] == 0 and mode.endswith("_anchorfb"):
            selected = points_world[masks[0]]
        if selected.shape[0] == 0:
            return

        # The NOVA target convention transforms complete points into the first
        # input camera frame. A point that is visible in view 1/2/3 can still lie
        # behind the first/anchor camera after that canonical transform. Those
        # behind-anchor points are out-of-domain for the generator target and
        # caused large negative-z fractions in ScanNet src_complete GT.
        world_to_anchor = np.linalg.inv(views[0]["camera_pose"]).astype(np.float32)
        selected_anchor = (world_to_anchor[:3, :3] @ selected.T).T + world_to_anchor[:3, 3]
        selected = selected[selected_anchor[:, 2] > 1e-6]
        if selected.shape[0] == 0 and mode.endswith("_anchorfb"):
            selected = points_world[masks[0]]
            selected_anchor = (world_to_anchor[:3, :3] @ selected.T).T + world_to_anchor[:3, 3]
            selected = selected[selected_anchor[:, 2] > 1e-6]
        if selected.shape[0] == 0:
            return

        if selected.shape[0] > self.complete_gt_points:
            choice = rng.choice(selected.shape[0], size=self.complete_gt_points, replace=False)
            selected = selected[choice]
        valid_num = int(selected.shape[0])

        padded = np.zeros((self.complete_gt_points, 3), dtype=np.float32)
        padded[:valid_num] = selected.astype(np.float32)

        for view_idx, view in enumerate(views):
            view["pts3d_complete"] = padded.copy()
            view["pts3d_complete_valid_num"] = np.array(valid_num if view_idx == 0 else 0, dtype=np.int64)

    def _get_views(self, idx, resolution, rng, num_views):
        start_id = self.start_img_ids[idx]
        all_image_ids = self.scene_img_list[self.sceneids[start_id]]
        if num_views == 1:
            image_idxs = np.array([start_id])
            ordered_video = False
        else:
            pos, ordered_video = self.get_seq_from_start_id(
                num_views,
                start_id,
                all_image_ids,
                rng,
                max_interval=self.max_interval,
                video_prob=0.6,
                fix_interval_prob=0.6,
                block_shuffle=16,
            )
            image_idxs = np.array(all_image_ids)[pos]

        views = []
        scene_dir = None
        for v, view_idx in enumerate(image_idxs):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.scene_root, self.scenes[scene_id])
            rgb_dir = osp.join(scene_dir, "color")
            depth_dir = osp.join(scene_dir, "depth")
            cam_dir = osp.join(scene_dir, "cam")

            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
            # Load depthmap
            depthmap = imread_cv2(
                osp.join(depth_dir, basename + ".png"), cv2.IMREAD_UNCHANGED
            )
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            cam = np.load(osp.join(cam_dir, basename + ".npz"))
            camera_pose = cam["pose"]
            intrinsics = cam["intrinsics"]
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="ScanNet",
                    label=self.scenes[scene_id] + "_" + basename,
                    instance=f"{str(idx)}_{str(view_idx)}",
                    view_label=f"input_view{v}",
                    is_metric=self.is_metric,
                    is_video=ordered_video,
                    quantile=np.array(0.98, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )
        if scene_dir is not None:
            self._attach_complete_gt(scene_dir, views, rng)
        assert len(views) == num_views
        return views
