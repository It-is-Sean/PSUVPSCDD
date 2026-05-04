from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa

_OPTIONAL_IMPORT_ERRORS = {}


def _optional_import(module_name, *symbols):
    try:
        module = __import__(f"{__name__}.{module_name}", fromlist=list(symbols))
        for symbol in symbols:
            globals()[symbol] = getattr(module, symbol)
    except Exception as exc:  # noqa: BLE001
        _OPTIONAL_IMPORT_ERRORS[module_name] = exc


_optional_import("arkitscenes", "ARKitScenes_Multi")
_optional_import("arkitscenes_highres", "ARKitScenesHighRes_Multi")
_optional_import("bedlam", "BEDLAM_Multi")
_optional_import("blendedmvs", "BlendedMVS_Multi")
_optional_import("co3d", "Co3d_Multi")
_optional_import("cop3d", "Cop3D_Multi")
_optional_import("dl3dv", "DL3DV_Multi")
_optional_import("dynamic_replica", "DynamicReplica")
_optional_import("eden", "EDEN_Multi")
_optional_import("hypersim", "HyperSim_Multi")
_optional_import("hoi4d", "HOI4D_Multi")
_optional_import("irs", "IRS")
_optional_import("mapfree", "MapFree_Multi")
_optional_import("megadepth", "MegaDepth_Multi")
_optional_import("mp3d", "MP3D_Multi")
_optional_import("mvimgnet", "MVImgNet_Multi")
_optional_import("mvs_synth", "MVS_Synth_Multi")
_optional_import("omniobject3d", "OmniObject3D_Multi")
_optional_import("pointodyssey", "PointOdyssey_Multi")
_optional_import("realestate10k", "RE10K_Multi")
_optional_import("scannet", "ScanNet_Multi")
_optional_import("scannetpp", "ScanNetpp_Multi")
_optional_import("smartportraits", "SmartPortraits_Multi")
_optional_import("spring", "Spring")
_optional_import("synscapes", "SynScapes")
_optional_import("tartanair", "TartanAir_Multi")
_optional_import("threedkb", "ThreeDKenBurns")
_optional_import("uasol", "UASOL_Multi")
_optional_import("urbansyn", "UrbanSyn")
_optional_import("unreal4k", "UnReal4K_Multi")
_optional_import("vkitti2", "VirtualKITTI2_Multi")
_optional_import("waymo", "Waymo_Multi")
_optional_import("wildrgbd", "WildRGBD_Multi")

from accelerate import Accelerator


def get_data_loader(
    dataset,
    batch_size,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    accelerator: Accelerator = None,
    fixed_length=False,
):
    import torch

    if isinstance(dataset, str):
        dataset = eval(dataset)

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            world_size=accelerator.num_processes,
            fixed_length=fixed_length,
        )
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    except (AttributeError, NotImplementedError):
        sampler = None

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
        )

    return data_loader
