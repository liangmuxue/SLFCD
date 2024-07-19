import os
import torch
from tqdm import tqdm
from clam.utils.utils import get_simple_loader
from clam.datasets.wsi_dataset import Wsi_Region
from wsi_core.WholeSlideImage import WholeSlideImage
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore


def score2percentile(score, ref):
    percentile = percentileofscore(ref.flatten(), score)
    return percentile


def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level=-1, k=15, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)

    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)

    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, k=k, **kwargs)
    return heatmap


def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object


def compute_from_patches(wsi_object, feature_extractor=None, batch_size=512, feat_save_path=None, device=None, **wsi_kwargs):
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=0)
    mode = "w"
    with torch.no_grad():
        for idx, (roi, coords) in tqdm(enumerate(roi_loader), total=len(roi_loader), desc=f"process one {os.path.splitext(feat_save_path)[-1][:-3]}"):
            roi = roi.to(device)
            coords = coords.numpy()

            features = feature_extractor(roi)

            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)
            mode = "a"
    return wsi_object
