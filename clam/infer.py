from __future__ import print_function

# The path can also be read from a config file, etc.
OPENSLIDE_PATH = r'D:\BaiduNetdiskDownload\openslide-bin-4.0.0.3-windows-x64\bin'

import os

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import sys

sys.path.append(r"D:\project\SLFCD1\project")

import argparse
import os
from clam.utils.utils import *
from clam.utils.eval_utils import initiate_model as initiate_model
from clam.models.model_clam import CLAM_MB, CLAM_SB
import h5py
import yaml
from clam.wsi_core.batch_process_utils import initialize_df
from clam.vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches
from clam.wsi_core.wsi_utils import sample_rois
from clam.utils.file_utils import save_hdf5
import warnings
from custom.train_with_clamdata import CoolSystem
from tqdm import tqdm

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--data_dir', type=str, default=r'D:\project\SLFCD\dataset\ais\data',
                    help='svs file or svs dir')
parser.add_argument('--save_path', type=str, default="heatmaps/output")
parser.add_argument('--config_file', type=str, default="config_template.yaml")
parser.add_argument('--device', type=str, default="cpu")
args = parser.parse_args()


def infer_single_slide(model, features, reverse_label_dict, n_classes=1):
    with torch.no_grad():
        features = features.to(device)
        if isinstance(model, (CLAM_SB, CLAM_MB)):
            logits, Y_prob, Y_hat, A, _ = model(features)
            Y_hat = Y_hat.item()

            if isinstance(model, (CLAM_MB, )):
                A = A[Y_hat]

            A = A.view(-1, 1).cpu().numpy()
        else:
            raise NotImplementedError

        print('predict Y_hat: {}'.format(reverse_label_dict[Y_hat]))

        probs, ids = torch.topk(Y_prob, n_classes)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])
        return ids, preds_str, probs, A


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params


if __name__ == '__main__':
    device = args.device
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict["data_arguments"]["data_dir"] = args.data_dir

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes'], "device": device})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))

    # preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_easy'}

    if os.path.isfile(data_args.data_dir):
        svs_path = [data_args.data_dir]
        slides = [os.path.split(data_args.data_dir)[-1]]
    elif os.path.isdir(data_args.data_dir):
        svs_path = [data_args.data_dir + "/" + i for i in sorted(os.listdir(data_args.data_dir))]
        slides = [os.path.split(slide)[-1] for slide in svs_path if data_args.slide_ext in slide]
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                       use_heatmap_args=False)
    df.loc[:, 'svs_path'] = svs_path

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)

    print('----------- initializing model from checkpoint -----------')
    # 第一阶段
    print('first ckpt path: {}'.format(model_args.slf_ckpt_path))
    feature_extractor = CoolSystem.load_from_checkpoint(model_args.slf_ckpt_path).to(device)
    feature_extractor = torch.nn.Sequential(*(list(feature_extractor.model.children())[:-1]))
    feature_extractor.eval()

    # 第二阶段
    ckpt_path = model_args.ckpt_path
    print('second ckpt path: {}'.format(ckpt_path))

    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path)
    else:
        raise NotImplementedError

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}

    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': step_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift, 'contour_fn': def_patch_params["contour_fn"]}

    # 一条一条的处理数据
    for i in range(len(process_stack)):
        slide_name = process_stack.loc[i, 'slide_id']
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        try:
            label = process_stack.loc[i, 'label']
        except KeyError:
            label = 'Unspecified'
        slide_id = slide_name.replace(data_args.slide_ext, '')

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        # ----------------------- save -------------------------------------------
        p_slide_save_dir = os.path.join(save_path, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)
        r_slide_save_dir = os.path.join(save_path, str(grouping), slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))

        pt_save_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        h5_save_path = os.path.join(r_slide_save_dir, slide_id + '.h5')

        heatmap_path = os.path.join(r_slide_save_dir, f'{slide_id}_blockmap.png')
        marked_image_save_dir = os.path.join(r_slide_save_dir, f'{slide_id}_marked_original.png')

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None
        slide_path = process_stack.loc[i, "svs_path"]

        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        print('----------- Initializing WSI object -----------')
        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)
        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # 热图可视化的实际补丁大小，补丁大小*下采样因子*自定义下采样因子
        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level

        print('----------- first model h5_features_file & pt_features_file -----------')
        _wsi_object = compute_from_patches(wsi_object=wsi_object, feature_extractor=feature_extractor,
                                           batch_size=exp_args.batch_size, **blocky_wsi_kwargs,
                                           feat_save_path=h5_save_path, device=device)

        # 从 h5 文件中获取特征数据并保存
        file = h5py.File(h5_save_path, "r")
        features = torch.tensor(file['features'][:])
        torch.save(features, pt_save_path)
        file.close()

        process_stack.loc[i, 'bag_size'] = len(features)
        features = features[:, :, 0, 0]

        # 保存 maks pkl
        wsi_object.saveSegmentation(mask_file)
        print('----------- second model Classification & attention -----------')
        Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, reverse_label_dict, exp_args.n_classes)
        del features

        file = h5py.File(h5_save_path, "r")
        coords = file['coords'][:]
        file.close()
        asset_dict = {'attention_scores': A, 'coords': coords}
        block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

        # 保存最高的3个预测结果
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        process_stack.to_csv(f'{save_path}/infer_data.csv', index=False)

        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        for sample in sample_args.samples:
            if sample['sample']:
                # 保存高注意力的图片
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                sample_save_dir = os.path.join(save_path, 'sampled_patches', str(tag), sample['name'], slide_id)
                os.makedirs(sample_save_dir, exist_ok=True)

                # 选取前15个结果
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                             score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in tqdm(
                        enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])),
                        total=sample["k"], desc='save ' + sample['name']):
                    # 截取注意力区域并保存
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level,
                                                       (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir,
                                            '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0],
                                                                                  s_coord[1], s_score)))

        # 绘制热力图
        heatmap, original_image = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,
                                              cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True,
                                              binarize=False, vis_level=-1, k=sample_args.samples[0]['k'],
                                              blank_canvas=False, thresh=-1, patch_size=vis_patch_size,
                                              convert_to_percentiles=True)

        heatmap.save(heatmap_path)
        original_image.save(marked_image_save_dir)
        del heatmap

    with open(os.path.join(save_path, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
