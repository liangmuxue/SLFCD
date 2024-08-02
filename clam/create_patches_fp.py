from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
import os
import numpy as np
import argparse
import pandas as pd
import shutil
import cv2
from segmodel.segone import load_Seg_model


def stitching(file_path, wsi_object, downscale=64):
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    return heatmap


def segment(WSI_object, seg_params=None, filter_params=None, mask_file=None):
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
    return WSI_object


def patching(WSI_object, **kwargs):
    file_path = WSI_object.process_contours(**kwargs)
    return file_path


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, patch_size=256, step_size=256,
                  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                              'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt_easy'},
                  patch_level=0, use_default_params=False, seg=False, save_mask=True,
                  stitch=False, patch=False, process_list=None, seg_model=False, model=None, device='cpu'):
    # 如果有覆盖的标志，则先清除相关文件
    # 清除所有h5文件
    h5_path = os.path.join(source, "patches_level{}".format(patch_level))
    if os.path.exists(h5_path):
        shutil.rmtree(h5_path)
    os.mkdir(h5_path)
    # 清除列表文件
    if os.path.exists(os.path.join(save_dir, 'process_list_autogen.csv')):
        os.remove(os.path.join(save_dir, 'process_list_autogen.csv'))

    wsi_source = os.path.join(source, "data")
    slides = sorted(os.listdir(wsi_source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(wsi_source, slide))]
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    remove_df = []
    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        if not slide.endswith(".svs"):
            continue
        print("\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        # Inialize WSI
        full_path = os.path.join(source, "data", slide)
        xml_file = slide.replace(".svs", ".xml")
        xml_path = os.path.join(source, "xml", xml_file)

        WSI_object = WholeSlideImage(full_path)
        if os.path.exists(xml_path):
            WSI_object.initXML(xml_path)

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        # w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        # if w * h > 1e8:
        #     print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
        #     df.loc[idx, 'status'] = 'failed_seg'
        #     continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        # 很重要，通过分割来获取到轮廓
        if seg:
            if not seg_model:
                WSI_object = segment(WSI_object, current_seg_params, current_filter_params)
            else:
                WSI_object.segmentTissue_new_model(**current_seg_params, filter_params=current_filter_params,
                                                   model=model, device=device, mask_save_dir=mask_save_dir)

        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            if not os.path.exists(os.path.join(mask_save_dir, slide_id)):
                os.makedirs(os.path.join(mask_save_dir, slide_id))
            mask_path = os.path.join(mask_save_dir, slide_id, slide_id + '.jpg')
            mask.save(mask_path)

        # 这一步是最主要的，将标注区域保存为h5
        if patch:
            current_patch_params.update(
                {'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                 'save_path': patch_save_dir})
            patching(WSI_object=WSI_object, **current_patch_params)
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        file_path = os.path.join(patch_save_dir, slide_id + '.h5')
        if stitch:
            if os.path.isfile(file_path):
                heatmap = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        df.loc[idx, 'status'] = 'processed'
        if not os.path.isfile(file_path):
            remove_df.append(idx)

    df = pd.read_csv(os.path.join(save_dir, 'process_list_autogen.csv'))
    for i in remove_df:
        df = df.drop(i)
    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), mode='w', index=False)


parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type=str, default=r"/home/bavon/datasets/wsi/ais",
                    help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type=int, default=64, help='step_size')
parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
parser.add_argument('--seg_model', type=bool, default=False, help='seg model True or False')
parser.add_argument('--seg_model_path', type=str, default=r'/home/bavon/project/SLFCD/SLFCD/clam/segmodel/epoch-30.pth', help='seg model path')
parser.add_argument('--device', type=str, default='cuda:1', help='gpu or cpu')
parser.add_argument('--patch', default=True, action='store_true')
parser.add_argument('--seg', default=True, action='store_true')
parser.add_argument('--stitch', default=True, action='store_true')
parser.add_argument('--save_dir', type=str, default=r"/home/bavon/datasets/wsi/ais",
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=1, help='downsample level at which to patch')
parser.add_argument('--process_list', type=str, default=None,
                    help='name of list of images to process with parameters (.csv)')


if __name__ == '__main__':
    args = parser.parse_args()

    patch_save_dir = os.path.join(args.save_dir, 'patches_level{}'.format(args.patch_level))
    mask_save_dir = os.path.join(args.save_dir, 'mask')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None

    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    # 分割参数列表如下：
    # seg_level：用于分割 WSI 的下采样级别（默认值：-1，它使用最接近 64x 下采样的 WSI 中的下采样）
    # sthresh：分割阈值（正整数，默认值：8，使用更高的阈值会导致更少的前台和更多的后台检测）
    # mthresh：中值滤波器大小（正，奇数整数，默认值：7）
    # use_otsu：使用 otsu 的方法代替简单的二进制阈值（默认值：False）
    # close：在初始阈值（正整数或 -1，默认值：4）之后应用的附加形态闭合

    # 等值线滤波参数列表如下：
    # a_t：组织的区域过滤器阈值（正整数，相对于级别 0 时 512 x 512 的参考斑块大小，要考虑的检测到的前景轮廓的最小大小，例如，值 10 表示仅检测到大小大于 10 的前景轮廓 0 大小的 512 x 512 大小的斑块，默认值：100）
    # a_h：孔的区域过滤器阈值（正整数，前景轮廓中要避免的检测到的孔/腔的最小尺寸，再次相对于级别 0 的 512 x 512 大小的面片，默认值：16）
    # max_n_holes：每个检测到的前景等值线要考虑的最大孔数（正整数，默认值：10，最大值越高，修补越准确，但会增加计算成本）

    # 分割可视化参数列表如下：
    # vis_level：下采样级别以可视化分割结果（默认值：-1，使用WSI中最接近64倍下采样的下采样）
    # line_thickness：用于绘制的线粗细 可视化分割结果（正整数，以绘制线在级别 0 处占用的像素数表示，默认值：250）

    # 补丁参数列表如下：
    # use_padding：是否填充幻灯片的边框（默认值：True）
    # contour_fn：轮廓检查功能，用于决定应将面片视为前景还是背景（在“four_pt”之间进行选择 - 检查围绕面片中心的小网格中的所有四个点是否都在等高线内，
    # “center” - 检查面片的中心是否在等高线内， “basic” - 检查面片的左上角是否在等高线内， 默认值： 'four_pt_easy'）

    seg_params = {'seg_level': args.patch_level, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': args.patch_level, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt_easy'}

    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print("parameters: ", parameters)
    
    model = None
    if args.seg_model:
        model = load_Seg_model(args.seg_model_path, args.device)
    

    seg_and_patch(**directories, **parameters, patch_size=args.patch_size, step_size=args.step_size,
                  seg=args.seg, use_default_params=False, save_mask=True, stitch=args.stitch,
                  patch_level=args.patch_level, patch=args.patch, process_list=process_list,
                  seg_model=args.seg_model, model=model, device=args.device)
    print("process success!!!")


