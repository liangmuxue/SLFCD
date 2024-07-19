import os
import sys
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/')
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/extras/')
sys.path.append(r'/home/bavon/project/SLFCD/SLFCD/project/')
from topk.svm import SmoothTop1SVM
import argparse
from clam.utils.utils import *
from clam.models.model_clam import CLAM_SB
import h5py
import yaml
from clam.wsi_core.wsi_utils import sample_rois
from clam.utils.file_utils import save_hdf5
from custom.train_with_clamdata import CoolSystem
from tqdm import tqdm
from wsi_core.WholeSlideImage import WholeSlideImage
from clam.datasets.dataset_h5 import Whole_Slide_Bag_FP

import warnings

warnings.filterwarnings("ignore")

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from multiprocessing import Process, Queue
from threading import Thread
import cv2


class Infer:
    def __init__(self, args):
        config_path = os.path.join('heatmaps/configs', args.config_file)
        config_dict = yaml.safe_load(open(config_path, 'r'))
        config_dict["data_arguments"]["data_dir"] = args.data_dir
        config_dict["data_arguments"]["save_path"] = args.save_path
        config_dict["model_arguments"]["device"] = args.device

        self.data_args = argparse.Namespace(**config_dict['data_arguments'])
        self.model_args = argparse.Namespace(**config_dict['model_arguments'])
        self.heatmap_args = argparse.Namespace(**config_dict['heatmap_arguments'])
        self.sample_args = argparse.Namespace(**config_dict['sample_arguments'])

        self.ais_params = argparse.Namespace(**config_dict['ais_arguments'])
        self.hsil_params = argparse.Namespace(**config_dict['hsil_arguments'])
        self.lsil_params = argparse.Namespace(**config_dict['lsil_arguments'])

        self.def_seg_params = config_dict['seg_arguments']
        self.def_filter_params = config_dict['filter_arguments']
        self.def_vis_params = config_dict['vis_arguments']
        self.def_patch_params = config_dict['patch_arguments']

        self.model_dict = {"dropout": self.model_args.drop_out, 'n_classes': self.model_args.n_classes,
                           "size_arg": self.model_args.model_size, 'k_sample': self.model_args.k_sample}

        self.kwargs = {"cmap": self.heatmap_args.cmap, "alpha": self.heatmap_args.alpha,
                       "use_holes": self.heatmap_args.use_holes, "binarize": self.heatmap_args.binarize,
                       "vis_level": 256, "k": self.sample_args.samples["k"],
                       "blank_canvas": self.heatmap_args.blank_canvas, "thresh": -1,
                       "patch_size": 256, "convert_to_percentiles": True}

        self.custom_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)])

        self.results = {}
        self.orig_results = {
            "category": "normal",
            "boxes": {
                "ais": {"corr": []},
                "hsil": {"corr": []},
                "lsil": {"corr": []}
            }
        }

        self.flag = True
        self.color_value = (85, 85, 205)
        self.results_ais_hsil_lsil = {}

    def load_model(self, model_args):
        # 第一阶段
        model_one = CoolSystem.load_from_checkpoint(model_args['slf_ckpt_path'])
        model_one = model_one.to(self.model_args.device)
        model_one.eval()
        print('first ckpt path: {}'.format(model_args['slf_ckpt_path']))

        # 第二阶段
        instance_loss_fn = SmoothTop1SVM(n_classes=self.model_args.n_classes)
        model_two = CLAM_SB(**self.model_dict, instance_loss_fn=instance_loss_fn)
        model_two.load_state_dict(torch.load(model_args['ckpt_path'], map_location=self.model_args.device))
        model_two.eval()
        print('second ckpt path: {}'.format(model_args['ckpt_path']))
        return model_one, model_two

    def infer(self, image_path, name, patch_level, patch_size, step_size, model, label_dict, seg_params, kwargs):
        slide_id = os.path.split(image_path)[-1][:-4]

        # ----------------------- save -------------------------------------------
        slide_save_dir = os.path.join(self.data_args.save_path, slide_id, name)
        os.makedirs(slide_save_dir, exist_ok=True)

        TwoPhase_img_save_dir = os.path.join(slide_save_dir, 'TwoPhase')
        os.makedirs(TwoPhase_img_save_dir, exist_ok=True)

        pt_save_path = os.path.join(slide_save_dir, slide_id + '.pt')
        h5_save_path_two = os.path.join(slide_save_dir, slide_id + '_two.h5')

        segment_mask_path = os.path.join(slide_save_dir, slide_id + '.jpg')
        heatmap_path = os.path.join(slide_save_dir, slide_id + '_blockmap.png')
        marked_image_save_path = os.path.join(slide_save_dir, slide_id + '_marked_original.png')

        # Load segmentation and filter parameters
        filter_params = self.def_filter_params.copy()
        patch_params = self.def_patch_params.copy()
        patch_params.update({'patch_level': patch_level, 'patch_size': patch_size[0],
                             'step_size': step_size[0], 'save_path': slide_save_dir})

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

        try:
            WSI_object = WholeSlideImage(image_path)
            WSI_object.segmentTissue(**seg_params, filter_params=filter_params)
            mask = WSI_object.visWSI(**self.def_vis_params)
            mask.save(segment_mask_path)

            if WSI_object.contours_tissue:
                WSI_object.process_contours(**patch_params)

                dataset = Whole_Slide_Bag_FP(file_path=os.path.join(slide_save_dir, slide_id + ".h5"),
                                             wsi=WSI_object.wsi, pretrained=True,
                                             custom_transforms=self.custom_transforms,
                                             custom_downsample=self.model_args.custom_downsample,
                                             target_patch_size=self.model_args.target_patch_size)

                loader = DataLoader(dataset=dataset, batch_size=self.model_args.batch_size, num_workers=0,
                                    pin_memory=True,
                                    collate_fn=collate_features)

                mode = 'w'
                with torch.no_grad():
                    with tqdm(total=len(loader)) as t:
                        for count, (batch, coords) in enumerate(loader):
                            t.set_description(desc=f"{name} {slide_id}")
                            t.set_postfix(steps=count)
                            t.update(1)

                            batch = batch.to(self.model_args.device)

                            features, cls_emb = model[0](batch)
                            features = features.cpu().numpy()

                            asset_dict = {'features': features, 'coords': coords}
                            save_hdf5(h5_save_path_two, asset_dict, attr_dict=None, mode=mode)
                            mode = 'a'

                file = h5py.File(h5_save_path_two, "r")
                features = file['features'][:]
                coords = file['coords'][:]

                features = torch.from_numpy(features)
                torch.save(features, pt_save_path)

                _, Y_prob, Y_hat, A, _ = model[1](features)
                Y_hat = Y_hat.item()
                Y_prob = Y_prob[0].detach().tolist()[Y_hat]
                scores = A.cpu().detach().numpy()

                # ----------------- 保存第二阶段的图片 -----------------
                # 选取前15个结果
                sample_results = sample_rois(scores, coords, k=self.sample_args.samples['k'],
                                             mode=self.sample_args.samples['mode'],
                                             seed=self.sample_args.samples['seed'],
                                             score_start=self.sample_args.samples.get('score_start', 0),
                                             score_end=self.sample_args.samples.get('score_end', 1))

                # 保存高注意力的图片
                tag = "pred_{}_prob_{:.3f}".format(label_dict[Y_hat], Y_prob)
                for idx, (s_coord, s_score) in tqdm(
                        enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])),
                        total=self.sample_args.samples["k"], desc=f'{name} save ' + self.sample_args.samples['name']):
                    # 截取注意力区域并保存
                    patch = WSI_object.wsi.read_region(tuple(s_coord), patch_level, patch_size).convert('RGB')
                    patch.save(os.path.join(TwoPhase_img_save_dir, '[{},{}].png'.format(s_coord[0], s_coord[1])))

                # 绘制热力图
                kwargs['patch_size'] = patch_size
                heatmap, original_image = WSI_object.visHeatmap(scores=sample_results['sampled_scores'],
                                                                coords=sample_results['sampled_coords'],
                                                                **kwargs)

                heatmap.save(heatmap_path.replace('.png', f'_{tag}.png'))
                original_image.save(marked_image_save_path.replace('.png', f'_{tag}.png'))
                return {slide_id: {name: {label_dict[Y_hat]: sample_results['sampled_coords']}}}
            else:
                return {slide_id: {name: "样本无效!!!"}}
        except Exception as e:
            print(e)

    def process_ais(self, q_ais_in, q_ais_out):
        # ----------------------------- ais -----------------------------
        model = self.load_model(self.model_args.ais)
        patch_level = self.ais_params.patch_level
        patch_size = tuple([self.ais_params.patch_size for i in range(2)])
        slide_size = tuple([self.ais_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.ais_params.label_dict.values(), self.ais_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            try:
                image_path = q_ais_in.get()
                seg_params = self.def_seg_params.copy()
                if not self.results_ais_hsil_lsil.get(image_path):
                    self.results_ais_hsil_lsil[image_path] = {}

                result = self.infer(image_path, 'ais', patch_level, patch_size, slide_size, model, label_dict,
                                    seg_params, self.kwargs)
                q_ais_out.put(result)
            except Exception as e:
                print(e)

    def process_hsil(self, q_hsil_in, q_hsil_out):
        # ----------------------------- hsil ------------------------------
        model = self.load_model(self.model_args.hsil)
        patch_level = self.hsil_params.patch_level
        patch_size = tuple([self.hsil_params.patch_size for i in range(2)])
        slide_size = tuple([self.hsil_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.hsil_params.label_dict.values(), self.hsil_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            image_path = q_hsil_in.get()
            seg_params = self.def_seg_params.copy()

            if not self.results_ais_hsil_lsil.get(image_path):
                self.results_ais_hsil_lsil[image_path] = {}

            process_thread_hsil = Thread(target=self.infer, args=(image_path, 'hsil', patch_level, patch_size,
                                                                  slide_size, model, label_dict, seg_params,
                                                                  q_hsil_out, self.kwargs))
            process_thread_hsil.start()

            # result = self.infer(image_path, 'hsil', patch_level, patch_size, slide_size, model, label_dict,
            #                     seg_params, self.kwargs)
            # q_hsil_out.put(result)

    def process_lsil(self, q_lsil_in, q_lsil_out):
        model = self.load_model(self.model_args.lsil)
        patch_level = self.lsil_params.patch_level
        patch_size = tuple([self.lsil_params.patch_size for i in range(2)])
        slide_size = tuple([self.lsil_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.lsil_params.label_dict.values(), self.lsil_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            image_path = q_lsil_in.get()
            seg_params = self.def_seg_params.copy()

            result = self.infer(image_path, 'lsil', patch_level, patch_size, slide_size, model, label_dict,
                                seg_params, self.kwargs)
            q_lsil_out.put(result)

    def process_ais_hsil(self, q_ais_hsil_in, q_ais_hsil_out):
        # ----------------------------- ais ------------------------------
        model_ais = self.load_model(self.model_args.ais)
        label_dict_ais = dict(zip(self.ais_params.label_dict.values(), self.ais_params.label_dict.keys()))

        # ----------------------------- hsil ------------------------------
        model_hsil = self.load_model(self.model_args.hsil)
        label_dict_hsil = dict(zip(self.hsil_params.label_dict.values(), self.hsil_params.label_dict.keys()))

        patch_level = self.ais_params.patch_level
        patch_size = tuple([self.ais_params.patch_size for i in range(2)])
        slide_size = tuple([self.ais_params.slide_size for i in range(2)])
        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            image_path = q_ais_hsil_in.get()
            seg_params = self.def_seg_params.copy()

            # if not self.results_ais_hsil_lsil.get(image_path):
            #     self.results_ais_hsil_lsil[image_path] = {}

            # ais_thread = Thread(target=self.infer, args=(image_path, 'ais', patch_level, patch_size, slide_size,
            #                                              model_ais, label_dict_ais, seg_params, result_dict, self.kwargs))
            # ais_thread.start()
            #
            # hsil_thread = Thread(target=self.infer, args=(image_path, 'hsil', patch_level, patch_size, slide_size,
            #                                               model_hsil, label_dict_hsil, seg_params, result_dict, self.kwargs))
            # hsil_thread.start()

            results = {}
            result = self.infer(image_path, 'ais', patch_level, patch_size, slide_size, model_ais, label_dict_ais,
                                seg_params, self.kwargs)
            results.update(result)

            seg_params = self.def_seg_params.copy()
            result = self.infer(image_path, 'hsil', patch_level, patch_size, slide_size, model_hsil, label_dict_hsil,
                                seg_params, self.kwargs)
            results[list(result.keys())[0]].update(list(result.values())[0])
            q_ais_hsil_out.put(results)

    def process_result(self, file_name, results):
        ais_results = list(results[file_name].get('ais').values())[0].tolist()
        hsil_results = list(results[file_name].get('hsil').values())[0].tolist()
        # lsil_results = list(results.get('lsil').values())[0].tolist()

        for true_key in results[file_name].keys():
            infer_key = next(iter(results[file_name][true_key]))
            if true_key == infer_key == 'ais':
                self.results[file_name]['category'] = infer_key
                self.results[file_name]['boxes']['ais'] = ais_results
                self.results[file_name]['boxes']['hsil'] = hsil_results
                # self.results[data_dir]['boxes']['lsil'] = lsil_results
                return
            elif true_key == infer_key == 'hsil':
                self.results[file_name]['category'] = infer_key
                self.results[file_name]['boxes']['ais'] = []
                self.results[file_name]['boxes']['hsil'] = hsil_results
                # self.results[data_dir]['boxes']['lsil'] = lsil_results
                return
            elif true_key == infer_key == 'lsil':
                self.results[file_name]['category'] = infer_key
                self.results[file_name]['boxes']['ais'] = []
                self.results[file_name]['boxes']['hsil'] = []
                # self.results[data_dir]['boxes']['lsil'] = lsil_results
                return
        self.results[file_name]['category'] = 'normal'
        self.results[file_name]['boxes']['ais'] = []
        self.results[file_name]['boxes']['hsil'] = []
        # self.results[data_dir]['boxes']['lsil'] = []
        return

        # keys = list(results.keys())
        # values_keys = [list(i.keys())[0] for i in list(results.values())]
        # for true_key, infer_key in zip(keys, values_keys):
        #     if true_key == infer_key:
        #         self.results[data_dir]['category'] = infer_key
        #         self.results[data_dir]['boxes']['ais']['corr'] = results['ais'][values_keys[0]].tolist()
        #         self.results[data_dir]['boxes']['hsil']['corr'] = results['hsil'][values_keys[1]].tolist()
        #         self.results[data_dir]['boxes']['lsil']['corr'] = results['lsil'][values_keys[2]].tolist()
        #         return
        # self.results[data_dir]['category'] = 'normal'
        # self.results[data_dir]['boxes']['ais']['corr'] = results['ais'][values_keys[0]].tolist()
        # self.results[data_dir]['boxes']['hsil']['corr'] = results['hsil'][values_keys[1]].tolist()
        # self.results[data_dir]['boxes']['lsil']['corr'] = results['lsil'][values_keys[2]].tolist()
        # return

    def stage_send(self, data_dir, queues_in, queues_out):
        try:
            for file_name in os.listdir(data_dir):
                self.results["infer_id"] = file_name[:-4]
                self.results[file_name[:-4]] = self.orig_results
                for q_in in queues_in.values():
                    q_in.put(os.path.join(data_dir, file_name))

                for q_out in queues_out.values():
                    result = q_out.get()
                    print(result)
        except:
            pass

    def main(self, data_dir):
        self.queues_in = {f'q_{name}_in': Queue() for name in ['ais']}
        self.queues_out = {f'q_{name}_out': Queue() for name in ['ais']}

        processes = []
        for name, func in zip(['ais'], [self.process_ais]):
            q_in = self.queues_in[f'q_{name}_in']
            q_out = self.queues_out[f'q_{name}_out']
            p = Process(target=func, args=(q_in, q_out))
            processes.append(p)
            p.start()
            self.flag = False

        process_thread2 = Thread(target=self.stage_send, args=(data_dir, self.queues_in, self.queues_out))
        process_thread2.start()
        process_thread2.join()

        # self.results["infer_id"] = os.path.split(data_dir)[1][:-4]
        # self.results[os.path.split(data_dir)[1][:-4]] = self.orig_results
        #
        # # 发送图片地址给子进程
        # for q_in in self.queues_in.values():
        #     q_in.put(data_dir)
        #
        # results = {}
        # for q_out in self.queues_out.values():
        #     output = q_out.get()
        #     if not results:
        #         results.update(output)
        #     else:
        #         results[list(output.keys())[0]].update(list(output.values())[0])
        # print(results)
        # self.process_result(os.path.split(data_dir)[1][:-4], results)
        # print(self.results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='inference script')
    parser.add_argument('--data_dir', type=str, default=r"/home/bavon/datasets/wsi/ais/11/",
                        help='svs file')
    parser.add_argument('--save_path', type=str, default="heatmaps/output")
    parser.add_argument('--config_file', type=str, default="infer.yaml")
    parser.add_argument('--device', type=str, default="cuda:1")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    infer = Infer(args)
    infer.main(args.data_dir)
