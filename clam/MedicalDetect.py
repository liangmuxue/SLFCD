import stat
import sys

sys.path.append("/home/bavon/project/SLFCD/SLFCD/extras/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/project/")
sys.path.append("/home/bavon/project/SLFCD/SLFCD/")
import os
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
import urllib.request

import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

from multiprocessing import Process, Queue
import json
import traceback
from threading import Thread
import requests
from flask import Flask, request, jsonify

import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='inference script')
parser.add_argument('--data_dir', type=str, default=r'D:\project\SLFCD\dataset\ais\data\1-CG23_18831_01.svs',
                    help='svs file')
parser.add_argument('--save_path', type=str, default="heatmaps/output")
parser.add_argument('--root_url', type=str, default="192.168.0.98")
parser.add_argument('--root_port', type=int, default=8088)
parser.add_argument('--receive_port', type=str, default=8091)
parser.add_argument('--config_file', type=str, default="infer.yaml")
parser.add_argument('--device', type=str, default="cuda:1")
args = parser.parse_args()
root_url, root_port, receive_port = args.root_url, args.root_port, args.receive_port

app = Flask(__name__)
svs_input_queue = Queue(6)
svs_output_queue = Queue(6)
stage_send_queue_multi = Queue(6)
stage_flag = {'ais': 1, 'hsil': 2, 'lsil': 3, 1: 'ais模型推理完成', 2: "hsil模型推理完成", 3: "lsil模型推理完成"}
temp_results = {}
global infer


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
                       "blank_canvas": self.heatmap_args.blank_canvas, "thresh":-1,
                       "patch_size": 256, "convert_to_percentiles": True}

        self.custom_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224)])

        self.results = {}
        self.orig_results = {
            "category": "normal",
            "size": [],
            "boxes": {
                "ais": [],
                "hsil": [],
                "lsil": []
            }
        }

        self.flag = True

    def load_model(self, model_args):
        # 第一阶段
        model_one = CoolSystem.load_from_checkpoint(model_args['slf_ckpt_path'])
        model_one.to(self.model_args.device)
        model_one.eval()
        print('first ckpt path: {}'.format(model_args['slf_ckpt_path']))

        # 第二阶段
        instance_loss_fn = SmoothTop1SVM(n_classes=self.model_args.n_classes)
        model_two = CLAM_SB(**self.model_dict, instance_loss_fn=instance_loss_fn)
        model_two.load_state_dict(torch.load(model_args['ckpt_path'], map_location=self.model_args.device))
        model_two.to(self.model_args.device)
        model_two.eval()
        print('second ckpt path: {}'.format(model_args['ckpt_path']))
        return model_one, model_two

    def infer(self, svs_content, name, patch_level, patch_size, step_size, model,
              label_dict, seg_params, stage_send_queue_multi, kwargs):
        save_path, svs_path, svs_id = svs_content[0], svs_content[1], svs_content[2]
        slide_id = os.path.split(svs_path)[-1][:-4]

        # ----------------------- save ------------------------------------------- 
        root_path = os.path.join(save_path, svs_id)       
        if not os.path.exists(root_path):
            os.makedirs(root_path, exist_ok=True)
            os.chmod(root_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
        slide_save_dir = os.path.join(save_path, svs_id, name)
        os.makedirs(slide_save_dir, exist_ok=True)
        os.chmod(slide_save_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        pt_save_path = os.path.join(slide_save_dir, svs_id + '.pt')
        h5_save_path_two = os.path.join(slide_save_dir, svs_id + '_two.h5')

        segment_mask_path = os.path.join(slide_save_dir, svs_id + '.jpg')
        heatmap_path = os.path.join(slide_save_dir, svs_id + '_blockmap.png')
        marked_image_save_path = os.path.join(slide_save_dir, svs_id + '_marked_original.png')

        sample_save_dir = os.path.join(slide_save_dir, 'smallPic')
        os.makedirs(sample_save_dir, exist_ok=True)
        os.chmod(sample_save_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

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

        WSI_object = WholeSlideImage(svs_path)
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
                                pin_memory=True, collate_fn=collate_features)

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

            features = features.to(self.model_args.device)
            _, Y_prob, Y_hat, A, _ = model[1](features)
            Y_hat = Y_hat.item()
            Y_prob = Y_prob[0].detach().tolist()[Y_hat]
            scores = A.cpu().detach().numpy()

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
                    total=self.sample_args.samples["k"], desc='save ' + self.sample_args.samples['name']):
                # 截取注意力区域并保存
                patch = WSI_object.wsi.read_region(tuple(s_coord), patch_level, patch_size).convert('RGB')
                patch.save(os.path.join(sample_save_dir, '[{},{}].jpg'.format(s_coord[0], s_coord[1])))

            # kwargs['patch_size'] = patch_size
            # heatmap, original_image = WSI_object.visHeatmap(scores=scores, coords=coords, **kwargs)
            #
            # heatmap.save(heatmap_path.replace('.png', f'_{tag}.png'))
            # original_image.save(marked_image_save_path.replace('.png', f'_{tag}.png'))

            if not stage_send_queue_multi.empty():
                stage_send_queue_multi.get()
            print(f"stage {stage_flag[name]} {stage_flag[stage_flag[name]]}")
            stage_send_queue_multi.put({'sampleId': svs_id, 'state': stage_flag[name]})
            return {name: {label_dict[Y_hat]: sample_results['sampled_coords'], "size": WSI_object.wsi.level_dimensions[patch_level]}}
        else:
            if not stage_send_queue_multi.empty():
                stage_send_queue_multi.get()
            print(f"stage {stage_flag[name]} {stage_flag[stage_flag[name]]}")
            stage_send_queue_multi.put({'sampleId': svs_id, 'state': stage_flag[name]})
            return {name: {'样本模糊无效!!!'}}

    def process_ais(self, stage_send_queue_multi, q_ais_in, q_ais_out):
        # ----------------------------- ais -----------------------------
        model = self.load_model(self.model_args.ais)
        patch_level = self.ais_params.patch_level
        patch_size = tuple([self.ais_params.patch_size for i in range(2)])
        slide_size = tuple([self.ais_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.ais_params.label_dict.values(), self.ais_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            svs_content = q_ais_in.get()
            seg_params = self.def_seg_params.copy()

            result = self.infer(svs_content, 'ais', patch_level, patch_size, slide_size, model, label_dict,
                                seg_params, stage_send_queue_multi, self.kwargs)
            q_ais_out.put(result)

    def process_hsil(self, stage_send_queue_multi, q_hsil_in, q_hsil_out):
        # ----------------------------- hsil ------------------------------
        model = self.load_model(self.model_args.hsil)
        patch_level = self.hsil_params.patch_level
        patch_size = tuple([self.hsil_params.patch_size for i in range(2)])
        slide_size = tuple([self.hsil_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.hsil_params.label_dict.values(), self.hsil_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            svs_content = q_hsil_in.get()
            seg_params = self.def_seg_params.copy()

            result = self.infer(svs_content, 'hsil', patch_level, patch_size, slide_size, model, label_dict,
                                seg_params, stage_send_queue_multi, self.kwargs)
            q_hsil_out.put(result)

    def process_lsil(self, stage_send_queue_multi, q_lsil_in, q_lsil_out):
        model = self.load_model(self.model_args.lsil)
        patch_level = self.lsil_params.patch_level
        patch_size = tuple([self.lsil_params.patch_size for i in range(2)])
        slide_size = tuple([self.lsil_params.slide_size for i in range(2)])
        label_dict = dict(zip(self.lsil_params.label_dict.values(), self.lsil_params.label_dict.keys()))

        self.kwargs.update({"vis_level": patch_level, "patch_size": patch_size})
        self.def_seg_params['seg_level'] = patch_level

        while True:
            svs_content = q_lsil_in.get()
            seg_params = self.def_seg_params.copy()

            result = self.infer(svs_content, 'lsil', patch_level, patch_size, slide_size, model, label_dict,
                                seg_params, stage_send_queue_multi, self.kwargs)
            q_lsil_out.put(result)

    def process_ais_hsil(self, stage_send_queue_multi, q_ais_hsil_in, q_ais_hsil_out):
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
            svs_content = q_ais_hsil_in.get()

            results = {}
            seg_params = self.def_seg_params.copy()
            result_ais = self.infer(svs_content, 'ais', patch_level, patch_size, slide_size, model_ais, label_dict_ais,
                                    seg_params, stage_send_queue_multi, self.kwargs)
            results.update(result_ais)

            seg_params = self.def_seg_params.copy()
            result_hsil = self.infer(svs_content, 'hsil', patch_level, patch_size, slide_size, model_hsil,
                                     label_dict_hsil,
                                     seg_params, stage_send_queue_multi, self.kwargs)
            results.update(result_hsil)

            q_ais_hsil_out.put(results)

    def process_result(self, file_id, results):
        ais_results = list(results.get('ais').values())[0].tolist()
        hsil_results = list(results.get('hsil').values())[0].tolist()
        lsil_results = list(results.get('lsil').values())[0].tolist()
        size = list(list(results.get('lsil').values())[1])
        
        for true_key in results.keys():
            infer_key = next(iter(results[true_key]))
            if true_key == infer_key == 'ais':
                self.results[file_id]['category'] = infer_key
                self.results[file_id]['size'] = size
                self.results[file_id]['boxes']['ais'] = ais_results
                self.results[file_id]['boxes']['hsil'] = hsil_results
                self.results[file_id]['boxes']['lsil'] = lsil_results
                return
            elif true_key == infer_key == 'hsil':
                self.results[file_id]['category'] = infer_key
                self.results[file_id]['size'] = size
                self.results[file_id]['boxes']['ais'] = []
                self.results[file_id]['boxes']['hsil'] = hsil_results
                self.results[file_id]['boxes']['lsil'] = lsil_results
                return
            elif true_key == infer_key == 'lsil':
                self.results[file_id]['category'] = infer_key
                self.results[file_id]['size'] = size
                self.results[file_id]['boxes']['ais'] = []
                self.results[file_id]['boxes']['hsil'] = []
                self.results[file_id]['boxes']['lsil'] = lsil_results
                return
        
        self.results[file_id]['category'] = 'normal'
        self.results[file_id]['size'] = size
        self.results[file_id]['boxes']['ais'] = []
        self.results[file_id]['boxes']['hsil'] = []
        self.results[file_id]['boxes']['lsil'] = []
        return
    
    def main(self, save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue):
        if self.flag:
            self.queues_in = {f'q_{name}_in': Queue() for name in ['ais_hsil', 'lsil']}
            self.queues_out = {f'q_{name}_out': Queue() for name in ['ais_hsil', 'lsil']}

            # 创建进程列表
            processes = []
            for name, func in zip(['ais_hsil', 'lsil'], [self.process_ais_hsil, self.process_lsil]):
                q_in = self.queues_in[f'q_{name}_in']
                q_out = self.queues_out[f'q_{name}_out']
                p = Process(target=func, args=(stage_send_queue_multi, q_in, q_out))
                processes.append(p)
                p.start()
            self.flag = False

        self.results["infer_id"] = file_id
        self.results[file_id] = self.orig_results

        # 发送图片地址给子进程
        for q_in in self.queues_in.values():
            q_in.put([save_url, file_path, file_id])

        # 从队列接收结果
        results = {}
        for q_out in self.queues_out.values():
            result = q_out.get()
            results.update(result)

        try:
            self.process_result(file_id, results)

            if not stage_send_queue_multi.empty():
                stage_send_queue_multi.get()
            stage_send_queue_multi.put({'sampleId': file_id, 'state': 4})
            svs_output_queue.put(self.results)
            print("stage 4 推理结果处理完成，样本有效")
        except:
            if not stage_send_queue_multi.empty():
                stage_send_queue_multi.get()
            print("stage 5 推理结果处理完成，样本模糊无效")
            stage_send_queue_multi.put({'sampleId': file_id, 'state': 5})


def Multi_threaded_download(file_url, file_id, save_path, stage_send_queue_multi, svs_input_queue):
    response = requests.get(file_url, stream=True)
    response.raise_for_status()
    if response.status_code == 200:
        # 使用tqdm包装迭代器，如果文件大小未知，则total=None
        progress = tqdm(unit='B', unit_scale=True, desc='Downloading')

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                # 过滤掉保持连接的chunk
                if chunk:
                    f.write(chunk)
                    # 更新进度条
                    progress.update(len(chunk))
                    # 完成下载后关闭进度条
        progress.close()

        stage_send_queue_multi.put(f"{file_id} 下载成功！！")
    else:
        stage_send_queue_multi.put(f"{file_id} 下载失败！！！")


def process_svs_file(save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue):
    if not stage_send_queue_multi.empty():
        stage_send_queue_multi.get()
    print("stage 0 文件正在分析")
    stage_send_queue_multi.put({'sampleId': file_id, 'state': 0})
    infer.main(save_url, file_path, file_id, stage_send_queue_multi, svs_output_queue)


def process_dzi_file(save_url, file_path, file_id):
    data = {'saveUrl': save_url, 'svs_path': file_path, "sampleId": file_id}
    response = requests.post(f'http://192.168.0.179:8088/svs2dzi', json=data)
    print("send_dzi: ", response.json())


def stage_send(stage_send_queue_multi):
    while True:
        try:
            if not stage_send_queue_multi.empty():
                results = stage_send_queue_multi.get()
                print("stage: ", results)
                req = urllib.request.Request(url=f'http://{root_url}:{receive_port}/system/job/stageSend',
                                             data=json.dumps(results).encode('utf-8'),
                                             headers={"Content-Type": "application/json"})
                res = urllib.request.urlopen(req)
                res = res.read().decode("utf-8")
                res = json.loads(res)
                print("status_code: ", res)
        except Exception as e:
            # 打印异常信息
            print('error: ', e)


@app.route('/upload_svs_file', methods=['POST'])
def download_SVS():
    try:
        data = request.get_json()
        save_url = data.get('saveUrl')
        file_url = data.get('svs_path')
        file_id = data.get('sampleId')

        print(save_url, file_url, file_id)

        if not os.path.exists(file_url):
            print("upload 1 文件不存在")
            return jsonify({'error': 1}), 404

        if not svs_output_queue.empty():
            results = svs_output_queue.get()
            if file_id in results:
                if svs_output_queue.empty():
                    svs_output_queue.put(results)
                print("upload 1 当前已有文件正在处理")
                return jsonify({'error': 2}), 200

        # 启动线程处理文件
        process_infer_thread = Thread(target=process_svs_file, args=(save_url, file_url, file_id, stage_send_queue_multi, svs_output_queue))
        process_infer_thread.start()
        # 等待处理完成
        # process_thread.join()  

        process_dzi_thread = Thread(target=process_dzi_file, args=(save_url, file_url, file_id))
        process_dzi_thread.start()

        stage_thread = Thread(target=stage_send, args=(stage_send_queue_multi,))
        stage_thread.start()

        # 文件接收成功，返回提示信息
        print("upload 0 文件上传成功")
        return jsonify({'error': 0}), 200
    except Exception as e:
        print("upload 3 当前状态异常")
        return jsonify({'error': 3, 'message': e}), 500


@app.route('/receive_svs_results', methods=['POST'])
def process_and_send_results():
    try:
        data = request.get_json()
        file_id = data.get('sampleId')
        print("file_id: ", file_id)
        if not svs_output_queue.empty():
            results = svs_output_queue.get()
            print("results: ", results)
            return jsonify(results[file_id]), 200
    except Exception as e:
        # 可以在这里记录到日志文件或其他地方
        return jsonify({'error': 'An error occurred', 'message': str(e)}), 500


if __name__ == "__main__":
    infer = Infer(args)
    app.run(root_url, port=root_port, debug=True)
