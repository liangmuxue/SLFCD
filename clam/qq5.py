import requests

root_url = 'http://192.168.0.179'
root_port = 8088
send_url = f'{root_url}:{root_port}/svs2dzi'


def main_loop():
    while True:
        try:
            # D:/project/SLFCD1/clam/create_heatmaps.py,111
            # D:/project/SLFCD1/clam/create_patches_fp.py,222
            # D:/project/SLFCD/dataset/hsil/11/2-CG23_12974_02.svs,11
            # D:/project/SLFCD/dataset/normal/data/10-CG23_19800_02.svs,22
            # /home/bavon/datasets/wsi/ais/11/1-CG23_18831_01.svs,3
            path = input("请输入路径： ")
            if not path:
                print("路径不能为空，请输入有效的路径。")
                continue

            # 发送
            data = {'saveUrl': '/home/bavon/SLFCD/', 'svs_path': path.split(",")[0], "sampleId": path.split(",")[1]}
            response = requests.post(send_url, json=data)
            print("send_url", response.json())

        except Exception as e:
            print(f"Error occurred: {e}")
            break


if __name__ == '__main__':
    main_loop()
