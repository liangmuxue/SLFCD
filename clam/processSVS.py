import argparse
import os
import shutil
import stat
import pyvips
from flask import Flask, request, jsonify

app = Flask(__name__)

parser = argparse.ArgumentParser(description='svs2dzi')
parser.add_argument('--root_url', type=str, default="192.168.0.179")
parser.add_argument('--root_port', type=int, default=8088)
args = parser.parse_args()
root_url, root_port = args.root_url, args.root_port


# 如果需要递归设置所有子目录和文件的权限为 777
def set_permissions_recursive(path):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        for f in files:
            os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


@app.route('/svs2dzi', methods=['POST'])
def svs2dzi():
    try:
        data = request.get_json()
        save_url = data.get('saveUrl')
        file_url = data.get('svs_path')
        file_id = data.get('sampleId')

        print(save_url, file_url, file_id)

        # 加载svs
        img = pyvips.Image.new_from_file("./" + file_url[file_url.index("upload"):], access='sequential')

        # 保存目录
        if os.path.exists('./image/' + file_id):
            root_svs = './image/' + file_id + "/"
            root_path = [root_svs + i for i in os.listdir(root_svs) if file_id in i]
            for i in root_path:
                if os.path.isfile(i):
                    os.remove(i)
                else:
                    for root, dirs, files in os.walk(i, topdown=False):
                        for name in files:
                            file_path = os.path.join(root, name)
                            os.remove(file_path)
                        for name in dirs:
                            dir_path = os.path.join(root, name)
                            os.rmdir(dir_path)
                    os.rmdir(i)
        else:
            os.makedirs('./image/' + file_id, exist_ok=True)
            # 设置目录权限为 777
            os.chmod('./image/' + file_id, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 生成dzi文件
        img.dzsave('./image/' + file_id + '/' + file_id)
        return jsonify({'error': 0}), 200
    except Exception as e:
        return jsonify({'error': 3, 'message': e}), 500


if __name__ == "__main__":
    app.run(root_url, port=root_port, debug=True)
