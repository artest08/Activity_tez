
from __future__ import print_function

import os
import sys
import glob
import argparse
from pipes import quote
from multiprocessing import Pool, current_process
import time

timelist = []
def run_optical_flow(vid_item):
    vid_path = vid_item[0]
    vid_id = vid_item[1]
    #vid_name = vid_path.split('/')[-1].split('.')[0]
    vid_name = vid_path.split('/')[-1][:-4]
    out_full_path = os.path.join(out_path, vid_name)
    try:
        os.mkdir(out_full_path)
    except OSError:
        pass

    # current = current_process()
    # dev_id = (int(current._identity[0]) - 1) % NUM_GPU
    image_path = '{}/img'.format(out_full_path)
    flow_x_path = '{}/flow_x'.format(out_full_path)
    flow_y_path = '{}/flow_y'.format(out_full_path)

    # temp_location = os.path.join('/tmp',vid_name+'.flv')
    # command = 'ffmpeg -i %s -s 340x256 %s' %(vid_path,temp_location)
    # os.system(command)
    dev_id = 0
    cmd = os.path.join(df_path + 'build/extract_gpu')+' -f={} -x={} -y={} -i={} -b=20 -t=1 -d={} -s=1 -o={} -w={} -h={}'.format(
        quote(vid_path), quote(flow_x_path), quote(flow_y_path), quote(image_path), dev_id, out_format, new_size[0], new_size[1])
    #print(cmd)
    start = time.time()
    os.system(cmd)
    end = time.time()
    sample_count = len(os.listdir(out_full_path))/3
    complexity = (end-start)/sample_count
    timelist.append(complexity)
    print('{} {} done'.format(vid_id, vid_name))
    sys.stdout.flush()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="extract optical flows")
    parser.add_argument("--src_dir", type=str, default='/media/esat/8234cf14-fc0e-441d-b606-0b7906d5d9c9/siser_dataset/window14',
                        help='path to the video data')
    parser.add_argument("--out_dir", type=str, default='/media/esat/6a7c4273-8106-47bc-b992-6760dfcea9a1/tsnCoffe/two-stream-pytorch/datasets/window_optic_flows3',
                        help='path to store frames and optical flow')
    parser.add_argument("--df_path", type=str, default='../../dense_flow/',
                        help='path to the dense_flow toolbox')

    parser.add_argument("--new_width", type=int, default=340, help='resize image width')
    parser.add_argument("--new_height", type=int, default=256, help='resize image height')

    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--num_gpu", type=int, default=2, help='number of GPU')
    parser.add_argument("--out_format", type=str, default='dir', choices=['dir','zip'],
                        help='path to the dense_flow toolbox')
    parser.add_argument("--ext", type=str, default='flv', choices=['flv','avi','mp4'],
                        help='video file extensions')

    args = parser.parse_args()

    out_path = args.out_dir
    src_path = args.src_dir
    num_worker = args.num_worker
    df_path = args.df_path
    out_format = args.out_format
    ext = args.ext
    new_size = (args.new_width, args.new_height)
    NUM_GPU = args.num_gpu

    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)

    vid_list = glob.glob(src_path+'/*/*.'+ext)
    print(len(vid_list))
    for vid_item in list(zip(vid_list, range(len(vid_list)))):
        run_optical_flow(vid_item)
    # pool = Pool(num_worker)
    # pool.map(run_optical_flow, list(zip(vid_list, range(len(vid_list)))))

