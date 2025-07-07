import glob
import os.path
import time
from threading import Thread

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm

from models import vae_models


def run_model(input_img):
    input_img = torch.cat(input_img)
    with torch.no_grad():
        output = model(input_img)[0]
        output = output * 0.5 + 0.5
        output = torch.permute(output * 255, (0, 2, 3, 1))
        output = output.cpu().numpy().astype(dtype=np.uint8)
        return output


def save_thread(image_data, out_file, in_file):
    image_data = Image.fromarray(image_data)
    image_data = image_data.resize(size=(1386, 1386), resample=Image.LANCZOS)
    image_data.save(out_file, dpi=(300, 300))


start_time = time.time()
np.set_printoptions(suppress=True)
assert torch.cuda.is_available(), '!!!you are not using cuda!!!'

batch_size = 32
exp_name = 'diffusion_color.v9.1-azi_fft.v9.1'
data_dir = '/playpen2/ttoha12/weapon/bi/bi_overall_color.v9.1-azi_fft.v9.1/'
ckpt_path = glob.glob(f'logs/ConditionalSiFall/{exp_name}/checkpoints/best*.ckpt')[0]
out_dir = f'data/{exp_name}_out/'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

model = vae_models['ConditionalSiFall'](name='ConditionalSiFall', in_channels=3, latent_dim=128)
ckpt = torch.load(ckpt_path)['state_dict']
new_state_dict = {}
for key, value in ckpt.items():
    new_key = key.replace('model.', '')
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)
model.cuda().eval()
print('loaded model', time.time() - start_time)

mode1_data = glob.glob(f'{data_dir}*/mode1/*/*.jpg')
total_images = len(mode1_data)
assert total_images > 0, 'dataset is empty'
print(f'{data_dir}: found {total_images} files')

os.makedirs(out_dir, exist_ok=True)
input_images, in_paths, out_paths = [], [], []
pbar = tqdm(total=total_images)
for img_idx, file_path1 in enumerate(mode1_data):
    file_path2 = file_path1.replace('/mode1/', '/mode2/')
    img_data1, img_data2 = Image.open(file_path1), Image.open(file_path2)
    img_data1, img_data2 = transform(img_data1), transform(img_data2)
    img_data = torch.cat((img_data1, img_data2), dim=0)
    img_data = img_data.unsqueeze(dim=0).cuda()

    input_images.append(img_data)
    in_paths.append(file_path1)

    out_path = os.path.basename(file_path1)
    out_path = out_path.replace('.jpg', '_rec.jpg')
    out_paths.append(out_path)

    if len(out_paths) > batch_size or img_idx == total_images - 1:
        output_images = run_model(input_images)

        thread_list = []
        for out_path, output_image, in_path in zip(out_paths, output_images, in_paths):
            out_path = f'{out_dir}{out_path}'
            t = Thread(target=save_thread, args=(output_image, out_path, in_path))
            t.start()
            thread_list.append(t)

        for t in thread_list:
            t.join()
            pbar.update(1)

        input_images.clear()
        out_paths.clear()

pbar.close()
