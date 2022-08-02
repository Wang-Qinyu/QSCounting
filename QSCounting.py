import os

import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont

from skimage import measure, color
# import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from easydict import EasyDict as edict
import yaml

# from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

def getParam(path=r'./configs_Counting.yaml'):
    """_summary_

    Args:
        path (regexp, optional): _description_. Defaults to r'./configs.yaml'.

    Returns:
        dict: _description_
    """
    with open(path, 'r', encoding='utf-8') as fp:
        cfg = yaml.safe_load(fp)
    return edict(cfg)


#! LOAD YAML FILE AND PRINT
cfg = getParam()
table = Table(show_header=True, header_style="bold magenta")
table.add_column("Key")
table.add_column("Value")

for key, value in cfg.items():
    table.add_row(key,str(value))
    # console.print(key, ':', value)
    # console.print("{} : {}" key + ':' + value,style = 'bold blue')
console.rule("[bold red]Parameters")
console.print(table)
print('\n')

try:
    FONT = ImageFont.truetype('SimHei.ttf', 150)
except IOError:
    FONT = ImageFont.load_default()

input_dir = cfg.INPUT_PATH
output_dir = cfg.OUTPUT_PATH

input_images = os.listdir(input_dir)

if not os.path.isdir(cfg.OUTPUT_PATH):
    os.mkdir(cfg.OUTPUT_PATH)

if cfg.VISUALIZATION:
    if not os.path.isdir(cfg.DRAW_PATH):
        os.mkdir(cfg.DRAW_PATH)

if cfg.SCATTER_PATH:
    if not os.path.isdir(cfg.SCATTER_PATH):
        os.mkdir(cfg.SCATTER_PATH)

total = []

nclass = len(cfg.DURATIONS) + 1 #! one for others
results_cols = []
for i in range(nclass):
    results_cols.append([])

#! 
# console.rule("[bold red]Load data")
# console.print('20 images retrieved, sorted by 5 radius scales\n',style = 'bold red')

image_id = 0
console.rule("[bold red]{} images retrieved, sorted by {} radius scales".format(len(input_images),nclass))


detail_info = []

for image in track(input_images, description='Processing...'):
    img_name = image.split('.')[0]
    img = Image.open(os.path.join(input_dir, image))
    draw = ImageDraw.Draw(img)

    mask = np.asarray(img.convert('L'))
    
    if cfg.VISUALIZATION:
        Image.fromarray(mask).save(
            './{}/{}_gray.png'.format(cfg.DRAW_PATH, img_name))  # ! save gray image
    mask = mask > cfg.THRESHOLD  # ! threshold
    mask = mask * 255
    mask = 255 - mask
    
    if cfg.VISUALIZATION:
        Image.fromarray(np.uint8(mask)).save(
            './{}/{}_binarization.png'.format(cfg.DRAW_PATH, img_name))

    label = measure.label(mask)
    a = measure.regionprops(label)

    idx = 0

    temp = []
    
    raw_imgs = []
    draws = []
    
    
    for i in range(nclass):
        temp.append(0)
        # results_cols.append([])
        #! for draw
        if cfg.VISUALIZATION:
            raw_img = Image.new(mode='RGB', size=(mask.shape[1], mask.shape[0]))
            raw_imgs.append(raw_img)
            draws.append(ImageDraw.Draw(raw_img))
    if cfg.VISUALIZATION:
        rgb = Image.new(mode='RGB', size=(mask.shape[1], mask.shape[0]))
        draw_rgb = ImageDraw.Draw(rgb)

    if cfg.SCATTER:
        radius = []
        areas = []
    for (j, i) in enumerate(a):
        if cfg.AREA_THRESHOLD < i.area:
            min_row, min_col, max_row, max_col = i.bbox
            h, w = max_row-min_row, max_col-min_col
            if cfg.RATIO[0] < w/h < cfg.RATIO[1]:
                if cfg.SCATTER:
                    radius.append(w)
                    areas.append(i.area)
                idx += 1
                detail_info.append([ image,w,i.area  ])
                # ! Insert a for loops,
                flag = 0
                for idx in range(nclass-1):
                    # print(cfg.DURATIONS[idx],type(print(cfg.DURATIONS[idx][0])))
                    if cfg.DURATIONS[idx][0] < w <= cfg.DURATIONS[idx][1]:
                        
                        temp[idx] += 1
                        color = tuple(cfg.COLORS[idx])
                        
                        if cfg.VISUALIZATION:
                            draws[idx].rectangle([min_col, min_row, max_col, max_row],
                                        width=cfg.BBOX_WIDTH, fill=color)
                            draw_rgb.rectangle([min_col, min_row, max_col, max_row],
                                       width=cfg.BBOX_WIDTH, fill=color)
                        draw.rectangle([min_col, min_row, max_col, max_row],
                                       width=cfg.BBOX_WIDTH, outline=color)
                        flag = 1
                        break
                if flag == 0:  #! OTHERS
                    temp[-1] += 1
                    color = tuple(cfg.COLORS[-1])
                    if cfg.VISUALIZATION:
                        draws[idx].rectangle([min_col, min_row, max_col, max_row],
                                        width=cfg.BBOX_WIDTH, fill=color)
                        draw_rgb.rectangle([min_col, min_row, max_col, max_row],
                                        width=cfg.BBOX_WIDTH, fill=color)
                    draw.rectangle([min_col, min_row, max_col, max_row],
                                       width=cfg.BBOX_WIDTH, outline=color)
    
    #! draw scatter 
    if cfg.SCATTER:
        plt.cla()
        rng = np.random.RandomState(0)
        colors = rng.rand(len(radius))
        plt.scatter(radius, areas, c=colors,  alpha=0.3, cmap='viridis')
        plt.savefig(os.path.join(cfg.SCATTER_PATH,img_name+'.svg'))
    
    #! save all types image

    if cfg.VISUALIZATION:
        for idx in range(nclass-1):
            raw_imgs[idx].save('./{}/{}_class{}.png'.format(cfg.DRAW_PATH, img_name,idx))

    total.append(sum(temp))
    
    if cfg.VISUALIZATION:
        rgb.save('./{}/{}_all.png'.format(cfg.DRAW_PATH, img_name))

    for idx, value in enumerate(temp):
        results_cols[idx].append(value)
    img.save(os.path.join(cfg.OUTPUT_PATH, image))


goal_text = {}
goal_text.setdefault('File_name',input_images)

for i in range(nclass-1):
    goal_text.setdefault('R={}'.format(cfg.DURATIONS[i]),results_cols[i])
goal_text.setdefault('R-{}'.format('Others'),results_cols[i+1])
goal_text.setdefault('{}'.format('Total'),total)


dataframe = DataFrame(goal_text)
dataframe.to_csv(cfg.EXCEL_PATH, index=False, sep=',')

#! write detail information to .csv
df_detail = DataFrame(columns=['imageNames','Radius','Areas'])
for idx,value in enumerate(detail_info) :
    df_detail.loc[idx+1] = value
df_detail.to_csv(cfg.SIZE_INFORMATION, index=False, sep=',')
    

print('\n')
console.rule("[bold red]Results")
console.print('The statistical results have been saved in file {}, and the visualization images are saved in {}\n'.format(cfg.EXCEL_PATH,cfg.OUTPUT_PATH))

console.print('Press the [bold red]Enter[/bold red] key to exit.')
input()
