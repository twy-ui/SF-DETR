import warnings, os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
warnings.filterwarnings('ignore')
from ultralytics import RTDETR


if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/SF-DETR.yaml')
    # model.load('') # loading pretrain weights
    model.train(data='VisDrone2019.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=4, # batchsize
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', #
                # resume='', # last.pt path
                patience=0, # 设置0代表不早提供，设置30代表精度持续30epoch没有比之前最高的高就早停
                project='runs/train',
                name='SF-DETR',
                )

