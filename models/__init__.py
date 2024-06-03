# from .VLTVG import build_vgmodel
# from .VLTVG_VD import build_vgmodel
# from .VLTVG_VD_Enhance import build_vgmodel
# from .VLTVG_VD_Enhance_ViT import build_vgmodel #这个是论文正式使用的model，切记
# from .VLTVG_VD_Enhance_ViT_ablation import build_vgmodel

from .transcp import build_vgmodel #这个是论文重构的model，精度对齐中

def build_model(args):
    return build_vgmodel(args)
