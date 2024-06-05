from .transcp import build_vgmodel #这个是论文重构的model，精度对齐中

def build_model(args):
    return build_vgmodel(args)
