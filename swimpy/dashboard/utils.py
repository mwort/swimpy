import os.path as osp

from dash import html


def static_path(path):
    return osp.join(osp.dirname(__file__), "static", path)


def local_img(path, imgtag=True, **kw):
    import base64
    encoded_image = base64.b64encode(open(path, 'rb').read()).decode()
    imgsrc = 'data:image/{0};base64,{1}'.format(osp.splitext(path)[1], encoded_image)
    return html.Img(src=imgsrc, **kw) if imgtag else imgsrc
