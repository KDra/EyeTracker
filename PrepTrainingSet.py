#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
from shutil import copy
from PIL import Image, ImageOps, ImageFilter
from numpy.random import choice


root = Path('images/')
fl = list(root.rglob('*.jpg'))


classes = set([f.stem.split('_')[-1] for f in fl])


new_root = Path(f'{root.stem}_train/')
for c in classes:
    if c in ['up', 'away', 'blink', 'unknown']:
        c = 'other'
    (new_root/c).mkdir(exist_ok=True, parents=True)


for f in fl:
    cl = f.stem.split('_')[-1]
    if cl in ['up', 'away', 'blink', 'unknown']:
        cl = 'other'
    copy(f, new_root/cl/'_'.join(f.parts[1::3]))
classes = set(new_root.glob('*/'))


mirror_classes = [new_root/i for i in ['other', 'center']]


for c in mirror_classes:
    for i in c.glob('*.jpg'):
        im = Image.open(i)
        im_mirror = ImageOps.mirror(im)
        im_mirror.save(str(i).replace('.jpg', '_mir.jpg'), quality=95)


for c in mirror_classes:
    for i in c.glob('*.jpg'):
        im = Image.open(i)
        im_mirror = im.filter(ImageFilter.UnsharpMask(radius=3, percent=120, threshold=2))
        im_mirror.save(str(i).replace('.jpg', '2.jpg'), quality=95)


for c in classes:
    img = choice(list(c.glob('*jpg')), 2000, replace=False)
    for i in img:
        im = Image.open(i)
        im = im.filter(ImageFilter.BLUR)
        im.save(str(i).replace('.jpg', '_blur.jpg'), quality=95)


for c in classes:
    img = choice(list(c.glob('*jpg')), 2000, replace=False)
    for i in img:
        im = Image.open(i)
        im = im.filter(ImageFilter.SMOOTH)
        im.save(str(i).replace('.jpg', '_smooth.jpg'), quality=95)


for c in classes:
    img = choice(list(c.glob('*jpg')), 2000, replace=False)
    for i in img:
        im = Image.open(i)
        im = im.filter(ImageFilter.UnsharpMask)
        im.save(str(i).replace('.jpg', '_unsharp.jpg'), quality=95)


for c in classes:
    img = choice(list(c.glob('*jpg')), 2000, replace=False)
    for i in img:
        im = Image.open(i)
        im = im.convert('L')
        im.save(str(i).replace('.jpg', '_gray.jpg'), quality=95)


classes = list(classes)
im_num = {}
for c in classes:
    im_num[c] = len(list(c.glob('*jpg')))
n = min(im_num.values())


for c in classes:
    sz = im_num[c] - n
    img = choice(list(c.glob('*jpg')), sz, replace=False)
    for i in img:
        i.unlink()


im_num_new = {}
for c in classes:
    im_num_new[c] = len(list(c.glob('*jpg')))
im_num_new


import tarfile
with tarfile.open(f'{new_root.stem}.tgz', "w:gz") as tar:
    tar.add(new_root)



