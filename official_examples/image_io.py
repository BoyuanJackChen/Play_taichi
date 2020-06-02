import taichi as ti
import os
ti.init(arch=ti.cuda)
pixel = ti.var(ti.u8, shape=(512, 512, 3))

@ti.kernel
def paint():
    for I in ti.grouped(pixel):
        pixel[I] = ti.random() * 255
# 'ti.grouped can only be used inside Taichi kernels'

paint()
pixel = pixel.to_numpy()
ti.imshow(pixel, 'Random Generated')
# Interesting - if I close the window the code halts here. I wonder why.
print("aha")

for ext in ['bmp', 'png', 'jpg']:
    fn = 'taichi-example-random-img.' + ext
    print(os.getcwd())
    ti.imwrite(pixel, fn)
    pixel_r = ti.imread(fn)
    if ext != 'jpg':
        assert (pixel_r == pixel).all()
    else:
        ti.imshow(pixel_r, 'JPEG Read Result')
    # os.remove(fn)
