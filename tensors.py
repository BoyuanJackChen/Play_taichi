# fractal.py

# --- Portability ---
import taichi as ti
# Run on GPU by default
ti.init(debug=False, arch=ti.gpu)
# # Run on NVIDIA GPU, CUDA required
# ti.init(arch=ti.cuda)
# # Run on GPU, with the OpenGL backend
# ti.init(arch=ti.opengl)
# # Run on GPU, with the Apple Metal backend, if you are on OS X
# ti.init(arch=ti.metal)
# # Run on CPU (default)
# ti.init(arch=ti.cpu)

# 0.6.7版本定义的时候必须把variables放在一起。
a = ti.var(dt=ti.f32, shape=(42,63))    # A tensor of 42x 63 scalars
b = ti.Vector(3, dt=ti.f32, shape=4)    # A tensor of 4 x 4D vectors
C = ti.Matrix(2,2, dt=ti.f32, shape=(3,5))   # A tensor of 3x5 2x2 matrices
loss = ti.var(dt=ti.f32, shape=())      # a 0-D tensor of a single scalar
n=320
pixels = ti.var(dt=ti.f32, shape=(n*2, n))

a[3,4]=1
print(f"a[3,4] = {a[3,4]}")

b[2] = [6,7,8]
print("b[0] = ", b[0][0], b[0][1], b[0][2])  # print(b[0]) is not yet supported

loss[None] = 3
print(loss[None])

@ti.kernel
def paint(t: ti.f32):
  for i, j in pixels:   # Parallized over all pixels. 包在里面一层的话就不行了。
    pixels[i,j] = i*0.001 + j*0.002 + t

paint(0.3)
print("aha")