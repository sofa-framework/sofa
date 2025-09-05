import Sofa
import matplotlib.pyplot as plt
import drjit as dr
from drjit.auto import Float, Array3f, TensorXf

def sdf(p: Array3f) -> Float:
    return dr.norm(p) - 1

def trace(o: Array3f, d: Array3f) -> Array3f:
    for i in range(10):
        o = dr.fma(d, sdf(o), o)
    return o

def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
    n = Array3f(
        sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
        sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
        sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps])
    ) / (2 * eps)
    return dr.maximum(0, dr.dot(n, l))

class RayMarching(Sofa.Core.Controller):
    def __init__(self, *args, **kwargs):
        Sofa.Core.Controller.__init__(self,*args, **kwargs)

    def onAnimateEndEvent(self, params):
        print("CALL RENDERING CODE")
        
        x = dr.linspace(Float, -1, 1, 1000)
        x, y = dr.meshgrid(x, x)

        p = trace(o=Array3f(0, 0, -2), d=dr.normalize(Array3f(x, y, 1)))
        sh = shade(p, l=Array3f(0, -1, -1))
        sh[sdf(p) > .1] = 0
        img = Array3f(.1, .1, .2) + Array3f(.4, .4, .2) * sh

        img_flat = dr.ravel(img)
        img_t = TensorXf(img_flat, shape=(1000, 1000, 3))

        #plt.imshow(img_t)
        #plt.savefig("dump.png")

def createScene(root):
    root.addObject(RayMarching(name="raymarching"))
