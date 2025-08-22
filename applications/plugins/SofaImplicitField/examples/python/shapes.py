import Sofa
from SofaImplicitField import ScalarField
import numpy
import jax 
import jax.numpy as jnp
import drjit
import math

class Sphere(ScalarField):
    def __init__(self, *args, **kwargs):
        ScalarField.__init__(self, *args, **kwargs)
        
        self.addData("center", type="Vec3d",value=kwargs.get("center", [0.0,0.0,0.0]), default=[0.0,0.0,0.0], help="center of the sphere", group="Geometry")
        self.addData("radius", type="double",value=kwargs.get("radius", 1.0), default=1, help="radius of the sphere", group="Geometry")

        print("DRJIT Version: ", drjit.__version__)

        self._center = jnp.array(self.center.value)
        self._radius = jnp.array(self.radius.value)
        print("PARSING +=============================== ")
        print(" CENTER ", self._center)
        print(" RADIUS ", self._radius)
        
        self.fast_getValues = jax.vmap( self.getValueJax, in_axes=[0,None,None] )
        self.fast_getValue = jax.jit(self.getValueJax)

        #print(jax.make_jaxpr(self.fast_getValues)(jnp.array([0,0,0]),self._center, self._radius))
        #print(jax.make_jaxpr(self.fast_getValue)(jnp.array([0,0,0]), self._center, self._radius))

        self.fast_getValues(jnp.array([[0,0,0]]), self._center, self._radius).block_until_ready()
        self.fast_getValue(jnp.array([0,0,0]), self._center, self._radius).block_until_ready()

        Sofa.msg_warning(self, f"Number of devices {jax.devices()}")

    @staticmethod 
    def getValuesJax(positions, center, radius):
        return jnp.sqrt( jnp.sum( (center - positions )**2, axis=1 ) ) - radius

    @staticmethod
    def getValueJax(position, center, radius):
        return jnp.sqrt( jnp.sum( (center - position )**2 ) ) - radius

    def getValue(self, position):
        x,y,z = position
        position = jnp.array([x,y,z])
        tt = self.fast_getValue(position, self._center, self._radius) 
        return tt

    @staticmethod 
    def getValueNumba(self, position):
        x,y,z = position
        return numpy.sqrt( numpy.sum((self.center.value - numpy.array([x,y,z]))**2) ) - self.radius.value 
    
    @staticmethod
    def getValuesNumba(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""
        out_values[:] = numpy.sqrt( numpy.sum( numpy.array([self.center.value] - positions)**2, axis=1 ) ) - numpy.array([self.radius.value])
    
    def getValues(self, positions, out_values):
        """This version of the overrides get numpy array as input and output"""        
        t = self.fast_getValues(positions, self._center, self._radius) 
        out_values[:] = t 