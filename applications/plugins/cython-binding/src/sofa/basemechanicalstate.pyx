from sofa.base cimport _Base, Base
from sofa.baseobject cimport _BaseObject, BaseObject
#from basemechanicalstate cimport BaseMechanicalState as _BaseMechanicalState 

cdef extern from "": 
        _BaseMechanicalState* dynamic_cast_basemechanicalobject "dynamic_cast< sofa::core::behavior::BaseMechanicalState* >" (_Base*) except NULL

cdef class BaseMechanicalState(BaseObject):
        """In Sofa a BaseMechanicalState is a component that stores the mechanical DOFs.
           The BaseMechanicalState inherits from BaseObject (and thus Base) and it is possible to downcast a 
           BaseObject (resp. a Base) to as BaseMechanicalState.
           
                Example to downcast from a BaseObject to a BaseMechanicalState:
                        aBm = BaseMechanicalState(aBaseObject) 
        
                Example to create such an object from the Simulation:
                        r = Simulation.getRoot()
                        o = BaseMechanicalState(r.createObject("MechanicalObject", name="MyMechanicalObject"))
                        
                Examples of method availables (not showing the in-herited ones):           
                        len(o)
                        o.getPX(int i)
                        o.getPY(int i)
                        o.getPZ(int i)
                        o.applyRotation(rx,ry,rz)
                        o.applyScale(sx, sy, sz)
                        o.applyTranslation(tx,ty,tz)                        
                                        
                To iterate on the DOFs you can do: 
                        for i in range(0, len(o)):
                                o.getPX(i)
        """
        def __init__(self, *args):
                if not args:
                        raise Exception("Cannot create an empty mechanical state, please use the ObjectFactory.createObject('myobject', 'MechanicalObject')) ")
                elif (len(args)==1):
                        if isinstance(args[0], (Base, BaseObject)):
                                self.__init__1(args[0])
                        else:
                                raise Exception("Too much parameters") 
                else:
                        raise Exception("Too much parameters") 
    
        def getPX(self,  i ):
                """Returns the X component of the i'th position DOFS"""
                assert isinstance(i, (int, long)), 'arg i wrong type'
    
                cdef double _r = (self.mechanicalstateptr).getPX((<size_t>i))
                py_result = <double>_r
                return py_result 
                
        def getPY(self,  i ):
                """Returns the Y component of the i'th position DOFS"""
                assert isinstance(i, (int, long)), 'arg i wrong type'
    
                cdef double _r = (self.mechanicalstateptr).getPY((<size_t>i))
                py_result = <double>_r
                return py_result 
                
        def getPZ(self,  i ):
                """Returns the Z component of the i'th position DOFS"""
                assert isinstance(i, (int, long)), 'arg i wrong type'
    
                cdef double _r = (self.mechanicalstateptr).getPZ((<size_t>i))
                py_result = <double>_r
                return py_result 
                
        def applyScale(self, double sx , double sy , double sz ):
                assert isinstance(sx, float), 'arg sx wrong type'
                assert isinstance(sy, float), 'arg sy wrong type'
                assert isinstance(sz, float), 'arg sz wrong type'
    
                self.mechanicalstateptr.applyScale((<double>sx), (<double>sy), (<double>sz)) 
        
        def applyTranslation(self, double sx , double sy , double sz ):
                assert isinstance(sx, float), 'arg sx wrong type'
                assert isinstance(sy, float), 'arg sy wrong type'
                assert isinstance(sz, float), 'arg sz wrong type'
    
                self.mechanicalstateptr.applyTranslation((<double>sx), (<double>sy), (<double>sz)) 
                
        def applyRotation(self, double sx , double sy , double sz ):
                assert isinstance(sx, float), 'arg sx wrong type'
                assert isinstance(sy, float), 'arg sy wrong type'
                assert isinstance(sz, float), 'arg sz wrong type'
    
                self.mechanicalstateptr.applyRotation((<double>sx), (<double>sy), (<double>sz)) 
                
        def resize(BaseMechanicalState self not None, size_t aNewSize):
                self.mechanicalstateptr.resize(aNewSize) 
                return self.mechanicalstateptr.getSize() 
       
        def __len__(self):
                return self.mechanicalstateptr.getSize() 
       
        @staticmethod
        cdef createFrom(_BaseMechanicalState* aptr):
                cdef BaseMechanicalState py_obj = BaseMechanicalState.__new__(BaseMechanicalState)
                py_obj.realptr = py_obj.baseobjectptr = py_obj.mechanicalstateptr = aptr
                return py_obj
      
    
        def __init__1(self, Base aBase):
                cdef _BaseMechanicalState* dc = dynamic_cast_basemechanicalobject(aBase.realptr) ;
                if dc == NULL:
                        raise Exception("Unable to create a BaseMechanicalState from a the provided argument")
                self.realptr = self.baseobjectptr = self.mechanicalstateptr = dc
                                          
       
