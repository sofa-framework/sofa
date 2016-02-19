# -*- coding: ASCII -*-
from boost cimport intrusive_ptr
from baseobject cimport _BaseObject, _SPtr as _BaseObjectSPtr
from basecontext cimport _BaseContext
from baseobjectdescription cimport _BaseObjectDescription 

cdef extern from "../../../../../framework/sofa/core/ObjectFactory.h" namespace "sofa::core": 
        cdef cppclass _ObjectFactory "sofa::core::ObjectFactory":
                _ObjectFactory() except +         
              
cdef extern from "../../../../../framework/sofa/core/ObjectFactory.h" namespace "sofa::core::ObjectFactory":       
        _BaseObjectSPtr CreateObject(_BaseContext* context, _BaseObjectDescription* arg)
        
        
cdef class ObjectFactory:
        pass
