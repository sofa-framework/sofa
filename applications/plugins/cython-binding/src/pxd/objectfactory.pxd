# -*- coding: ASCII -*-
from boost cimport intrusive_ptr
from baseobject cimport BaseObject, BaseObjectSPtr
from basecontext cimport BaseContext
from baseobjectdescription cimport BaseObjectDescription 

cdef extern from "../../../../../framework/sofa/core/ObjectFactory.h" namespace "sofa::core": 
        cdef cppclass ObjectFactory:
                ObjectFactory() except +         
              
cdef extern from "../../../../../framework/sofa/core/ObjectFactory.h" namespace "sofa::core::ObjectFactory":       
        BaseObjectSPtr CreateObject(BaseContext* context, BaseObjectDescription* arg)
        
