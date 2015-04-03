#include "python.h"

python::map_type python::_argtypes;
python::map_type python::_restype;

extern "C" {

    const char* argtypes(python::func_ptr_type func) {
        return python::argtypes( func );
    }

    const char* restype(python::func_ptr_type func) {
        return python::restype( func );
    }

}




