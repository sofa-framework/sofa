#include "PythonMacros.h"

#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/loader/BaseLoader.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/core/topology/Topology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/GridTopology.h>
#include <SofaBaseTopology/RegularGridTopology.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <sofa/core/BaseMapping.h>
#include "PythonScriptController.h"
#include "PythonEnvironment.h"
#include "PythonMacros.h"
using sofa::simulation::PythonEnvironment ;

typedef sofa::component::container::MechanicalObject< sofa::defaulttype::Vec3Types > MechanicalObject3;
typedef sofa::component::mapping::SubsetMultiMapping< sofa::defaulttype::Vec3Types, sofa::defaulttype::Vec3Types > SubsetMultiMapping3_to_3;

/// This function converts an PyObject into a sofa string.
/// string that can be safely parsed in helper::vector<int> or helper::vector<double>
std::ostream& pythonToSofaDataString(PyObject* value, std::ostream& out)
{
    /// String are just returned as string.
    if (PyString_Check(value))
    {
        return out << PyString_AsString(value) ;
    }

    /// Unicode are converted to string.
    if(PyUnicode_Check(value))
    {
        PyObject* tmpstr = PyUnicode_AsUTF8String(value);
        out << PyString_AsString(tmpstr) ;
        Py_DECREF(tmpstr);

        return out;
    }

    if( PySequence_Check(value) )
    {
        if(!PyList_Check(value))
        {
            msg_warning("SofaPython") << "A sequence which is not a list will be convert to a sofa string.";
        }
        /// It is a sequence...so we can iterate over it.
        PyObject *iterator = PyObject_GetIter(value);
        if(iterator)
        {
            bool first = true;
            while(PyObject* next = PyIter_Next(iterator))
            {
                if(first) first = false;
                else out << ' ';

                pythonToSofaDataString(next, out);
                Py_DECREF(next);
            }
            Py_DECREF(iterator);

            if (PyErr_Occurred())
            {
                msg_error("SofaPython") << "error while iterating." << msgendl
                                        << PythonEnvironment::getStackAsString() ;
            }
            return out;
        }
    }


    /// Check if the object has an explicit conversion to a Sofa path. If this is the case
    /// we use it.
    if( PyObject_HasAttrString(value, "getAsACreateObjectParameter") ){
        PyObject* retvalue = PyObject_CallMethod(value, (char*)"getAsACreateObjectParameter", nullptr) ;
        return pythonToSofaDataString(retvalue, out);
    }

    /// Default conversion for standard type:
    if( !(PyInt_Check(value) || PyLong_Check(value) || PyFloat_Check(value) || PyBool_Check(value) ))
    {
        msg_warning("SofaPython") << "You are trying to convert a non primitive type to Sofa using the 'str' operator." << msgendl
                                  << "Automatic conversion is provided for: String, Integer, Long, Float and Bool and Sequences." << msgendl
                                  << "Other objects should implement the method getAsACreateObjectParameter(). " << msgendl
                                  << "This function should return a string usable as a parameter in createObject()." << msgendl
                                  << "So to remove this message you must add a method getAsCreateObjectParameter(self) "
                                     "to the object you are passing the createObject function." << msgendl
                                  << PythonEnvironment::getStackAsString() ;
    }


    PyObject* tmpstr=PyObject_Repr(value);
    out << PyString_AsString(tmpstr) ;
    Py_DECREF(tmpstr) ;
    return out ;
}


void printPythonExceptions()
{
    PyObject *ptype, *pvalue /* error msg */, *ptraceback /*stack snapshot and many other informations (see python traceback structure)*/;
    PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    if( pvalue ) SP_MESSAGE_EXCEPTION( PyString_AsString(pvalue) )

    // TODO improve the error message by using ptraceback
}


void handle_python_error(const char* message) {
    SOFA_UNUSED(message) ;
    if(PyErr_ExceptionMatches(PyExc_SystemExit))  {
        PyErr_Clear();
        throw sofa::simulation::PythonEnvironment::system_exit();
    }
    PyErr_Print();
}
