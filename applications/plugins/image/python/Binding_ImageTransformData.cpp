#include <SofaPython/PythonMacros.h>
#include <SofaPython/Binding_SofaModule.h>
#include <SofaPython/PythonFactory.h>

#include "Binding_ImageTransformData.h"
#include <SofaPython/Binding_Data.h>



SP_DECLARE_CLASS_TYPE(ImageTransformData)

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

typedef ImageLPTransform<SReal> MyImageTransform;
typedef typename MyImageTransform::Coord Coord;


extern "C" PyObject * ImageTransformData_fromImage(PyObject * self, PyObject * args)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();
    Coord p = transform.fromImage( Coord(x,y,z) );

    PyObject* res = PyList_New(3);
    PyList_SetItem( res, 0, PyFloat_FromDouble( p[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( p[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( p[2] ) );

    return res;
}


extern "C" PyObject * ImageTransformData_toImage(PyObject * self, PyObject * args)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();
    Coord p = transform.toImage( Coord(x,y,z) );

    PyObject* res = PyList_New(3);
    PyList_SetItem( res, 0, PyFloat_FromDouble( p[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( p[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( p[2] ) );

    return res;
}

SP_CLASS_METHODS_BEGIN(ImageTransformData)
SP_CLASS_METHOD(ImageTransformData,fromImage)
SP_CLASS_METHOD(ImageTransformData,toImage)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(ImageTransformData)
// TODO add every attributes
SP_CLASS_ATTRS_END







SP_CLASS_TYPE_PTR_ATTR(ImageTransformData,Data<MyImageTransform>,Data)
