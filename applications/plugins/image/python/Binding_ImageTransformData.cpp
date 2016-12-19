#include <SofaPython/PythonMacros.h>
#include <SofaPython/Binding_SofaModule.h>
#include <SofaPython/PythonFactory.h>

#include "Binding_ImageTransformData.h"
#include <SofaPython/Binding_Data.h>



SP_DECLARE_CLASS_TYPE(ImageTransformData)

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

typedef SReal Real;
typedef ImageLPTransform<Real> MyImageTransform;
typedef typename MyImageTransform::Coord Coord;


extern "C" PyObject * ImageTransformData_fromImage(PyObject * self, PyObject * args)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        Py_RETURN_NONE;
    }

//    msg_info("ImageTransformData_fromImage")<<x<<" "<<y<<" "<<z;

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();
    Coord p = transform.fromImage( Coord(x,y,z) );


//    msg_info("ImageTransformData_fromImage")<<p;

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









extern "C" PyObject * ImageTransformData_getAttr_params(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    const MyImageTransform::Params& p = transform.getParams();

    PyObject* res = PyList_New(12);
    PyList_SetItem( res, 0, PyFloat_FromDouble( p[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( p[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( p[2] ) );
    PyList_SetItem( res, 3, PyFloat_FromDouble( p[3] ) );
    PyList_SetItem( res, 4, PyFloat_FromDouble( p[4] ) );
    PyList_SetItem( res, 5, PyFloat_FromDouble( p[5] ) );
    PyList_SetItem( res, 6, PyFloat_FromDouble( p[6] ) );
    PyList_SetItem( res, 7, PyFloat_FromDouble( p[7] ) );
    PyList_SetItem( res, 8, PyFloat_FromDouble( p[8] ) );
    PyList_SetItem( res, 9, PyFloat_FromDouble( p[9] ) );
    PyList_SetItem( res, 10, PyFloat_FromDouble( p[10] ) );
    PyList_SetItem( res, 11, PyFloat_FromDouble( p[11] ) );
    return res;
}


extern "C" int ImageTransformData_setAttr_params(PyObject *self, PyObject * args, void*)
{
    double p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11;
    if (!PyArg_ParseTuple(args, "dddddddddddd",&p0,&p1,&p2,&p3,&p4,&p5,&p6,&p7,&p8,&p9,&p10,&p11))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getParams().set(p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11);

    data->endEdit();
    return 0;
}


extern "C" PyObject * ImageTransformData_getAttr_translation(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    const Coord& t = transform.getTranslation();

    PyObject* res = PyList_New(3);
    PyList_SetItem( res, 0, PyFloat_FromDouble( t[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( t[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( t[2] ) );
    return res;
}


extern "C" int ImageTransformData_setAttr_translation(PyObject *self, PyObject * args, void*)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getTranslation().set(x,y,z);

    data->endEdit();
    return 0;
}

extern "C" PyObject * ImageTransformData_getAttr_rotation(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    const Coord& t = transform.getRotation();

    PyObject* res = PyList_New(3);
    PyList_SetItem( res, 0, PyFloat_FromDouble( t[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( t[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( t[2] ) );
    return res;
}

extern "C" int ImageTransformData_setAttr_rotation(PyObject *self, PyObject * args, void*)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getRotation().set(x,y,z);
    transform.update();

    data->endEdit();
    return 0;
}

extern "C" PyObject * ImageTransformData_getAttr_scale(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    const Coord& t = transform.getScale();

    PyObject* res = PyList_New(3);
    PyList_SetItem( res, 0, PyFloat_FromDouble( t[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( t[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( t[2] ) );
    return res;
}

extern "C" int ImageTransformData_setAttr_scale(PyObject *self, PyObject * args, void*)
{
    double x,y,z;
    if (!PyArg_ParseTuple(args, "ddd",&x,&y,&z))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getScale().set(x,y,z);

    data->endEdit();
    return 0;
}


extern "C" PyObject * ImageTransformData_getAttr_offsetT(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    return PyFloat_FromDouble( transform.getOffsetT() );
}

extern "C" int ImageTransformData_setAttr_offsetT(PyObject *self, PyObject * args, void*)
{
    double a;
    if (!PyArg_ParseTuple(args, "d",&a))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getOffsetT() = a;

    data->endEdit();
    return 0;
}


extern "C" PyObject * ImageTransformData_getAttr_scaleT(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    return PyFloat_FromDouble( transform.getScaleT() );
}

extern "C" int ImageTransformData_setAttr_scaleT(PyObject *self, PyObject * args, void*)
{
    double a;
    if (!PyArg_ParseTuple(args, "d",&a))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getScaleT() = a;

    data->endEdit();
    return 0;
}


extern "C" PyObject * ImageTransformData_getAttr_perspective(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    return PyFloat_FromDouble( transform.isPerspective() );
}

extern "C" int ImageTransformData_setAttr_perspective(PyObject *self, PyObject * args, void*)
{
    double a;
    if (!PyArg_ParseTuple(args, "d",&a))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.isPerspective() = a;

    data->endEdit();
    return 0;
}

extern "C" PyObject * ImageTransformData_getAttr_camPos(PyObject *self, void*)
{
    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    const MyImageTransform& transform = data->getValue();

    const Vec<2,Real>& t = transform.getCamPos();

    PyObject* res = PyList_New(2);
    PyList_SetItem( res, 0, PyFloat_FromDouble( t[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( t[1] ) );
    return res;
}

extern "C" int ImageTransformData_setAttr_camPos(PyObject *self, PyObject * args, void*)
{
    double x,y;
    if (!PyArg_ParseTuple(args, "dd",&x,&y))
    {
        PyErr_BadArgument();
        return 0;
    }

    Data<MyImageTransform>* data=((PyPtr< Data<MyImageTransform> >*)self)->object;
    MyImageTransform& transform = *data->beginEdit();

    transform.getCamPos().set(x,y);

    data->endEdit();
    return 0;
}



SP_CLASS_METHODS_BEGIN(ImageTransformData)
SP_CLASS_METHOD(ImageTransformData,fromImage)
SP_CLASS_METHOD(ImageTransformData,toImage)
SP_CLASS_METHODS_END


SP_CLASS_ATTRS_BEGIN(ImageTransformData)
SP_CLASS_ATTR(ImageTransformData,params)
SP_CLASS_ATTR(ImageTransformData,translation)
SP_CLASS_ATTR(ImageTransformData,rotation)
SP_CLASS_ATTR(ImageTransformData,scale)
SP_CLASS_ATTR(ImageTransformData,offsetT)
SP_CLASS_ATTR(ImageTransformData,scaleT)
SP_CLASS_ATTR(ImageTransformData,perspective)
SP_CLASS_ATTR(ImageTransformData,camPos)
SP_CLASS_ATTRS_END







SP_CLASS_TYPE_PTR_ATTR(ImageTransformData,Data<MyImageTransform>,Data)
