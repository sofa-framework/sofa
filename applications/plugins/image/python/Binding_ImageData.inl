// NOTE: this is a kind of old-school way to implement templates
// with preprocessor macros
// no to have to duplicate code

// preprocessor functions to concatenate 2 variables
#define PASTER(x,y) x ## y
#define EVALUATOR(x,y)  PASTER(x,y)

// the bound class name
#define BOUNDNAME EVALUATOR(IMAGETYPE,Data)


SP_DECLARE_CLASS_TYPE(BOUNDNAME)

using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;





static PyObject * EVALUATOR(BOUNDNAME,getPtrs)(PyObject * self, PyObject * /*args*/)
{
    Data<IMAGETYPE>* data = sofa::py::unwrap< Data<IMAGETYPE> >( self );
    IMAGETYPE& image = *data->beginEdit();  // where should be the endedit?

    IMAGETYPE::imCoord dim = image.getDimensions();


    PyObject* imglist = PyList_New(dim[4]); // t
    for( size_t i=0;i<dim[4];++i)
        PyList_SetItem( imglist, i, PyLong_FromVoidPtr( image.getCImg(i)._data ) ); // s


    PyObject* shape = PyTuple_New(4);
    PyTuple_SetItem( shape, 0, PyLong_FromSsize_t( dim[3] ) ); // s
    PyTuple_SetItem( shape, 1, PyLong_FromSsize_t( dim[2] ) ); // z
    PyTuple_SetItem( shape, 2, PyLong_FromSsize_t( dim[1] ) ); // y
    PyTuple_SetItem( shape, 3, PyLong_FromSsize_t( dim[0] ) ); // x


    // output = tuple( list(pointers), shape tuple, type name)
    PyObject* res = PyTuple_New(3);

    // the data pointer
    PyTuple_SetItem( res, 0, imglist );

    // the shape
    PyTuple_SetItem( res, 1, shape );

    // the type name
    PyTuple_SetItem( res, 2, PyString_FromString( DataTypeName<IMAGETYPE::T>::name() ) );

    return res;
}




static PyObject * EVALUATOR(BOUNDNAME,getDimensions)(PyObject * self, PyObject * /*args*/)
{
    Data<IMAGETYPE>* data = sofa::py::unwrap< Data<IMAGETYPE> >( self );
    const IMAGETYPE& image = data->getValue();

    IMAGETYPE::imCoord dim = image.getDimensions();

    PyObject* res = PyList_New(5);
    PyList_SetItem( res, 0, PyFloat_FromDouble( dim[0] ) );
    PyList_SetItem( res, 1, PyFloat_FromDouble( dim[1] ) );
    PyList_SetItem( res, 2, PyFloat_FromDouble( dim[2] ) );
    PyList_SetItem( res, 3, PyFloat_FromDouble( dim[3] ) );
    PyList_SetItem( res, 4, PyFloat_FromDouble( dim[4] ) );

    return res;
}

#define SP_CLASS_SIMPLE_METHOD(M) {#M, EVALUATOR(BOUNDNAME,M), METH_VARARGS, ""},
#define SP_CLASS_SIMPLE__METHOD_KW(M) {#M, (PyCFunction)EVALUATOR(BOUNDNAME,M), METH_KEYWORDS|METH_VARARGS, ""},

SP_CLASS_METHODS_BEGIN(BOUNDNAME)
SP_CLASS_SIMPLE_METHOD(getPtrs)
SP_CLASS_SIMPLE_METHOD(getDimensions)
SP_CLASS_METHODS_END

#undef SP_CLASS_SIMPLE__METHOD_KW
#undef SP_CLASS_SIMPLE_METHOD


#define SP_CLASS_SIMPLE_ATTR(A) {(char*)#A, EVALUATOR(BOUNDNAME,getAttr_##A), EVALUATOR(BOUNDNAME,setAttr_##A), NULL, 0},

// eventual attributes
SP_CLASS_ATTRS_BEGIN(BOUNDNAME)
//SP_CLASS_SIMPLE_ATTR(ImageType,name)
SP_CLASS_ATTRS_END

#undef SP_CLASS_SIMPLE_ATTR






SP_CLASS_TYPE_PTR_ATTR(BOUNDNAME,sofa::core::objectmodel::Data<sofa::defaulttype::IMAGETYPE>,Data)

#undef BOUNDNAME
