#include "Binding_TriangleSetTopologyModifier.h"
#include "Binding_PointSetTopologyModifier.h"
#include <sofa/core/topology/Topology.h>
#include <SofaBaseTopology/TriangleSetTopologyModifier.h>
#include <sofa/helper/vector.h>

using namespace sofa;
using namespace sofa::core::topology;
using namespace sofa::component::topology;


Topology::Triangle parseTriangleTuple( PyObject* tuple )
{
    Topology::Triangle T;

    if (!PyArg_ParseTuple(tuple, "III",&T[0],&T[1],&T[2]))
    {
        PyErr_BadArgument();
    }
    return T;
}

sofa::helper::vector< Topology::Triangle  > parseTriangleList( PyObject* args )
{
    sofa::helper::vector< Topology::Triangle > triangles;

    bool isList = PyList_Check(args);
    if( !isList ) return triangles;

    bool isEmptyList = (PyList_Size(args)==0);
    if( isEmptyList ) return triangles;

    bool isTwoDimensionsList = PyList_Check(PyList_GetItem(args,0));

    if( isTwoDimensionsList )
    {
         PyErr_BadArgument();
    }
    else
    {
        int nbRows = PyList_Size(args);
        for (int i=0; i<nbRows; ++i)
        {
            PyObject *tuple = PyList_GetItem(args,i);
            triangles.push_back( parseTriangleTuple(tuple) );
        }
    }

    return triangles;
}

template < class T >
T pyConvert( PyObject* /*obj */ )
{
    return T();
}

template < >
double pyConvert<double>( PyObject* obj )
{
    return PyFloat_AsDouble(obj);
}

template < >
float pyConvert<float>( PyObject* obj )
{
    return static_cast<float>( PyFloat_AsDouble(obj) );
}

template < >
unsigned int pyConvert<unsigned int>( PyObject* obj )
{
    return static_cast<unsigned int>(PyInt_AsLong(obj) );
}

template < class T >
sofa::helper::vector< T > parseVector( PyObject* args )
{
    std::size_t nbRows = PyList_Size(args);
    sofa::helper::vector<T> values;
    values.reserve(nbRows);

    for(std::size_t i=0;i<nbRows;++i)
    {
        PyObject * item = PyList_GetItem(args,i);
        values.push_back( pyConvert<T>( item ) );
    }

    return values;
}

template < class T >
sofa::helper::vector< sofa::helper::vector< T > > parseVectorOfVector( PyObject* args )
{
    sofa::helper::vector< sofa::helper::vector< T > > vectorOfvector;

    bool isList = PyList_Check(args);
    bool isTwoDimensionsList = PyList_Check(PyList_GetItem(args,0));

    std::size_t nbRows = PyList_Size(args);
    for (std::size_t i=0; i<nbRows; ++i)
    {
        PyObject *row = PyList_GetItem(args,i);

        sofa::helper::vector<T> values = parseVector<T>( row );

        vectorOfvector.push_back(values);
    }

    return vectorOfvector;
}


extern "C" PyObject * TriangleSetTopologyModifier_addTriangles(PyObject *self, PyObject * args)
{

    TriangleSetTopologyModifier* obj=dynamic_cast<TriangleSetTopologyModifier*>(((PySPtr<Base>*)self)->object.get());
    
    PyObject* triangleArgs  = NULL;
    PyObject* ancestorsArgs = NULL;
    PyObject* coefsArgs     = NULL;

    if (PyArg_UnpackTuple(args, "addTriangles", 1, 3, &triangleArgs, &ancestorsArgs, &coefsArgs)) 
    {
        sofa::helper::vector< Topology::Triangle > triangles = parseTriangleList( triangleArgs );

        if( !triangles.empty() )
        {
            if(ancestorsArgs && coefsArgs )
            {
                sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors = parseVectorOfVector<unsigned int>( ancestorsArgs );
                sofa::helper::vector< sofa::helper::vector< double       > > coefs     = parseVectorOfVector<double>(coefsArgs);
                obj->addTriangles(triangles, ancestors, coefs );
            }
            else
            {
                obj->addTriangles( triangles );
            }  
        }
    }
    Py_RETURN_NONE;
}


extern "C" PyObject * TriangleSetTopologyModifier_removeTriangles(PyObject *self, PyObject * args)
{
    TriangleSetTopologyModifier* obj=dynamic_cast<TriangleSetTopologyModifier*>(((PySPtr<Base>*)self)->object.get());
    
    PyObject* triangleIndicesArg      = NULL;
    PyObject* removeIsolatedEdgesArg  = NULL;
    PyObject* removeIsolatedPointsArg = NULL;

    if (PyArg_UnpackTuple(args, "removeTriangles", 1, 3, &triangleIndicesArg, &removeIsolatedEdgesArg, &removeIsolatedPointsArg)) 
    {
        sofa::helper::vector< unsigned int > triangleIndices;
        bool removeIsolatedEdges=true;
        bool removeIsolatedPoints=true;
        
        if( ! PyList_Check(triangleIndicesArg) )
        {
            PyErr_BadArgument();
            Py_RETURN_NONE;
        }

        std::size_t nbTriangles = PyList_Size(triangleIndicesArg);
        for(std::size_t i=0;i<nbTriangles;++i)
        {
            triangleIndices.push_back(  PyLong_AsUnsignedLong( PyList_GetItem(triangleIndicesArg,i) ) );
        }

        if( removeIsolatedEdgesArg && (removeIsolatedEdgesArg == Py_False) )
        {
            removeIsolatedEdges=false;
        }

        if( removeIsolatedPointsArg && ( removeIsolatedPointsArg == Py_False) )
        {
            removeIsolatedPoints=false;
        }
        
        obj->removeTriangles(triangleIndices,removeIsolatedEdges,removeIsolatedPoints);
    }

    Py_RETURN_NONE;
}


extern "C" PyObject * TriangleSetTopologyModifier_addRemoveTriangles(PyObject *self, PyObject * args)
{

    TriangleSetTopologyModifier* obj=dynamic_cast<TriangleSetTopologyModifier*>(((PySPtr<Base>*)self)->object.get());

    PyObject* trianglesArg            = NULL;
    PyObject* triangleIndicesArg      = NULL;
    PyObject* ancestorsArg            = NULL;
    PyObject* coefsArg                = NULL;
    PyObject* triangles2RemoveArg      = NULL;

    if (PyArg_UnpackTuple(args, "removeTriangles", 5, 5, &trianglesArg, 
                                                         &triangleIndicesArg, 
                                                         &ancestorsArg,
                                                         &coefsArg,
                                                         &triangles2RemoveArg) )
    {
        sofa::helper::vector< Topology::Triangle > triangles = parseTriangleList( trianglesArg );
        sofa::helper::vector< unsigned int       > triangleIndices    = parseVector<unsigned int>( triangleIndicesArg );
        sofa::helper::vector< sofa::helper::vector< unsigned int > > ancestors = parseVectorOfVector<unsigned int>( ancestorsArg );
        sofa::helper::vector< sofa::helper::vector< double       > > coefs     = parseVectorOfVector<double>(coefsArg);
        sofa::helper::vector< unsigned int > triangles2remove = parseVector<unsigned int>(triangles2RemoveArg);

        obj->addRemoveTriangles(triangles.size(),triangles,triangleIndices,ancestors,coefs, triangles2remove );

    }

    Py_RETURN_NONE;
}

SP_CLASS_METHODS_BEGIN(TriangleSetTopologyModifier)
SP_CLASS_METHOD(TriangleSetTopologyModifier,addTriangles)
SP_CLASS_METHOD(TriangleSetTopologyModifier,removeTriangles)
SP_CLASS_METHOD(TriangleSetTopologyModifier,addRemoveTriangles)
SP_CLASS_METHODS_END

SP_CLASS_TYPE_SPTR(TriangleSetTopologyModifier,TriangleSetTopologyModifier,PointSetTopologyModifier)
