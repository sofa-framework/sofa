#include "PythonMacros.h"
#include "PythonVisitor.h"
#include "PythonFactory.h"



namespace sofa
{

namespace simulation
{


PythonVisitor::PythonVisitor(const core::ExecParams* params, PyObject *pyVisitor)
    : Visitor(params)
    , m_PyVisitor(pyVisitor)
{
}

Visitor::Result PythonVisitor::processNodeTopDown(simulation::Node* node)
{
    PythonEnvironment::gil lock(__func__);
    PyObject *res = PyObject_CallMethod(m_PyVisitor,const_cast<char*>("processNodeTopDown"),const_cast<char*>("(O)"),sofa::PythonFactory::toPython(node));
    if( !res || !PyBool_Check(res) )
    {
        Py_XDECREF(res);
        return Visitor::RESULT_CONTINUE;
    }

    bool r = ( res == Py_True );
    Py_DECREF(res);

    return r?Visitor::RESULT_CONTINUE:Visitor::RESULT_PRUNE;
}

void PythonVisitor::processNodeBottomUp(simulation::Node* node)
{
    PythonEnvironment::gil lock(__func__);    
    PyObject *res = PyObject_CallMethod(m_PyVisitor,const_cast<char*>("processNodeBottomUp"),const_cast<char*>("(O)"),sofa::PythonFactory::toPython(node));
    Py_XDECREF(res);
}

bool PythonVisitor::treeTraversal(TreeTraversalRepetition& repeat)
{
    PythonEnvironment::gil lock(__func__);        
    PyObject *res = PyObject_CallMethod(m_PyVisitor,const_cast<char*>("treeTraversal"),const_cast<char*>("()"));


    if( !res || !PyInt_Check(res) )
    {
        Py_XINCREF(res);
        repeat = NO_REPETITION;
        return false;
    }

    int r = PyInt_AsLong(res);
    Py_DECREF(res);

    switch( r )
    {
    case 0: /// tree no repeat
        repeat = NO_REPETITION;
        return true;
        break;
    case 1: /// tree repeat once
        repeat = REPEAT_ONCE;
        return true;
        break;
    case 2: /// tree repeat all
        repeat = REPEAT_ALL;
        return true;
        break;
    default:
    case -1: /// dag
        repeat = NO_REPETITION;
        return false;
    }
}

} // namespace simulation
} // namespace sofa
