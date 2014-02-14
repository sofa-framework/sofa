#include "PythonVisitor.h"
#include "PythonMacros.h"


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
    PyObject *res=PyObject_CallMethod(m_PyVisitor,const_cast<char*>("processNodeTopDown"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node));
    if (!res)
    {
        printf("<SofaPython> exception\n");
        PyErr_Print();
        return Visitor::RESULT_PRUNE;
    }

    if PyBool_Check(res)
        return Visitor::RESULT_CONTINUE;

    return Visitor::RESULT_PRUNE;
}

void PythonVisitor::processNodeBottomUp(simulation::Node* node)
{
    // SP_OBJ_CALL(m_PyVisitor, processNodeBottomUp, "(O)", SP_BUILD_PYSPTR(node))
    PyObject *res=PyObject_CallMethod(m_PyVisitor,const_cast<char*>("processNodeBottomUp"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node));
    if (!res)
    {
        printf("<SofaPython> exception\n");
        PyErr_Print();
    }
}


} // namespace simulation

} // namespace sofa
