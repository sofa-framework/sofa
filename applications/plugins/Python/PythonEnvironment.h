#ifndef PYTHONENVIRONMENT_H
#define PYTHONENVIRONMENT_H


#include <sofa/simulation/tree/GNode.h>

#include <Python.h>

namespace sofa
{

namespace simulation
{


class PythonEnvironment
{
public:
    static void     Init();
    static void     Release();

    // helper functions
    static sofa::simulation::tree::GNode::SPtr  initGraphFromScript( const char *filename );        // returns root node

    // basic script functions
    static PyObject*    importScript( const char *filename );
    static bool         initGraph(PyObject *script, sofa::simulation::tree::GNode::SPtr graphRoot);  // calls the method "initGraph(root)" of the script
};


} // namespace core

} // namespace sofa


#endif // PYTHONENVIRONMENT_H
