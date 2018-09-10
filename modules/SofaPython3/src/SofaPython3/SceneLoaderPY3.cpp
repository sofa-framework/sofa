/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sstream>
#include <fstream>

#include <sofa/simulation/Simulation.h>
#include <sofa/helper/ArgumentParser.h>
//#include <SofaSimulationCommon/xml/NodeElement.h>
//#include <SofaSimulationCommon/FindByTypeVisitor.h>

#include <SofaPython3/PythonEnvironment.h>
#include <SofaPython3/SceneLoaderPY3.h>

using namespace sofa::core::objectmodel;
using sofa::helper::system::SetDirectory;

MSG_REGISTER_CLASS(sofapython3::SceneLoaderPY3, "SofaPython3::SceneLoader")

namespace sofapython3
{

bool SceneLoaderPY3::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="py" || ext=="py3" || ext=="py3scn" || ext=="pyscn");
}

bool SceneLoaderPY3::canWriteFileExtension(const char *extension)
{
    return canLoadFileExtension(extension);
}

/// get the file type description
std::string SceneLoaderPY3::getFileTypeDesc()
{
    return "Python3 Scenes";
}

/// get the list of file extensions
void SceneLoaderPY3::getExtensionList(ExtensionList* list)
{
    list->clear();
    list->push_back("py3scn");
    list->push_back("py3");
    list->push_back("pyscn");
    list->push_back("py");
}

sofa::simulation::Node::SPtr SceneLoaderPY3::load(const char *filename)
{
    sofa::simulation::Node::SPtr root;
    loadSceneWithArguments(filename, sofa::helper::ArgumentParser::extra_args(), &root);
    return root;
}

void SceneLoaderPY3::loadSceneWithArguments(const char *filename,
                                           const std::vector<std::string>& arguments,
                                           Node::SPtr* root_out)
{
    PythonEnvironment::gil lock(__func__);

    std::cout << "COUCOUC loading..: "<< filename << std::endl ;

    std::stringstream tmp;
    tmp << "with open('"<<filename<<"') as f:                             " << std::endl
        << "    global_vars={}                                            " << std::endl
        << "    local_vars={}                                             " << std::endl
        << "    code = compile(f.read(), '" << filename    << "', 'exec') " << std::endl
        << "    exec(code, global_vars, local_vars)                       " << std::endl
        << "    print(str(global_vars))" ;

    PythonEnvironment::runString(tmp.str());


//    if(!OurHeader.empty() && 0 != PyRun_SimpleString(OurHeader.c_str()))
//    {
//        msg_error("SofaPython3::SceneLoader") << "header script run error." ;
//        if( root_out ) *root_out = 0;
//        return;
//    }

//    PythonEnvironment::runString("createScene=None");
//    PythonEnvironment::runString("createSceneAndController=None");

//    PythonEnvironment::runString(std::string("__file__=\"") + filename + "\"");

//    // We go to the current file's directory so that all relative path are correct
//    SetDirectory chdir ( filename );

//    notifyLoadingScene();
//    PythonEnvironment::setArguments(SetDirectory::GetFileName(filename), arguments);
//    if(!PythonEnvironment::runFile(SetDirectory::GetFileName(filename)))
//    {
//        msg_error() << "scene script load error." ;
//        if( root_out ) *root_out = 0;
//        return;
//    }

//    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

//    // pFunc is also a borrowed reference
//    PyObject *pFunc = PyDict_GetItemString(pDict, "createScene");
//    if (PyCallable_Check(pFunc))
//    {
//        Node::SPtr rootNode = Node::create("root");
//        if(root_out) *root_out = rootNode;

//        SP_CALL_MODULEFUNC(pFunc, "(O)", sofa::PythonFactory::toPython(rootNode.get()));
//        return;
//    }
//    else
//    {
//        PyObject *pFunc = PyDict_GetItemString(pDict, "createSceneAndController");
//        if (PyCallable_Check(pFunc))
//        {
//            Node::SPtr rootNode = Node::create("root");
//            if(root_out) *root_out = rootNode;

//            SP_CALL_MODULEFUNC(pFunc, "(O)", sofa::PythonFactory::toPython(rootNode.get()));
//            rootNode->addObject( core::objectmodel::New<component::controller::PythonMainScriptController>( filename ) );

//            return;
//        }
//    }

//    SP_MESSAGE_ERROR( "cannot create Scene, no \"createScene(rootNode)\" nor \"createSceneAndController(rootNode)\" module method found." );
//    if( root_out ) *root_out = 0;
//    return;
}


bool SceneLoaderPY3::loadTestWithArguments(const char *filename, const std::vector<std::string>& arguments)
{
    PythonEnvironment::gil lock(__func__);

    std::stringstream tmp;

    std::cout << "COUCOUCOCU " << std::endl ;
    tmp << "with open(" << filename << ") as f:                   " << std::endl
        << "     code = compile(f.read(), "<<filename<<", 'exec') " << std::endl
        << "     exec(code, global_vars, local_vars)              " << std::endl
        << "print(dir())                                          " << std::endl;

    PythonEnvironment::runString(tmp.str());

//    PythonEnvironment::gil lock(__func__);
//    if(!OurHeader.empty() && 0 != PyRun_SimpleString(OurHeader.c_str()))
//    {
//        SP_MESSAGE_ERROR( "header script run error." )
//        return false;
//    }

//    PythonEnvironment::runString("createScene=None");
//    PythonEnvironment::runString("createSceneAndController=None");

//    PythonEnvironment::runString(std::string("__file__=\"") + filename + "\"");

//    // it runs the unecessary SofaPython script but it is not a big deal
//    if(!PythonEnvironment::runFile(filename,arguments))
//    {
//        // LOAD ERROR
//        SP_MESSAGE_ERROR( "script load error." )
//        return false;
//    }

//    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));

//    // pFunc is also a borrowed reference
//    PyObject *pFunc = PyDict_GetItemString(pDict, "run");
//    if (PyCallable_Check(pFunc))
//    {
//        PyObject *res = PyObject_CallObject(pFunc,0);
//        printPythonExceptions();

//        if( !res )
//        {
//            SP_MESSAGE_ERROR( "Python test 'run' function does not return any value" )
//            return false;
//        }
//        else if( !PyBool_Check(res) )
//        {
//            SP_MESSAGE_ERROR( "Python test 'run' function does not return a boolean" )
//            Py_DECREF(res);
//            return false;
//        }

//        bool result = ( res == Py_True );
//        Py_DECREF(res);

//        return result;
//    }
//    else
//    {
//        SP_MESSAGE_ERROR( "Python test has no 'run'' function" )
//        return false;
//    }
}


} // namespace sofapython3

