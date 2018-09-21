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

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;

using namespace sofa::core::objectmodel;
using sofa::helper::system::SetDirectory;

MSG_REGISTER_CLASS(sofapython3::SceneLoaderPY3, "SofaPython3::SceneLoader")

PYBIND11_DECLARE_HOLDER_TYPE(Base, sofa::core::sptr<Base>, true)
template class py::class_<sofa::core::objectmodel::Base,
sofa::core::sptr<sofa::core::objectmodel::Base>>;

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

py::module import(const std::string& module, const std::string& path, py::object& globals)
{
    py::dict locals;
    locals["module_name"] = py::cast(module); // have to cast the std::string first
    locals["path"]        = py::cast(path);

    py::eval<py::eval_statements>(            // tell eval we're passing multiple statements
                                              "import imp\n"
                                              "new_module = imp.load_module(module_name, open(path), path, ('py', 'U', imp.PY_SOURCE))\n",
                                              globals,
                                              locals);
    return py::cast<py::module>(locals["new_module"]);
}

void SceneLoaderPY3::loadSceneWithArguments(const char *filename,
                                            const std::vector<std::string>& arguments,
                                            Node::SPtr* root_out)
{
    notifyLoadingScene();
    PythonEnvironment::gil lock(__func__);

    try{
        py::module::import("Sofa");
        py::object globals = py::module::import("__main__").attr("__dict__");
        py::module module;

        SetDirectory localDir(filename);
        std::string basename = SetDirectory::GetFileNameWithoutExtension(SetDirectory::GetFileName(filename).c_str());
        module = import(basename, SetDirectory::GetFileName(filename), globals);
        if(!module.attr("createScene"))
        {
            msg_error() << "Missing createScene function";
            return ;
        }

        *root_out = Node::create("root");
        py::object createScene = module.attr("createScene");
        createScene(*root_out);
    }catch(std::exception& e)
    {
        msg_error() << e.what();
    }
}

} // namespace sofapython3

