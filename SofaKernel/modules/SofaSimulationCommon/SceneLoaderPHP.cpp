/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "SceneLoaderPHP.h"

#include <SofaSimulationCommon/SceneLoaderXML.h>
#include <sofa/helper/system/PipeProcess.h>
#include <SofaSimulationCommon/xml/NodeElement.h>
#include <sofa/helper/system/FileRepository.h>

namespace sofa
{

namespace simulation
{

// register the loader in the factory
const SceneLoader* loaderPHP = SceneLoaderFactory::getInstance()->addEntry(new SceneLoaderPHP());




bool SceneLoaderPHP::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="php" || ext=="pscn");
}

/// get the file type description
std::string SceneLoaderPHP::getFileTypeDesc()
{
    return "Php Scenes";
}

/// get the list of file extensions
void SceneLoaderPHP::getExtensionList(ExtensionList* list)
{
    list->clear();
    list->push_back("pscn");
//    list->push_back("php");
}


sofa::simulation::Node::SPtr SceneLoaderPHP::load(const char *filename)
{
    sofa::simulation::Node::SPtr root;

    if (!canLoadFileName(filename))
        return 0;

    std::string out="",error="";
    std::vector<std::string> args;


    //TODO : replace when PipeProcess will get file as stdin
    //at the moment, the filename is given as an argument
    args.push_back(std::string("-f" + std::string(filename)));
    //args.push_back("-w");
    std::string newFilename="";
    //std::string newFilename=filename;

    helper::system::FileRepository fp("PATH", ".");
#ifdef WIN32
    std::string command = "php.exe";
#else
    std::string command = "php";
#endif
    if (!fp.findFile(command,""))
    {
        msg_error("SceneLoaderPHP") << "Php not found in your PATH environment." ;
        return NULL;
    }

    sofa::helper::system::PipeProcess::executeProcess(command.c_str(), args,  newFilename, out, error);

    if(error != "")
    {
        msg_error("SceneLoaderPHP") << error ;
        if (out == "")
            return NULL;
    }
    root = SceneLoaderXML::loadFromMemory(filename, out.c_str(), (unsigned int)out.size());

    return root;
}


} // namespace simulation

} // namespace sofa

