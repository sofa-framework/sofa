/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/common/SceneLoaderXML.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/cast.h>

#include <sofa/simulation/common/xml/XML.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>

namespace sofa::simulation
{

// register the loader in the factory
const SceneLoader* loaderXML = SceneLoaderFactory::getInstance()->addEntry(new SceneLoaderXML());
bool SceneLoaderXML::loadSucceed = true;


bool SceneLoaderXML::canLoadFileExtension(const char *extension)
{
    std::string ext = extension;
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext=="xml" || ext=="scn");
}


bool SceneLoaderXML::canWriteFileExtension(const char *extension)
{
    return canLoadFileExtension(extension);
}

/// get the file type description
std::string SceneLoaderXML::getFileTypeDesc()
{
    return "Scenes";
}

/// get the list of file extensions
void SceneLoaderXML::getExtensionList(ExtensionList* list)
{
    list->clear();
    list->push_back("xml");
    list->push_back("scn");
}

bool SceneLoaderXML::syntaxForAddingRequiredPlugin(const std::string& pluginName,
                                                   const std::vector<std::string>& listComponents, std::ostream& ss, sofa::simulation::Node* nodeWhereAdded)
{
    SOFA_UNUSED(nodeWhereAdded);

    ss << "<RequiredPlugin name=\"" << pluginName << "\"/> <!-- Needed to use components [";
    if (!listComponents.empty())
    {
        ss << sofa::helper::join(listComponents, ',');
    }
    ss << "] -->" << msgendl;
    return true;
}

sofa::simulation::Node::SPtr SceneLoaderXML::doLoad(const std::string& filename, const std::vector<std::string>& sceneArgs)
{
    SOFA_UNUSED(sceneArgs);
    sofa::simulation::Node::SPtr root;

    if (!canLoadFileName(filename.c_str()))
        return 0;

    xml::BaseElement* xml = xml::loadFromFile ( filename.c_str() );
    root = processXML(xml, filename.c_str());

    delete xml;

    return root;
}

void SceneLoaderXML::write(Node *node, const char *filename)
{
    sofa::simulation::node::exportInXML(node, filename);
}

/// Load a scene from a file
Node::SPtr SceneLoaderXML::processXML(xml::BaseElement* xml, const char *filename)
{
    loadSucceed = true;

    if ( xml==nullptr )
    {
        return nullptr;
    }

    // We go the current file's directory so that all relative path are correct
    helper::system::SetDirectory chdir ( filename );

    // Temporarily set the numeric formatting locale to ensure that
    // floating-point values are interpreted correctly by tinyXML. (I.e. the
    // decimal separator is a dot '.').
    helper::system::TemporaryLocale locale(LC_NUMERIC, "C");

    sofa::simulation::xml::NodeElement* nodeElt = dynamic_cast<sofa::simulation::xml::NodeElement *>(xml);
    if( nodeElt==nullptr )
    {
        msg_fatal_withfile("SceneLoaderXML", xml->getSrcFile(), xml->getSrcLine()) << "XML Root Node is not an Element. \n" ;
        loadSucceed = false;
        std::exit(EXIT_FAILURE);
    }
    else if( !(nodeElt->init()) )
    {
        msg_error_withfile("SceneLoaderXML", xml->getSrcFile(), xml->getSrcLine()) << "Node initialization failed. \n" ;
        loadSucceed = false;
    }

    core::objectmodel::BaseNode* baseroot = xml->getObject()->toBaseNode();
    if ( baseroot == nullptr )
    {
        msg_error_withfile("SceneLoaderXML", xml->getSrcFile(), xml->getSrcLine()) << "Objects initialization failed." ;
        loadSucceed = false;
        return nullptr;
    }

    Node::SPtr root = down_cast<Node> ( baseroot );

    return root;
}

NodeSPtr SceneLoaderXML::doLoadFromMemory(const char* filename, const char* data)
{
    notifyLoadingSceneBefore(this);

    xml::BaseElement* xml = xml::loadFromMemory(filename, data);

    Node::SPtr root = processXML(xml, filename);

    delete xml;
    notifyLoadingSceneAfter(root, this);
    return root;
}

/// Load from a string in memory
Node::SPtr SceneLoaderXML::loadFromMemory(const char* filename, const char* data)
{
    SceneLoaderXML sceneLoader;
    return sceneLoader.doLoadFromMemory(filename, data);
}


} // namespace sofa::simulation
