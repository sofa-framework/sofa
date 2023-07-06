
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
#include <sofa/simulation/SceneLoaderFactory.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/SetDirectory.h>


namespace sofa
{
namespace simulation
{

SceneLoader::Listeners SceneLoader::s_listeners;

/// load the file
sofa::simulation::NodeSPtr SceneLoader::load(const std::string& filename, bool reload, const std::vector<std::string>& sceneArgs)
{
    if(reload)
        notifyReloadingSceneBefore(this);
    else
        notifyLoadingSceneBefore(this);

    sofa::simulation::NodeSPtr root = doLoad(filename, sceneArgs);

    if(reload)
        notifyReloadingSceneAfter(root, this);
    else
        notifyLoadingSceneAfter(root, this);

    return root;
}

void SceneLoader::notifyLoadingSceneBefore(SceneLoader* sceneLoader)
{
    for (auto* l : s_listeners)
    {
        l->rightBeforeLoadingScene(sceneLoader);
    }
}

void SceneLoader::notifyReloadingSceneBefore(SceneLoader* sceneLoader)
{
    for (auto* l : s_listeners)
    {
        l->rightBeforeReloadingScene(sceneLoader);
    }
}

void SceneLoader::notifyLoadingSceneAfter(sofa::simulation::NodeSPtr node, SceneLoader* sceneLoader)
{
    for (auto* l : s_listeners)
    {
        l->rightAfterLoadingScene(node, sceneLoader);
    }
}

void SceneLoader::notifyReloadingSceneAfter(sofa::simulation::NodeSPtr node,
                                            SceneLoader* sceneLoader)
{
    for (auto* l : s_listeners)
    {
        l->rightAfterReloadingScene(node, sceneLoader);
    }
}

bool SceneLoader::canLoadFileName(const char *filename)
{
    const std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
    return canLoadFileExtension(ext.c_str());
}

/// Pre-saving check
bool SceneLoader::canWriteFileName(const char *filename)
{
    const std::string ext = sofa::helper::system::SetDirectory::GetExtension(filename);
    return canWriteFileExtension(ext.c_str());
}

bool SceneLoader::syntaxForAddingRequiredPlugin(const std::string& pluginName,
                                                const std::vector<std::string>& listComponents,
                                                std::ostream& ss,
                                                sofa::simulation::Node* nodeWhereAdded)
{
    SOFA_UNUSED(pluginName);
    SOFA_UNUSED(listComponents);
    SOFA_UNUSED(ss);
    SOFA_UNUSED(nodeWhereAdded);

    return false;
}

void SceneLoader::Listener::rightBeforeLoadingScene(SceneLoader* sceneLoader)
{
    SOFA_UNUSED(sceneLoader);
}

void SceneLoader::Listener::rightAfterLoadingScene(sofa::simulation::NodeSPtr,
                                                   SceneLoader* sceneLoader)
{
    SOFA_UNUSED(sceneLoader);
}

void SceneLoader::Listener::rightBeforeReloadingScene(SceneLoader* sceneLoader)
{
    this->rightBeforeLoadingScene(sceneLoader);
}

void SceneLoader::Listener::rightAfterReloadingScene(sofa::simulation::NodeSPtr root,
                                                     SceneLoader* sceneLoader)
{
    this->rightAfterLoadingScene(root, sceneLoader);
}

/// adding a listener
void SceneLoader::addListener( Listener* l ) { s_listeners.insert(l); }

/// removing a listener
void SceneLoader::removeListener( Listener* l ) { s_listeners.erase(l); }


SceneLoaderFactory* SceneLoaderFactory::getInstance()
{
    static SceneLoaderFactory instance;
    return &instance;
}

/// This function resturns a real object but it is RVO optimized.
std::vector<std::string> SceneLoaderFactory::extensions()
{
    std::vector<std::string> tmp ;
    SceneLoaderFactory::SceneLoaderList* loaders = getEntries();
    for (SceneLoaderFactory::SceneLoaderList::iterator it=loaders->begin(); it!=loaders->end(); ++it)
    {
        SceneLoader::ExtensionList extensions;
        (*it)->getExtensionList(&extensions);
        for (SceneLoader::ExtensionList::iterator itExt=extensions.begin(); itExt!=extensions.end(); ++itExt)
        {
            tmp.push_back(*itExt) ;
        }
    }
    return tmp ;
}


/// Get an entry given a file extension
SceneLoader* SceneLoaderFactory::getEntryFileExtension(std::string extension)
{
    SceneLoaderList::iterator it = registry.begin();
    while (it!=registry.end())
    {
        if ((*it)->canLoadFileExtension(extension.c_str()))
            return *it;
        ++it;
    }
    // not found, sorry....
    return nullptr;
}

/// Get an entry given a file extension
SceneLoader* SceneLoaderFactory::getEntryFileName(std::string filename)
{
    SceneLoaderList::iterator it = registry.begin();
    while (it!=registry.end())
    {
        if ((*it)->canLoadFileName(filename.c_str()))
            return *it;
        ++it;
    }
    // not found, sorry....
    return nullptr;
}


SceneLoader* SceneLoaderFactory::getExporterEntryFileExtension(std::string extension)
{
    SceneLoaderList::iterator it = registry.begin();
    while (it!=registry.end())
    {
        if ((*it)->canWriteFileExtension(extension.c_str()))
            return *it;
        it++;
    }
    // not found, sorry....
    return nullptr;
}

SceneLoader* SceneLoaderFactory::getExporterEntryFileName(std::string filename)
{
    SceneLoaderList::iterator it = registry.begin();
    while (it!=registry.end())
    {
        if ((*it)->canWriteFileName(filename.c_str()))
            return *it;
        it++;
    }
    // not found, sorry....
    return nullptr;
}

/// Add a scene loader
SceneLoader* SceneLoaderFactory::addEntry(SceneLoader *loader)
{
    registry.push_back(loader);
    return loader;
}

} // namespace simulation

} // namespace sofa


