
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
#include "SceneLoaderFactory.h"




namespace sofa
{
namespace simulation
{

SceneLoader::Listeners SceneLoader::s_listerners;

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
    return 0;
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
    return 0;
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
    return 0;
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
    return 0;
}

/// Add a scene loader
SceneLoader* SceneLoaderFactory::addEntry(SceneLoader *loader)
{
    registry.push_back(loader);
    return loader;
}



} // namespace simulation

} // namespace sofa


