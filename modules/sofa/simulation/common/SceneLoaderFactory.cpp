
/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SceneLoaderFactory.h"




namespace sofa
{
namespace simulation
{

SceneLoaderFactory* SceneLoaderFactory::getInstance()
{
    static SceneLoaderFactory instance;
    return &instance;
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


