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
#define SOFA_COMPONENT_MISC_ADDRESOURCEREPOSITORY_CPP

#include <SofaMisc/AddResourceRepository.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace misc
{

SOFA_DECL_CLASS(AddResourceRepository)


int AddResourceRepositoryClass = core::RegisterObject("Add a repository to the pool of resources")
.add< AddResourceRepository >();

AddResourceRepository::AddResourceRepository()
    : Inherit()
    , d_repositoryPath(initData(&d_repositoryPath, "path", "Path to add to the pool of resources"))
    , m_currentAddedPath()
{

}

AddResourceRepository::~AddResourceRepository()
{

}

void AddResourceRepository::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit::parse(arg);
    std::string tmpAddedPath;
    tmpAddedPath = d_repositoryPath.getValue();

    //first
    //if absolute, add directly in the list of paths
    //else prepend (absolute) current directory to the given path and add it
    if (!sofa::helper::system::FileSystem::isAbsolute(tmpAddedPath))
    {
        tmpAddedPath = sofa::helper::system::SetDirectory::GetCurrentDir() + "/" + tmpAddedPath;
    }
    //second, check if the path exists
    if (!sofa::helper::system::FileSystem::exists(tmpAddedPath))
    {
        msg_error(this) << tmpAddedPath + " does not exist !";
        return;
    }
    //third, check if it is really a directory
    if (!sofa::helper::system::FileSystem::isDirectory(tmpAddedPath))
    {
        msg_error(this) << tmpAddedPath + " is not a valid directory !";
        return;
    }

    m_currentAddedPath = tmpAddedPath;
    sofa::helper::system::DataRepository.addLastPath(m_currentAddedPath);

    if(this->f_printLog.getValue())
        sofa::helper::system::DataRepository.print();
}

void AddResourceRepository::cleanup()
{
    Inherit::cleanup();
    sofa::helper::system::DataRepository.removePath(m_currentAddedPath);

}

} // namespace misc

} // namespace component

} // namespace sofa
