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
#define SOFA_COMPONENT_MISC_ADDRESOURCEREPOSITORY_CPP

#include <sofa/component/sceneutility/AddResourceRepository.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::sceneutility
{

BaseAddResourceRepository::BaseAddResourceRepository()
    : Inherit1()
    , m_repository(nullptr)
    , d_repositoryPath(initData(&d_repositoryPath, "path", "Path to add to the pool of resources"))
    , m_currentAddedPath("")
{
    d_repositoryPath.setPathType(core::objectmodel::PathType::DIRECTORY);
    addUpdateCallback("path", {&d_repositoryPath}, [this](const core::DataTracker& tracker)
    {
        SOFA_UNUSED(tracker);
        if (this->updateRepositoryPath())
            return sofa::core::objectmodel::ComponentState::Valid;
        return sofa::core::objectmodel::ComponentState::Invalid;
    }, {});
}

BaseAddResourceRepository::~BaseAddResourceRepository()
{

}

bool BaseAddResourceRepository::updateRepositoryPath()
{
    m_repository = getFileRepository();

    std::string tmpAddedPath;
    tmpAddedPath = d_repositoryPath.getValue();

    //first
    //if absolute, add directly in the list of paths
    //else prepend (absolute) current directory to the given path and add it
    if (!sofa::helper::system::FileSystem::isAbsolute(tmpAddedPath))
    {
        tmpAddedPath = FileSystem::append(sofa::helper::system::SetDirectory::GetCurrentDir(), tmpAddedPath);
    }
    //second, check if the path exists
    if (!sofa::helper::system::FileSystem::exists(tmpAddedPath))
    {
        msg_error(this) << tmpAddedPath + " does not exist !";
        return false;
    }
    //third, check if it is really a directory
    if (!sofa::helper::system::FileSystem::isDirectory(tmpAddedPath))
    {
        msg_error(this) << tmpAddedPath + " is not a valid directory !";
        return false;
    }

    if (!m_currentAddedPath.empty())
        m_repository->removePath(m_currentAddedPath);

    m_currentAddedPath = FileSystem::cleanPath(tmpAddedPath);
    if (m_currentAddedPath != d_repositoryPath.getValue())
        d_repositoryPath.setValue(m_currentAddedPath);

    m_repository->addLastPath(m_currentAddedPath);
    msg_info(this) << "Added path: " << m_currentAddedPath;

    if(this->f_printLog.getValue())
        m_repository->print();
    return true;
}


void BaseAddResourceRepository::parse(sofa::core::objectmodel::BaseObjectDescription* arg)
{
    Inherit1::parse(arg);
    updateRepositoryPath();
}

void BaseAddResourceRepository::cleanup()
{
    Inherit1::cleanup();
    m_repository->removePath(m_currentAddedPath);
}


static int AddDataRepositoryClass = core::RegisterObject("Add a path to DataRepository")
    .add< AddDataRepository >()
    .addAlias("AddResourceRepository") // Backward compatibility
    ;

static int AddPluginRepositoryClass = core::RegisterObject("Add a path to PluginRepository")
    .add< AddPluginRepository >();


} // namespace sofa::component::sceneutility
