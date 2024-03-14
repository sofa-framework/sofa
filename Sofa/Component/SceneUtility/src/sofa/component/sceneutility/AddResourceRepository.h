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
#pragma once
#include <sofa/component/sceneutility/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/system/FileRepository.h>
using sofa::helper::system::FileRepository;
using sofa::core::objectmodel::DataFileName;

namespace sofa::component::sceneutility
{

class SOFA_COMPONENT_SCENEUTILITY_API BaseAddResourceRepository: public sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(BaseAddResourceRepository, sofa::core::objectmodel::BaseObject);

protected:
    BaseAddResourceRepository();
    ~BaseAddResourceRepository() override;

    FileRepository* m_repository;

public:
    DataFileName d_repositoryPath; ///< Path to add to the pool of resources

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override;
    bool updateRepositoryPath();
    void cleanup() override;

private:
    std::string m_currentAddedPath;

    virtual FileRepository* getFileRepository() = 0;
};


/// Add a new path to DataRepository
class SOFA_COMPONENT_SCENEUTILITY_API AddDataRepository: public BaseAddResourceRepository
{
public:
    SOFA_CLASS(AddDataRepository, BaseAddResourceRepository);

protected:
    FileRepository* getFileRepository() override { return &sofa::helper::system::DataRepository; }
};


/// Add a new path to PluginRepository
class SOFA_COMPONENT_SCENEUTILITY_API AddPluginRepository: public BaseAddResourceRepository
{
public:
    SOFA_CLASS(AddPluginRepository, BaseAddResourceRepository);

protected:
    FileRepository* getFileRepository() override { return &sofa::helper::system::PluginRepository; }
};


} // namespace sofa::component::sceneutility
