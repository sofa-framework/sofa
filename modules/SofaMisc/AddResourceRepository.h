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
#ifndef SOFA_COMPONENT_MISC_ADDRESOURCEREPOSITORY_H
#define SOFA_COMPONENT_MISC_ADDRESOURCEREPOSITORY_H

#include <SofaMisc/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/helper/system/FileRepository.h>


namespace sofa
{

namespace component
{

namespace misc
{

/** Add a new repository path at startup
*/
class AddResourceRepository: public sofa::core::objectmodel::BaseObject
{

public:
    SOFA_CLASS(AddResourceRepository, sofa::core::objectmodel::BaseObject);

    typedef sofa::core::objectmodel::BaseObject Inherit;

protected:
    /** Default constructor
    */
    AddResourceRepository();
    virtual ~AddResourceRepository();
public:
    //cannot be a DataFilename
    Data<std::string> d_repositoryPath;

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override;
    void cleanup() override;
private:
    std::string m_currentAddedPath;


};

} // namespace misc

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MISC_ADDRESOURCEREPOSITORY_H
