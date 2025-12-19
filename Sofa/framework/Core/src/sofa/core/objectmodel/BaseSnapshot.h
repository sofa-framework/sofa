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
// #include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseData.h>
#include <sofa/core/objectmodel/BaseLink.h>

namespace sofa::core::objectmodel
{
class Base;
class SOFA_CORE_API BaseSnapshot 
{
    
protected:

    struct DataSnapshot
    {
        std::vector<BaseData*> dataContainer;
        std::vector<BaseLink*> linkContainer;
    };

    DataSnapshot dataSnapshot_;

    std::vector<DataSnapshot> snapshot;

public:
    virtual void printSnapshot() = 0;
    virtual void exportSnapshot() = 0;
    virtual void importSnapshot() = 0;
    virtual void fillDataSnapshot(BaseData* dat) = 0 ;
    virtual void fillSnapshot(DataSnapshot datasnap) = 0;
    virtual void fillLinkSnapshot(BaseLink* link) = 0;
    virtual void collectData(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& componentlinks) = 0;

    BaseSnapshot();
    virtual ~BaseSnapshot() = 0;


    
};
} // namespace sofa::core::objectmodel