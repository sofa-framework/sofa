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
    struct DataInfo
    {
        std::string name;
        std::string type;
        std::string value;
        std::string ownername;
    };

    struct LinkInfo
    {
        std::string name;
        std::string linkedpath;
        std::string path;
    };

    struct SparseDataSnapshot
    {
        std::vector<DataInfo> dataContainer;
        std::vector<LinkInfo> linkContainer;
    };

    SparseDataSnapshot SparseDataSnapshot_;

    std::vector<SparseDataSnapshot> SparseSnapshot;

    std::vector<std::string> ComponentSnapshot;

    std::vector<std::vector<SparseDataSnapshot>> NodeSnapshot;

public:
    virtual void printSnapshot() = 0;
    virtual void importSnapshot() = 0;
    virtual void fillDataSnapshot(BaseData* dat) = 0 ;
    virtual void fillSnapshot(DataSnapshot datasnap) = 0;
    virtual void fillLinkSnapshot(BaseLink* link) = 0;
    virtual void collectData(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& linkfield) = 0;
    virtual void groupComponent() = 0;
    virtual void putData(std::vector<BaseData*>& datafield, std::vector<BaseLink*>& linkfield,BaseSnapshot::DataInfo& di) = 0;
    virtual void exportTo(const std::string filename) = 0;

    BaseSnapshot();
    virtual ~BaseSnapshot() = 0;


    
};
} // namespace sofa::core::objectmodel