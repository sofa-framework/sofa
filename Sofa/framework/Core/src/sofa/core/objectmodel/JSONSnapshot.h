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
#include <sofa/core/objectmodel/Base.h>
#include <sofa/core/objectmodel/BaseSnapshot.h>
#include <nlohmann/json.hpp>



namespace sofa::core::objectmodel
{

class SOFA_CORE_API JSONSnapshot : public BaseSnapshot 
{

public:
    void printSnapshot() override;
    //void exportSnapshot(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& linkfield) override;
    void importSnapshot() override;

    void fillDataSnapshot(BaseData* dat) override;
    void fillSnapshot(DataSnapshot datasnap) override;
    void fillLinkSnapshot(BaseLink* link) override;
    void collectData(const std::vector<BaseData*>& datafield, const std::vector<BaseLink*>& linkfield) override;
    void putData(std::vector<BaseData*>& datafield, std::vector<BaseLink*>& linkfield) override;
        // std::vector<BaseLink*> getLinkField() const;
        // std::vector<BaseLink*> getLinkField() const;
    void exportToJSON(const std::string filename) override;
    void importFromJSON(const std::string filename,nlohmann::json& j);
    JSONSnapshot();
    ~JSONSnapshot();




};
} // namespace sofa::core::objectmodel