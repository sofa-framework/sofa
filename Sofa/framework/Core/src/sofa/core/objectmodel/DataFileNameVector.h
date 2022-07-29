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
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/type/SVector.h>

namespace sofa::core::objectmodel
{

class SOFA_CORE_API DataFileNameVector : public sofa::core::objectmodel::Data< sofa::type::SVector<std::string> >
{
public:
    typedef sofa::core::objectmodel::Data<sofa::type::SVector<std::string> > Inherit;

    DataFileNameVector( const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false);

    DataFileNameVector( const sofa::type::SVector<std::string>& value, const char* helpMsg=nullptr, bool isDisplayed=true, bool isReadOnly=false );

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileNameVector(const BaseData::BaseInitData& init);

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileNameVector(const Inherit::InitData& init);

    ~DataFileNameVector() override = default;

    void endEdit() override;

    void addPath(const std::string& v, bool clear = false);

    void setValueAsString(const std::string& v);

    bool read(const std::string& s ) override;

    virtual const std::string& getRelativePath(unsigned int i);
    virtual const std::string& getFullPath(unsigned int i) const;

    virtual const std::string& getAbsolutePath(unsigned int i) const;

    void doOnUpdate() override;

    void setPathType(PathType pathType);

    PathType getPathType() const;

protected:
    void updatePath();

    sofa::type::vector<std::string> m_fullpath;
    PathType m_pathType; ///< used to determine how file dialogs should be opened

public:
    DataFileNameVector(const Inherit& d) = delete;
    DataFileNameVector& operator=(const DataFileNameVector&) = delete;
};

}
