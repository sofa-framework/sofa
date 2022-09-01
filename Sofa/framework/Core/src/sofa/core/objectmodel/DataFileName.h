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
#include <sofa/core/objectmodel/Data.h>
#include <sofa/type/SVector.h>

namespace sofa::core::objectmodel
{

enum class PathType {
    FILE,
    DIRECTORY,
    BOTH
};

/**
 *  \brief Data specialized to store filenames, potentially relative to the current directory at the time it was specified.
 *
 */
class SOFA_CORE_API DataFileName : public sofa::core::objectmodel::Data<std::string>
{
public:


    typedef sofa::core::objectmodel::Data<std::string> Inherit;

    DataFileName( const std::string& helpMsg="", bool isDisplayed=true, bool isReadOnly=false );

    DataFileName( const std::string& value, const std::string& helpMsg="", bool isDisplayed=true, bool isReadOnly=false );

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileName(const BaseData::BaseInitData& init);

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileName(const Inherit::InitData& init);

    ~DataFileName() override = default;


    void setPathType(PathType pathType);

    PathType getPathType() const;

    bool read(const std::string& s ) override;

    void endEdit() override;


    virtual const std::string& getRelativePath() const;

    virtual const std::string& getFullPath() const;

    virtual const std::string& getAbsolutePath() const;

    virtual const std::string& getExtension() const;

    void doOnUpdate() override;

protected:
    void updatePath();

    std::string m_fullpath;
    std::string m_relativepath;
    std::string m_extension;
    PathType    m_pathType; ///< used to determine how file dialogs should be opened

public:
    DataFileName(const Inherit& d) = delete;
    DataFileName& operator=(const DataFileName&) = delete;
};

} // namespace sofa::core::objectmodel
