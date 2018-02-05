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
#ifndef SOFA_CORE_OBJECTMODEL_DATAFILENAME_H
#define SOFA_CORE_OBJECTMODEL_DATAFILENAME_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/SVector.h>

namespace sofa
{

namespace core
{

namespace objectmodel
{

/**
 *  \brief Data specialized to store filenames, potentially relative to the current directory at the time it was specified.
 *
 */
class SOFA_CORE_API DataFileName : public sofa::core::objectmodel::Data<std::string>
{
public:
    typedef sofa::core::objectmodel::Data<std::string> Inherit;

    DataFileName( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : Inherit(helpMsg, isDisplayed, isReadOnly)
    {
    }

    DataFileName( const std::string& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : Inherit(value, helpMsg, isDisplayed, isReadOnly)
    {
        updatePath();
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileName(const BaseData::BaseInitData& init)
        : Inherit(init)
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileName(const Inherit::InitData& init)
        : Inherit(init)
    {
        updatePath();
    }

    virtual ~DataFileName()
    {
    }

    void endEdit(const core::ExecParams* params = 0)
    {
        updatePath();
        Inherit::endEdit(params);
    }

    void setValue(const std::string& v)
    {
        *beginEdit()=v;
        endEdit();
    }
    virtual void virtualEndEdit() { endEdit(); }
    virtual void virtualSetValue(const std::string& v) { setValue(v); }
    virtual bool read(const std::string& s );

    virtual const std::string& getRelativePath() const
    {
        this->updateIfDirty();
        return m_relativepath ;
    }

    virtual const std::string& getFullPath() const
    {
        this->updateIfDirty();
        return m_fullpath;
    }
    virtual const std::string& getAbsolutePath() const
    {
        this->updateIfDirty();
        return m_fullpath;
    }

    virtual void update()
    {
        this->Inherit::update();
        this->updatePath();
    }

protected:
    void updatePath();

    std::string m_fullpath;
    std::string m_relativepath;

private:
    DataFileName(const Inherit& d);
    DataFileName& operator=(const DataFileName&);
};



class SOFA_CORE_API DataFileNameVector : public sofa::core::objectmodel::Data< sofa::helper::SVector<std::string> >
{
public:
    typedef sofa::core::objectmodel::Data<sofa::helper::SVector<std::string> > Inherit;

    DataFileNameVector( const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : Inherit(helpMsg, isDisplayed, isReadOnly)
    {
    }

    DataFileNameVector( const sofa::helper::vector<std::string>& value, const char* helpMsg=0, bool isDisplayed=true, bool isReadOnly=false )
        : Inherit(value, helpMsg, isDisplayed, isReadOnly)
    {
        updatePath();
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileNameVector(const BaseData::BaseInitData& init)
        : Inherit(init)
    {
    }

    /** Constructor
        this constructor should be used through the initData() methods
     */
    explicit DataFileNameVector(const Inherit::InitData& init)
        : Inherit(init)
    {
        updatePath();
    }

    virtual ~DataFileNameVector()
    {
    }

    void endEdit(const core::ExecParams* params = 0)
    {
        updatePath();
        Inherit::endEdit(params);
    }

    void setValue(const sofa::helper::vector<std::string>& v)
    {
        *beginEdit() = v;
        endEdit();
    }
    virtual void virtualEndEdit() { endEdit(); }

    void addPath(const std::string& v, bool clear = false)
    {
        sofa::helper::vector<std::string>& val = *beginEdit();
        if(clear) val.clear();
        val.push_back(v);
        endEdit();
    }
    void setValueAsString(const std::string& v)
    {
        sofa::helper::SVector<std::string>& val = *beginEdit();
        val.clear();
        std::istringstream ss( v );
        ss >> val;
        endEdit();
    }
    virtual void virtualSetValueAsString(const std::string& v) { setValueAsString(v); }

    virtual bool read(const std::string& s )
    {
        bool ret = Inherit::read(s);
        if (ret || fullpath.empty()) updatePath();
        return ret;
    }

    virtual const std::string& getRelativePath(unsigned int i) { return getValue()[i]; }
    virtual const std::string& getFullPath(unsigned int i) const
    {
        this->updateIfDirty();
        return fullpath[i];
    }
    virtual const std::string& getAbsolutePath(unsigned int i) const
    {
        this->updateIfDirty();
        return fullpath[i];
    }

    virtual void update()
    {
        this->Inherit::update();
        this->updatePath();
    }

protected:
    void updatePath();

    sofa::helper::vector<std::string> fullpath;

private:
    DataFileNameVector(const Inherit& d);
    DataFileNameVector& operator=(const DataFileNameVector&);
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
