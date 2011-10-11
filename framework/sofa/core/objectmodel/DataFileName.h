/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_DATAFILENAME_H
#define SOFA_CORE_OBJECTMODEL_DATAFILENAME_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/vector.h>

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
    }

    virtual ~DataFileName()
    {
    }

    void endEdit()
    {
        updatePath();
        Inherit::endEdit();
    }

    void setValue(const std::string& v)
    {
        *beginEdit()=v;
        endEdit();
    }
    virtual void virtualSetValue(const std::string& v) { setValue(v); }
    virtual bool read( std::string& s )
    {
        bool ret = Inherit::read(s);
        if (ret) updatePath();
        return ret;
    }

    virtual const std::string& getRelativePath() const { return getValue(); }
    virtual const std::string& getFullPath() const
    {
        this->updateIfDirty();
        return fullpath;
    }
    virtual const std::string& getAbsolutePath() const
    {
        this->updateIfDirty();
        return fullpath;
    }

    virtual void update()
    {
        this->Inherit::update();
        this->updatePath();
    }

protected:
    void updatePath();

    std::string fullpath;

private:
    DataFileName(const Inherit& d);
    DataFileName& operator=(const DataFileName&);
};



class DataFileNameVector : public sofa::core::objectmodel::Data< sofa::helper::vector<std::string> >
{
public:
    typedef sofa::core::objectmodel::Data<sofa::helper::vector<std::string> > Inherit;

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
    }

    virtual ~DataFileNameVector()
    {
    }

    void endEdit()
    {
        updatePath();
        Inherit::endEdit();
    }

    void setValue(const std::string& v)
    {
        beginEdit()->push_back(v);
        endEdit();
    }
    virtual void virtualSetValue(const std::string& v) { setValue(v); }
    virtual bool read( std::string& s )
    {
        bool ret = Inherit::read(s);
        if (ret) updatePath();
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
