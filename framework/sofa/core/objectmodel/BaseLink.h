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
#ifndef SOFA_CORE_OBJECTMODEL_BASELINK_H
#define SOFA_CORE_OBJECTMODEL_BASELINK_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/core/core.h>
#include <sofa/core/ExecParams.h>
#include <string>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;
class BaseData;

/**
 *  \brief Abstract base class for all links in the scene grapn, independently of their type.
 *
 */
class SOFA_CORE_API BaseLink
{
public:
    enum LinkFlagsEnum
    {
        FLAG_NONE       = 0,
        FLAG_MULTILINK  = 1 << 0, ///< True if link is an array
        FLAG_STRONGLINK = 1 << 1, ///< True if link has ownership of linked object(s)
        FLAG_DOUBLELINK = 1 << 2, ///< True if link has a reciprocal link in linked object(s)
        FLAG_DATALINK   = 1 << 3, ///< True if link points to a Data
        FLAG_DUPLICATE  = 1 << 4, ///< True if link duplicates another one (possibly with a different/specialized DestType)
        FLAG_STOREPATH  = 1 << 5, ///< True if link requires a path string in order to be created
    };
    typedef unsigned LinkFlags;

    /// This internal class is used by the initLink() methods to store initialization parameters of a Data
    class BaseInitLink
    {
    public:
        BaseInitLink(const char* name, const char* help) : /*link(NULL), linkFlags(FLAG_NONE), owner(NULL), ownerData(NULL), */ name(name), help(help) {}
        //BaseLink* link;
        //LinkFlags linkFlags;
        //Base* owner;
        //BaseData* ownerData;
        const char* name;
        const char* help;
    };

    /// This internal class is used by the initLink() methods to store initialization parameters of a Data
    template<class Owner>
    class InitLink : public BaseInitLink
    {
    public:
        InitLink(Owner* owner, const char* name, const char* help) : BaseInitLink(name, help), owner(owner) {}
        Owner* owner;
    };

    BaseLink(const BaseInitLink& init, LinkFlags flags);
    virtual ~BaseLink();

    const std::string& getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    /// Get help message
    const char* getHelp() const { return m_help; }

    /// Set help message
    void setHelp(const char* val) { m_help = val; }

    virtual Base* getOwnerBase() const = 0;
    virtual BaseData* getOwnerData() const = 0;

    /// Get one flag
    bool getFlag(LinkFlagsEnum flag) const { return (m_flags&(LinkFlags)flag)!=0; }

    bool isMultiLink() const { return getFlag(FLAG_MULTILINK); }
    bool isDataLink() const { return getFlag(FLAG_DATALINK); }
    bool isStrongLink() const { return getFlag(FLAG_STRONGLINK); }
    bool isDoubleLink() const { return getFlag(FLAG_DOUBLELINK); }
    bool isDuplicate() const { return getFlag(FLAG_DUPLICATE); }
    bool storePath() const { return getFlag(FLAG_STOREPATH); }

    virtual unsigned int getSize() const = 0;
    virtual Base* getLinkedBase(unsigned int index=0) const = 0;
    virtual BaseData* getLinkedData(unsigned int index=0) const = 0;

    /// Copy the value of an aspect into another one.
    virtual void copyAspect(int destAspect, int srcAspect) = 0;

    /// Release memory allocated for the specified aspect.
    virtual void releaseAspect(int aspect);


protected:
    unsigned int m_flags;
    std::string m_name;
    const char* m_help;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
