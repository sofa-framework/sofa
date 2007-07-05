/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_CORE_OBJECTMODEL_BASEOBJECTDESCRIPTION_H
#define SOFA_CORE_OBJECTMODEL_BASEOBJECTDESCRIPTION_H

#include <string>
#include <list>
#include <map>

namespace sofa
{

namespace core
{

namespace objectmodel
{

class Base;

/**
 *  \brief Base Interface for classes containing the description of an object, used to construct it.
 *
 *  This class defines what informations are used as input (read from a file for instance) to create an object.
 *  This default implementation simply stores an attributes map and does not support any hierarchy.
 *
 */
class BaseObjectDescription
{
public:
    class Attribute
    {
    protected:
        std::string value;
        bool accessed;
    public:
        Attribute() : accessed(false) {}
        Attribute(const std::string& v) : value(v), accessed(false) {}
        void operator=(const std::string& v) { value = v; }
        void operator=(const char* v) { value = v; }
        operator std::string() { accessed = true; return value; }
        const char* c_str() { accessed = true; return value.c_str(); }
        bool isAccessed() { return accessed; }
        void setAccessed(bool v) { accessed = v; }
    };

    typedef std::map<std::string,Attribute> AttributeMap;

    BaseObjectDescription(const char* name=NULL, const char* type=NULL);

    virtual ~BaseObjectDescription();

    /// Get the associated object (or NULL if it is not created yet)
    virtual Base* getObject();

    /// Get the object instance name
    virtual std::string getName();

    /// Get the parent node
    virtual BaseObjectDescription* getParent() const;

    /// Get the file where this description was read from. Useful to resolve relative file paths.
    virtual std::string getBaseFile();

    ///// Get all attribute data, read-only
    //virtual const AttributeMap& getAttributeMap() const;

    ///// Get list of all attributes
    template<class T> void getAttributeList(T& container)
    {
        for (AttributeMap::iterator it = attributes.begin();
                it != attributes.end(); ++it)
            container.push_back(it->first);
    }

    /// Find an object description given its name (relative to this object)
    virtual BaseObjectDescription* find(const char* nodeName, bool absolute=false);

    /// Find an object given its name (relative to this object)
    virtual Base* findObject(const char* nodeName);

    /// Get an attribute given its name (return defaultVal if not present)
    virtual const char* getAttribute(const std::string& attr, const char* defaultVal=NULL);

    /// Remove an attribute given its name
    virtual bool removeAttribute(const std::string& attr);

    /// Get the full name of this object (i.e. concatenation if all the names of its ancestors and itself)
    virtual std::string getFullName();

protected:
    AttributeMap attributes;
};

} // namespace objectmodel

} // namespace core

} // namespace sofa

#endif
