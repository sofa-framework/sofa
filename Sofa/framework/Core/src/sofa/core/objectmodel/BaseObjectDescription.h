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

#include <sofa/type/vector.h>
#include <map>

#include <sofa/core/config.h>

namespace sofa::core::objectmodel
{

class Base;

/**
 *  \brief Base Interface for classes containing the description of an object, used to construct it.
 *
 *  This class defines what informations are used as input (read from a file for instance) to create an object.
 *  This default implementation simply stores an attributes map and does not support any hierarchy.
 *
 */
class SOFA_CORE_API BaseObjectDescription
{
public:
    class Attribute
    {
    protected:
        std::string value;
        mutable bool accessed;
    public:
        Attribute() : accessed(false) {}
        Attribute(const std::string& v) : value(v), accessed(false) {}
        void operator=(const std::string& v) { value = v; }
        void operator=(const char* v) { value = v; }
        operator std::string() const { accessed = true; return value; }
        const char* c_str() const { accessed = true; return value.c_str(); }
        bool isAccessed() const { return accessed; }
        void setAccessed(bool v) { accessed = v; }
    };

    typedef std::map<std::string,Attribute> AttributeMap;

    BaseObjectDescription(const char* name=nullptr, const char* type=nullptr);

    virtual ~BaseObjectDescription();

    /// Get the associated object (or nullptr if it is not created yet)
    virtual Base* getObject();

    /// Get the object instance name
    virtual std::string getName();

    /// Set the object instance name
    virtual void setName(const std::string& name);

    /// Get the parent node
    virtual BaseObjectDescription* getParent() const;

    /// Get the file where this description was read from. Useful to resolve relative file paths.
    virtual std::string getBaseFile();

    ///// Get all attribute data, read-only
    virtual const AttributeMap& getAttributeMap() const;

    ///// Get list of all attributes
    template<class T> void getAttributeList(T& container) const
    {
        for (AttributeMap::const_iterator it = attributes.begin();
                it != attributes.end(); ++it)
            container.push_back(it->first);
    }

    /// Find an object description given its name (relative to this object)
    virtual BaseObjectDescription* find(const char* nodeName, bool absolute=false);

    /// Find an object given its name (relative to this object)
    virtual Base* findObject(const char* nodeName);

    /// Get an attribute given its name (return defaultVal if not present)
    virtual const char* getAttribute(const std::string& attr, const char* defaultVal=nullptr);

    /// Get an attribute converted to a float given its name.
    /// returns defaultVal if not present or in case the attribute cannot be parsed totally
    /// adds a message in the logError if the attribute cannot be totally parsed.
    virtual float getAttributeAsFloat(const std::string& attr, const float defaultVal=0.0);

    /// Get an attribute converted to a int given its name.
    /// returns defaultVal if not present or in case the attribute cannot be parsed totally
    /// adds a message in the logError if the attribute cannot be totally parsed.
    virtual int getAttributeAsInt(const std::string& attr, const int defaultVal=0.0) ;

    /// Set an attribute. Override any existing value
    virtual void setAttribute(const std::string& attr, const std::string& val);

    /// Remove an attribute given its name
    virtual bool removeAttribute(const std::string& attr);

    /// Get the full name of this object (i.e. concatenation if all the names of its ancestors and itself)
    virtual std::string getFullName();

    virtual void logError(const std::string & s) {errors.push_back(s);}
    virtual void logErrors(const std::vector<std::string> & e) {errors.insert(errors.end(), e.begin(), e.end());}

    std::vector< std::string > const& getErrors() const {return errors;}
    virtual void clearErrors() {errors.clear();}

protected:
    AttributeMap attributes;
    std::vector< std::string > errors;
};
} // namespace sofa
