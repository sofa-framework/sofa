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

#include <sofa/core/config.h>
#include <string>

namespace sofa::core::objectmodel
{

class BaseData;

/// @brief AbstractDataLink is the base class for every link between two data fields
/// The targetted BaseData is called the "target",
/// The base object owning the current "child" object is the "owner"
/// it is possible to store a path in a DataLink, in that case, at each DataLink access
/// the path is resolved to search for a corresponding Data until one is found.
/// Once a Data is set, the path is discarded.
class AbstractDataLink
{
public:
    /// Returns the BaseData object that this DataLink belong to.
    /// There is a one to one owner relationship.
    /// A BaseData have one and only one DataLink object.
    /// A DataLink object has one and only one BaseData as owner.
    const BaseData& getOwner();

    /// Change the targetted DataField and set the path to the empty string
    void setTarget(BaseData* target);

    /// Get the targetted DataField
    BaseData* getTarget();

    /// Get the path (is any)
    const std::string getPath() const;

    /// Set the path, try to resolve it, on success set the DataField
    void setPath(const std::string& path);

    /// Returns true if the path is set (and thus getTarget() == nullptr)
    bool hasPath() const;

    /// Search for the Data pointed by the path (if any).
    /// If there is no path set, returns false
    /// If there is no owner set, returns false
    /// If there is no compatible Data at pointed path, returns false
    /// Otherwise, returns true.
    /// After a successfull call, the path is set to empty string.
    bool resolvePathAndSetData();

protected:
    ///////////////////////////////////////////////////////////////////////////
    /// The three folowing methods must be implemented by any child class.
    /// This design delegates to child class the work of actually storing
    /// the real Data.
    /// Real implementation for the setTarget() method
    virtual void __doSetTarget__(BaseData* target) = 0;

    /// Real implementation for the getTarget() method
    virtual BaseData* __doGetTarget__() = 0;

    /// Real implementation for the GetOwner() method
    virtual const BaseData& __doGetOwner__() = 0;
    ///////////////////////////////////////////////////////////////////////////

    std::string m_path {""};
};

} /// namespace sofa::core::objectmodel

