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

#include <map>
#include <sofa/core/fwd.h>

namespace sofa::core
{

/// Tracking Data mechanism
/// to be able to check when selected Data changed since their last clean.
///
/// The Data must be added to tracking system by calling "trackData".
/// Then it can be checked if it changed with "hasChanged" since its last "clean".
///
/// Use datatrackers to check if your data have changed! Do not use
/// BaseData's "isDirty()" method, as it has a completely different purpose:
/// BaseData::isDirty() checks whether or not the data is up-to-date with its
/// parent values while DataTracker::hasChanged(myData) checks whether the data
/// has been modified since it has last been checked
struct SOFA_CORE_API DataTracker
{
    /// select a Data to track to be able to check
    /// if it was dirtied since the previous clean.
    /// @see isTrackedDataDirty
    void trackData( const objectmodel::BaseData& data );

    /// Did the data change since its last access?
    /// @warning data must be a tracked Data @see trackData
    bool hasChanged( const objectmodel::BaseData& data ) const;

    /// Did one of the tracked data change since the last call to clean()?
    bool hasChanged() const;

    /// comparison point is cleaned for the specified tracked Data
    /// @warning data must be a tracked Data @see trackData
    void clean( const objectmodel::BaseData& data );

    /// comparison point is cleaned for all tracked Data
    void clean();

    /// Provide the map of tracked Data
    const std::map<const objectmodel::BaseData*,int>&  getMapTrackedData() {return m_dataTrackers;}

protected:
    /// map a tracked Data to a DataTracker (storing its call-counter at each 'clean')
    typedef std::map<const objectmodel::BaseData*,int> DataTrackers;
    DataTrackers m_dataTrackers;
};

} // namespace sofa::core

