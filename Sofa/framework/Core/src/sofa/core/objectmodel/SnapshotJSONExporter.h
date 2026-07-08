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
#include <sofa/core/objectmodel/Snapshot.h>

namespace sofa::core::objectmodel
{

    /// Export a single Snapshot to a JSON file
    SOFA_CORE_API void exportToJSON(Snapshot& snapshot, const std::string& filename);

    /// Import a single Snapshot from a JSON file
    SOFA_CORE_API void importFrom(Snapshot& snapshot, const std::string& filename);

    /// Read a JSON file and returns its content as a string
    SOFA_CORE_API std::string fileToString(const std::string& filename);

    /// Serialize a Snapshot to a JSON string
    SOFA_CORE_API std::string snapshotToString(const Snapshot& snapshot);

    /// Export a collection of Snapshots to a single JSON file
    SOFA_CORE_API void exportToJSON(std::map<std::string, std::shared_ptr<Snapshot>>& snapshots, const std::string& filename);

    /// Import a collection of Snapshots from a single JSON file
    SOFA_CORE_API void importFrom(std::map<std::string, std::shared_ptr<Snapshot>>& snapshots, const std::string& filename);

} // namespace sofa::core::objectmodel
