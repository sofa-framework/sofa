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
#include <sofa/component/engine/select/config.h>

#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/type/SVector.h>

namespace sofa::component::engine::select
{

/**
 * This class outputs indices of boundary vertices of a triangle/quad mesh
 * The boundary is detected using the number elements associated to the edges.
 * An edge is considered on the boundary if it has a unique associated element.
 * @author benjamin gilles
 */
class MeshBoundaryROI : public core::DataEngine
{
public:
    SOFA_CLASS(MeshBoundaryROI, DataEngine);
    SOFA_ATTRIBUTE_REPLACED__TYPEMEMBER(Inherited, Inherit1);

    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::SeqTriangles SeqTriangles;
    typedef core::topology::BaseMeshTopology::SeqQuads SeqQuads;
    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef std::pair<PointID, PointID> PointPair;

    /// inputs
    Data< SeqTriangles > d_triangles;
    Data< SeqQuads > d_quads; ///< input quads
    Data< SetIndex > d_inputROI; ///< optional subset of the input mesh

    /// outputs
    Data< SetIndex > d_indices;

protected:

    MeshBoundaryROI();

    ~MeshBoundaryROI() override {}

public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    /// edge is used as a key to be found in the edgeCount map. If found, its value is incremented. Otherwise,
    /// the value is set to 1.
    static void countEdge(std::map<PointPair, unsigned int>& edgeCount, PointPair& edge);

    /// Check if the point with PointID index is part of the indices defined in d_inputROI.
    /// @return true if d_inputROI is empty or if index is in d_inputROI, false otherwise.
    inline bool inROI(const PointID& index) const;
};

} //namespace sofa::component::engine::select
