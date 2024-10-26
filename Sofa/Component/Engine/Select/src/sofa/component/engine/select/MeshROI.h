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



#include <sofa/type/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/RigidTypes.h>

#include <sofa/component/engine/select/BaseROI.h>

#include <sofa/core/objectmodel/RenamedData.h>

namespace sofa::component::engine::select
{

/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given Mesh.
 */
template <class DataTypes>
class MeshROI : public BaseROI<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshROI, DataTypes), SOFA_TEMPLATE(BaseROI, DataTypes));
    using Inherit = BaseROI<DataTypes>;

    using VecCoord = VecCoord_t<DataTypes>;
    using Real = Real_t<DataTypes>;
    using typename Inherit::SetIndex;
    using typename Inherit::CPos;
    using typename Inherit::Edge;
    using typename Inherit::Triangle;
    using typename Inherit::Quad;
    using typename Inherit::Tetra;
    using typename Inherit::Hexa;

protected:
    MeshROI();
    ~MeshROI() override = default;

public:
    void roiInit() override;
    bool roiDoUpdate() override;
    void roiDraw(const core::visual::VisualParams* vparams) override;
    void roiComputeBBox(const core::ExecParams* params, type::BoundingBox& bbox) override;

protected:
    bool checkSameOrder(const CPos& A, const CPos& B, const CPos& pt, const CPos& norm) const;

    bool isPointInIndices(const unsigned int i) const;
    bool isPointInBoundingBox(const CPos& p) const;

    bool isPointInROI(const CPos& p) const override;
    bool isEdgeInROI(const Edge& e) const override;
    bool isEdgeInStrictROI(const Edge& e) const override;
    bool isTriangleInROI(const Triangle& t) const override;
    bool isTriangleInStrictROI(const Triangle& t) const override;
    bool isQuadInROI(const Quad& q) const override;
    bool isQuadInStrictROI(const Quad& q) const override;
    bool isTetrahedronInROI(const Tetra& t) const override;
    bool isTetrahedronInStrictROI(const Tetra& t) const override;
    bool isHexahedronInROI(const Hexa& t) const override;
    bool isHexahedronInStrictROI(const Hexa& t) const override;

protected:
    void checkInputData();
    void computeBoundingBox();

public:
    //Input
    // ROI mesh
    Data<VecCoord> d_roiPositions; ///< ROI position coordinates of the degrees of freedom
    Data<type::vector<Edge> > d_roiEdges; ///< ROI Edge Topology
    Data<type::vector<Triangle> > d_roiTriangles; ///< ROI Triangle Topology
    Data<bool> d_computeTemplateTriangles; ///< Compute with the mesh (not only bounding box)

    //Output
    Data<type::Vec6> d_box; ///< Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax

    //Parameter
    Data<bool> d_drawOut; ///< Draw the data not contained in the ROI
    Data<bool> d_drawBox; ///< Draw the Bounding box around the mesh used for the ROI

    static bool isPointInIndices(const unsigned int i, const SetIndex& indices);

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData<VecCoord> d_X0_i;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData<type::vector<Edge> > d_edges_i;;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ENGINE_SELECT()
    sofa::core::objectmodel::RenamedData<type::vector<Triangle> > d_triangles_i;
};

#if !defined(SOFA_COMPONENT_ENGINE_MESHROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Vec6Types>;
#endif

} //namespace sofa::component::engine::select
