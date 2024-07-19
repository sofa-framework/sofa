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

protected:
    MeshROI();
    ~MeshROI() override = default;

public:
    void roiInit() override;
    bool roiDoUpdate() override;
    void roiDraw(const core::visual::VisualParams* vparams) override;

protected:
    bool checkSameOrder(const CPos& A, const CPos& B, const CPos& pt, const CPos& norm);

    bool isPointInIndices(const unsigned int& i);
    bool isPointInBoundingBox(const CPos& p);

    bool isPointInMesh(const CPos& p);
    bool isEdgeInMesh(const Edge& e);
    bool isTriangleInMesh(const Triangle& t);
    bool isTetrahedronInMesh(const Tetra& t);

    bool isPointIn(const CPos& p) override;
    bool isEdgeIn(const Edge& e) override;
    bool isEdgeInStrict(const Edge& e) override;
    bool isTriangleIn(const Triangle& t) override;
    bool isTriangleInStrict(const Triangle& t) override;
    bool isQuadIn(const Quad& q) override;
    bool isQuadInStrict(const Quad& q) override;
    bool isTetrahedronIn(const Tetra& t) override;
    bool isTetrahedronInStrict(const Tetra& t) override;
    bool isHexahedronIn(const Hexa& t) override;
    bool isHexahedronInStrict(const Hexa& t) override;


protected:
    void checkInputData();
    void computeBoundingBox();

public:
    //Input
    // ROI mesh
    Data<VecCoord> d_X0_i; ///< ROI position coordinates of the degrees of freedom
    Data<type::vector<Edge> > d_edges_i; ///< ROI Edge Topology
    Data<type::vector<Triangle> > d_triangles_i; ///< ROI Triangle Topology
    Data<bool> d_computeTemplateTriangles; ///< Compute with the mesh (not only bounding box)

    //Output
    Data<type::Vec6> d_box; ///< Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax

    //Parameter
    Data<bool> d_drawOut; ///< Draw the data not contained in the ROI
    Data<bool> d_drawBox; ///< Draw the Bounding box around the mesh used for the ROI
};

#if !defined(SOFA_COMPONENT_ENGINE_MESHROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API MeshROI<defaulttype::Vec6Types>;
#endif

} //namespace sofa::component::engine::select
