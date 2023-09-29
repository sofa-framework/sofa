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

namespace sofa::component::engine::select
{


/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given sphere.
 */
template <class DataTypes>
class SphereROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SphereROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef type::Vec<3,Real> Vec3;
    typedef type::Vec<6,Real> Vec6;
    typedef type::vector<sofa::core::topology::BaseMeshTopology::EdgeID> SetEdge;
    typedef type::vector<sofa::core::topology::BaseMeshTopology::TriangleID> SetTriangle;
    typedef type::vector<sofa::core::topology::BaseMeshTopology::QuadID> SetQuad;
    typedef sofa::core::topology::BaseMeshTopology::SetIndex SetIndex;

    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::Quad Quad;

protected:

    SphereROI();

    ~SphereROI() override {}
public:
    void init() override;

    void reinit() override;

    void doUpdate() override;

    void draw(const core::visual::VisualParams* vparams) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == nullptr)
            {
                arg->logError(std::string("No mechanical state with the datatype '") + DataTypes::Name() +
                              "' found in the context node.");
                return false; // this template is not the same as the existing MechanicalState
            }
        }

        return BaseObject::canCreate(obj, context, arg);
    }

protected:
	bool isPointInSphere(const Vec3& c, const Real& r, const Coord& p);
    bool isPointInSphere(const PointID& pid, const Real& r, const Coord& p);
    bool isEdgeInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Edge& edge);
    bool isTriangleInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Triangle& triangle);
    bool isQuadInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Quad& quad);
    bool isTetrahedronInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Tetra& tetrahedron);

public:
    //Input
    Data< type::vector<Vec3> > centers; ///< Center(s) of the sphere(s)
    Data< type::vector<Real> > radii; ///< Radius(i) of the sphere(s)

    Data< Vec3 > direction; ///< Edge direction(if edgeAngle > 0)
    Data< Vec3 > normal; ///< Normal direction of the triangles (if triAngle > 0)
    Data< Real > edgeAngle; ///< Max angle between the direction of the selected edges and the specified direction
    Data< Real > triAngle; ///< Max angle between the normal of the selected triangle and the specified normal direction

    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    Data<type::vector<Edge> > f_edges; ///< Edge Topology
    Data<type::vector<Triangle> > f_triangles; ///< Triangle Topology
    Data<type::vector<Quad> > f_quads; ///< Quads Topology
    Data<type::vector<Tetra> > f_tetrahedra; ///< Tetrahedron Topology
    Data<bool> f_computeEdges; ///< If true, will compute edge list and index list inside the ROI.
    Data<bool> f_computeTriangles; ///< If true, will compute triangle list and index list inside the ROI.
    Data<bool> f_computeQuads; ///< If true, will compute quad list and index list inside the ROI.
    Data<bool> f_computeTetrahedra; ///< If true, will compute tetrahedra list and index list inside the ROI.

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> f_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> f_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> f_quadIndices; ///< Indices of the quads contained in the ROI
    Data<SetIndex> f_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI


    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI
    Data<type::vector<Edge> > f_edgesInROI; ///< Edges contained in the ROI
    Data<type::vector<Triangle> > f_trianglesInROI; ///< Triangles contained in the ROI
    Data<type::vector<Quad> > f_quadsInROI; ///< Quads contained in the ROI
    Data<type::vector<Tetra> > f_tetrahedraInROI; ///< Tetrahedra contained in the ROI
    Data<SetIndex> f_indicesOut; ///< Indices of the points not contained in the ROI

    //Parameter
    Data<bool> p_drawSphere; ///< Draw shpere(s)
    Data<bool> p_drawPoints; ///< Draw Points
    Data<bool> p_drawEdges; ///< Draw Edges
    Data<bool> p_drawTriangles; ///< Draw Triangles
    Data<bool> p_drawQuads; ///< Draw Quads
    Data<bool> p_drawTetrahedra; ///< Draw Tetrahedra
    Data<float> _drawSize; ///< rendering size for box and topological elements

};

template<> bool SphereROI<defaulttype::Rigid3Types>::isPointInSphere(const Vec3& c, const Real& r, const Coord& p);
template<> bool SphereROI<defaulttype::Rigid3Types>::isPointInSphere(const PointID& pid, const Real& r, const Coord& p);
template<> bool SphereROI<defaulttype::Rigid3Types>::isEdgeInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Edge& edge);
template<> bool SphereROI<defaulttype::Rigid3Types>::isTriangleInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Triangle& triangle);
template<> bool SphereROI<defaulttype::Rigid3Types>::isQuadInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Quad& quad);
template<> bool SphereROI<defaulttype::Rigid3Types>::isTetrahedronInSphere(const Vec3& c, const Real& r, const sofa::core::topology::BaseMeshTopology::Tetra& tetrahedron);
template<> void SphereROI<defaulttype::Rigid3Types>::doUpdate();


#if !defined(SOFA_COMPONENT_ENGINE_SPHEREROI_CPP)
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_ENGINE_SELECT_API SphereROI<defaulttype::Rigid3Types>;
#endif

} //namespace sofa::component::engine::select
