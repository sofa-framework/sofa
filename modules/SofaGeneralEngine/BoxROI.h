/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ENGINE_BOXROI_H
#define SOFA_COMPONENT_ENGINE_BOXROI_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given box.
 */
template <class DataTypes>
class BoxROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::Hexa Hexa;
    typedef core::topology::BaseMeshTopology::Quad Quad;

protected:

    BoxROI();

    ~BoxROI() {}
public:
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams*);

    virtual void computeBBox(const core::ExecParams*  params, bool onlyVisible=false );

    virtual void handleEvent(core::objectmodel::Event *event);


    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false; // this template is not the same as the existing MechanicalState
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::create(tObj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const BoxROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected:
    bool isPointInBox(const CPos& p, const Vec6& b);
    bool isPointInBox(const PointID& pid, const Vec6& b);
    bool isEdgeInBox(const Edge& e, const Vec6& b);
    bool isTriangleInBox(const Triangle& t, const Vec6& b);
    bool isTetrahedronInBox(const Tetra& t, const Vec6& b);
    bool isHexahedronInBox(const Hexa& t, const Vec6& b);
    bool isQuadInBox(const Quad& q, const Vec6& b);

public:
    //Input
    Data< helper::vector<Vec6> > boxes; ///< each box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    Data<helper::vector<Edge> > f_edges; ///< Edge Topology
    Data<helper::vector<Triangle> > f_triangles; ///< Triangle Topology
    Data<helper::vector<Tetra> > f_tetrahedra; ///< Tetrahedron Topology
    Data<helper::vector<Hexa> > f_hexahedra; ///< Hexahedron Topology
    Data<helper::vector<Quad> > f_quad; ///< Quad Topology
    Data<bool> f_computeEdges; ///< If true, will compute edge list and index list inside the ROI.
    Data<bool> f_computeTriangles; ///< If true, will compute triangle list and index list inside the ROI.
    Data<bool> f_computeTetrahedra; ///< If true, will compute tetrahedra list and index list inside the ROI.
    Data<bool> f_computeHexahedra; ///< If true, will compute hexahedra list and index list inside the ROI.
    Data<bool> f_computeQuad; ///< If true, will compute quad list and index list inside the ROI.

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> f_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> f_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> f_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI
    Data<SetIndex> f_hexahedronIndices; ///< Indices of the hexahedra contained in the ROI
    Data<SetIndex> f_quadIndices; ///< Indices of the quad contained in the ROI
    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI
    Data<helper::vector<Edge> > f_edgesInROI; ///< Edges contained in the ROI
    Data<helper::vector<Triangle> > f_trianglesInROI; ///< Triangles contained in the ROI
    Data<helper::vector<Tetra> > f_tetrahedraInROI; ///< Tetrahedra contained in the ROI
    Data<helper::vector<Hexa> > f_hexahedraInROI; ///< Hexahedra contained in the ROI
    Data<helper::vector<Quad> > f_quadInROI; ///< Quad contained in the ROI
	Data< unsigned int > f_nbIndices; ///< Number of selected indices

    //Parameter
    Data<bool> p_drawBoxes; ///< Draw Box(es)
    Data<bool> p_drawPoints; ///< Draw Points
    Data<bool> p_drawEdges; ///< Draw Edges
    Data<bool> p_drawTriangles; ///< Draw Triangles
    Data<bool> p_drawTetrahedra; ///< Draw Tetrahedra
    Data<bool> p_drawHexahedra; ///< Draw Tetrahedra
    Data<bool> p_drawQuads; ///< Draw Quads
    Data<double> _drawSize; ///< rendering size for box and topological elements
    Data<bool> p_doUpdate; ///< Boolean for updating the Box
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_BOXROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Vec3dTypes>;
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Rigid3dTypes>;
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Vec6dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Vec3fTypes>;
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Rigid3fTypes>;
extern template class SOFA_ENGINE_API BoxROI<defaulttype::Vec6fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
