/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

/// This namespace is used to avoid the leaking of the 'using' on includes.
/// BoxROI is defined in namespace in sofa::component::engine::boxroi:BoxROI
/// It is then import into sofa::component::engine::BoxROI to not break the
/// API.
namespace boxroi
{
    using core::objectmodel::BaseObjectDescription ;
    using sofa::core::behavior::MechanicalState ;
    using core::topology::BaseMeshTopology ;
    using core::behavior::MechanicalState ;
    using core::objectmodel::BaseContext ;
    using core::objectmodel::BaseObject ;
    using core::visual::VisualParams ;
    using core::objectmodel::Event ;
    using core::ExecParams ;
    using core::DataEngine ;
    using helper::vector ;
    using std::string ;



/**
 * This class find all the points/edges/triangles/quads/tetrahedras/hexahedras located inside given boxes.
 */
template <class DataTypes>
class BoxROI : public DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxROI,DataTypes), DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef defaulttype::Vec<10,Real> Vec10;
    typedef BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
    typedef BaseMeshTopology::Edge Edge;
    typedef BaseMeshTopology::Triangle Triangle;
    typedef BaseMeshTopology::Tetra Tetra;
    typedef BaseMeshTopology::Hexa Hexa;
    typedef BaseMeshTopology::Quad Quad;

public:
    void init();
    void reinit();
    void update();
    void draw(const VisualParams*);

    virtual void computeBBox(const ExecParams*  params, bool onlyVisible=false );
    virtual void handleEvent(Event *event);

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, BaseContext* context, BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false; // this template is not the same as the existing MechanicalState
        }

        return BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, BaseContext* context, BaseObjectDescription* arg)
    {
        return BaseObject::create(tObj, context, arg);
    }

    virtual string getTemplateName() const
    {
        return templateName(this);
    }

    static string templateName(const BoxROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    //Input
    Data<vector<Vec6> >  d_alignedBoxes; ///< each box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    Data<vector<Vec10> > d_orientedBoxes; ///< each box is defined using three point coordinates and a depth value
    Data<VecCoord> d_X0;
    Data<vector<Edge> > d_edges;
    Data<vector<Triangle> > d_triangles;
    Data<vector<Tetra> > d_tetrahedra;
    Data<vector<Hexa> > d_hexahedra;
    Data<vector<Quad> > d_quad;
    Data<bool> d_computeEdges;
    Data<bool> d_computeTriangles;
    Data<bool> d_computeTetrahedra;
    Data<bool> d_computeHexahedra;
    Data<bool> d_computeQuad;

    //Output
    Data<SetIndex> d_indices;
    Data<SetIndex> d_edgeIndices;
    Data<SetIndex> d_triangleIndices;
    Data<SetIndex> d_tetrahedronIndices;
    Data<SetIndex> d_hexahedronIndices;
    Data<SetIndex> d_quadIndices;
    Data<VecCoord > d_pointsInROI;
    Data<vector<Edge> > d_edgesInROI;
    Data<vector<Triangle> > d_trianglesInROI;
    Data<vector<Tetra> > d_tetrahedraInROI;
    Data<vector<Hexa> > d_hexahedraInROI;
    Data<vector<Quad> > d_quadInROI;
    Data< unsigned int > d_nbIndices;

    //Parameter
    Data<bool> d_drawBoxes;
    Data<bool> d_drawPoints;
    Data<bool> d_drawEdges;
    Data<bool> d_drawTriangles;
    Data<bool> d_drawTetrahedra;
    Data<bool> d_drawHexahedra;
    Data<bool> d_drawQuads;
    Data<double> d_drawSize;
    Data<bool> d_doUpdate;

    /// Deprecated input parameters... should be kept until
    /// the corresponding attribute is not supported any more.
    Data<VecCoord> d_deprecatedX0;
    Data<bool> d_deprecatedIsVisible;


protected:

    struct OrientedBox
    {
        typedef typename DataTypes::Real Real;
        typedef defaulttype::Vec<3,Real> Vec3;

        Vec3 p0, p2;
        Vec3 normal;
        Vec3 plane0, plane1, plane2, plane3;
        double width, length, depth;
    };

    vector<OrientedBox> m_orientedBoxes;

    BoxROI();
    ~BoxROI() {}

    void computeOrientedBoxes();

    bool isPointInOrientedBox(const CPos& p, const OrientedBox& box);
    bool isPointInAlignedBox(const typename DataTypes::CPos& p, const Vec6& box);
    bool isPointInBoxes(const CPos& p);
    bool isPointInBoxes(const PointID& pid);
    bool isEdgeInBoxes(const Edge& e);
    bool isTriangleInBoxes(const Triangle& t);
    bool isTetrahedronInBoxes(const Tetra& t);
    bool isHexahedronInBoxes(const Hexa& t);
    bool isQuadInBoxes(const Quad& q);

    void getPointsFromOrientedBox(const Vec10& box, vector<Vec3> &points);
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

} // namespace boxroi

/// Import sofa::component::engine::boxroi::BoxROI into
/// into the sofa::component::engine namespace.
using boxroi::BoxROI ;

} // namespace engine

} // namespace component

} // namespace sofa

#endif
