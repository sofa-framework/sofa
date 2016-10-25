/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
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
 * This class find all the points/edges/triangles/tetrahedra located inside a given box.
 */
template <class DataTypes>
class BoxROI : public DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(BoxROI,DataTypes), DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<6,Real> Vec6;
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
    Data<vector<Vec6> > boxes; ///< each box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    Data<VecCoord> f_X0;
    Data<vector<Edge> > f_edges;
    Data<vector<Triangle> > f_triangles;
    Data<vector<Tetra> > f_tetrahedra;
    Data<vector<Hexa> > f_hexahedra;
    Data<vector<Quad> > f_quad;
    Data<bool> f_computeEdges;
    Data<bool> f_computeTriangles;
    Data<bool> f_computeTetrahedra;
    Data<bool> f_computeHexahedra;
    Data<bool> f_computeQuad;

    //Output
    Data<SetIndex> f_indices;
    Data<SetIndex> f_edgeIndices;
    Data<SetIndex> f_triangleIndices;
    Data<SetIndex> f_tetrahedronIndices;
    Data<SetIndex> f_hexahedronIndices;
    Data<SetIndex> f_quadIndices;
    Data<VecCoord > f_pointsInROI;
    Data<vector<Edge> > f_edgesInROI;
    Data<vector<Triangle> > f_trianglesInROI;
    Data<vector<Tetra> > f_tetrahedraInROI;
    Data<vector<Hexa> > f_hexahedraInROI;
    Data<vector<Quad> > f_quadInROI;
    Data< unsigned int > f_nbIndices;

    //Parameter
    Data<bool> p_drawBoxes;
    Data<bool> p_drawPoints;
    Data<bool> p_drawEdges;
    Data<bool> p_drawTriangles;
    Data<bool> p_drawTetrahedra;
    Data<bool> p_drawHexahedra;
    Data<bool> p_drawQuads;
    Data<double> _drawSize;
    Data<bool> p_doUpdate;

    /// Deprecated input parameters... should be kept until
    /// the corresponding attribute is not supported any more.
    Data<VecCoord> d_deprecatedX0;
    Data<bool> d_deprecatedIsVisible;

protected:
    BoxROI();
    ~BoxROI() {}

    bool isPointInBox(const CPos& p, const Vec6& b);
    bool isPointInBox(const PointID& pid, const Vec6& b);
    bool isEdgeInBox(const Edge& e, const Vec6& b);
    bool isTriangleInBox(const Triangle& t, const Vec6& b);
    bool isTetrahedronInBox(const Tetra& t, const Vec6& b);
    bool isHexahedronInBox(const Hexa& t, const Vec6& b);
    bool isQuadInBox(const Quad& q, const Vec6& b);
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
