/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_ENGINE_ROTATABLEBOXROI_H
#define SOFA_COMPONENT_ENGINE_ROTATABLEBOXROI_H
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
 * This class find all the points/edges/triangles/quads/tetrahedras/hexahedras located inside given rotated boxes.
 */
template <class DataTypes>
class RotatableBoxROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(RotatableBoxROI,DataTypes), core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::Vec<4,Real> Vec4;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::Hexa Hexa;
    typedef core::topology::BaseMeshTopology::Quad Quad;

public:
    void init() override;
    void reinit() override;
    void update() override;
    void draw(const core::visual::VisualParams*) override;

    virtual void computeBBox(const core::ExecParams* params, bool onlyVisible=false ) override;
    virtual void handleEvent(core::objectmodel::Event *event) override;

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (!arg->getAttribute("template"))
        {
            // only check if this template is correct if no template was given
            if (context->getMechanicalState() && dynamic_cast<core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
                return false; // this template is not the same as the existing MechanicalState
        }

        return core::objectmodel::BaseObject::canCreate(obj, context, arg);
    }

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T* tObj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        return core::objectmodel::BaseObject::create(tObj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const RotatableBoxROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

public:
    //Input
    Data<helper::vector<Vec6> >  d_alignedBoxes; ///< each box is defined using xmin, ymin, zmin, xmax, ymax, zmax
    Data<helper::vector<Vec3> >  d_rotations;    ///< each box is rotated according to specified data
    Data<VecCoord> d_X0;
    Data<helper::vector<Edge> > d_edges;
    Data<helper::vector<Triangle> > d_triangles;
    Data<helper::vector<Tetra> > d_tetrahedra;
    Data<helper::vector<Hexa> > d_hexahedra;
    Data<helper::vector<Quad> > d_quad;
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
    Data<helper::vector<Edge> > d_edgesInROI;
    Data<helper::vector<Triangle> > d_trianglesInROI;
    Data<helper::vector<Tetra> > d_tetrahedraInROI;
    Data<helper::vector<Hexa> > d_hexahedraInROI;
    Data<helper::vector<Quad> > d_quadInROI;
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


protected:
    struct RotatedBox
    {
        Vec4 upperPlane, lowerPlane;
        Vec4 frontPlane, backPlane;
        Vec4 leftPlane, rightPlane;
        helper::vector<Vec3> boxPoints;   // points to draw rotated boxes
    };

    helper::vector<RotatedBox> m_rotatedBoxes;

    RotatableBoxROI();
    ~RotatableBoxROI() {}

    void initialiseBoxes();
    void computeRotatedBoxes();
    void rotatePlane(Vec3& centerPlanePoint, Vec3& firstPlanePoint, Vec3& secondPlanePoint, Vec3& centerShift, const defaulttype::Quaternion& rotationData, Vec4& resPlane);
    void rotateBoxPoints(const Vec6& alignedBox, Vec3& centerShift, defaulttype::Quaternion& rotationData, helper::vector<Vec3> *resPoints);
    void rotatePoint(Vec3 initialPoint, Vec3& centerShift, const defaulttype::Quaternion& rotationData, Vec3& resPoint);

    bool inNegativeHalfOfSpace(const typename DataTypes::CPos& point, const Vec4& plane);
    bool isPointInRotatedBox(const CPos& p, const RotatedBox& box);
    bool isPointInBoxes(const CPos& p);
    bool isPointInBoxes(const PointID& pid);
    bool isEdgeInBoxes(const Edge& e);
    bool isTriangleInBoxes(const Triangle& t);
    bool isTetrahedronInBoxes(const Tetra& t);
    bool isHexahedronInBoxes(const Hexa& t);
    bool isQuadInBoxes(const Quad& q);
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_ROTATABLEBOXROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Vec3dTypes>;
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Rigid3dTypes>;
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Vec6dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Vec3fTypes>;
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Rigid3fTypes>;
extern template class SOFA_ENGINE_API RotatableBoxROI<defaulttype::Vec6fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
