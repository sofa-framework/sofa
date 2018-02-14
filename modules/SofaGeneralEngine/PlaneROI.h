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
#ifndef SOFA_COMPONENT_ENGINE_PLANEROI_H
#define SOFA_COMPONENT_ENGINE_PLANEROI_H
#include "config.h"

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace engine
{

/**
 * This class find all the points located inside a given box.
 */
template <class DataTypes>
class PlaneROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PlaneROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef defaulttype::Vec<10,Real> Vec10;
    typedef sofa::core::topology::BaseMeshTopology::SetIndex SetIndex;

    typedef typename DataTypes::CPos CPos;

    typedef defaulttype::Vec<3,Real> Point;
    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;

protected:

    PlaneROI();

    ~PlaneROI() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    void draw(const core::visual::VisualParams* vparams) override;

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

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const PlaneROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected:
    bool isPointInPlane(const CPos& p);
    bool isPointInPlane(const PointID& pid);
    bool isEdgeInPlane(const Edge& e);
    bool isTriangleInPlane(const Triangle& t);
    bool isTetrahedronInPlane(const Tetra& t);

    void computePlane(unsigned int planeIndex);


public:
    //Input
    Data< helper::vector<Vec10> > planes; ///< Plane defined by 3 points and a depth distance
    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    Data<helper::vector<Edge> > f_edges; ///< Edge Topology
    Data<helper::vector<Triangle> > f_triangles; ///< Triangle Topology
    Data<helper::vector<Tetra> > f_tetrahedra; ///< NOT YET
    Data<bool> f_computeEdges; ///< If true, will compute edge list and index list inside the ROI.
    Data<bool> f_computeTriangles; ///< If true, will compute triangle list and index list inside the ROI.
    Data<bool> f_computeTetrahedra; ///< If true, will compute tetrahedra list and index list inside the ROI.

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> f_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> f_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> f_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI
    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI
    Data<helper::vector<Edge> > f_edgesInROI; ///< Edges contained in the ROI
    Data<helper::vector<Triangle> > f_trianglesInROI; ///< Triangles contained in the ROI
    Data<helper::vector<Tetra> > f_tetrahedraInROI; ///< Tetrahedra contained in the ROI

    //Parameter
    Data<bool> p_drawBoxes; ///< Draw Box(es)
    Data<bool> p_drawPoints; ///< Draw Points
    Data<bool> p_drawEdges; ///< Draw Edges
    Data<bool> p_drawTriangles; ///< Draw Triangles
    Data<bool> p_drawTetrahedra; ///< Draw Tetrahedra
    Data<double> _drawSize; ///< rendering size for box and topological elements

private:

    Vec3 p0, p1, p2, p3, p4, p5, p6, p7, plane0, plane1, plane2, plane3, vdepth;
    Real width, length, depth;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_PLANEROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API PlaneROI<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API PlaneROI<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API PlaneROI<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API PlaneROI<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
