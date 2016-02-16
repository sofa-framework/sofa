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
#ifndef SOFA_COMPONENT_ENGINE_MESHROI_H
#define SOFA_COMPONENT_ENGINE_MESHROI_H
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
 * This class find all the points/edges/triangles/tetrahedra located inside a given Mesh.
 */
template <class DataTypes>
class MeshROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(MeshROI,DataTypes),core::DataEngine);
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

protected:

    MeshROI();

    ~MeshROI() {}
public:
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams*);

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

    static std::string templateName(const MeshROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    bool CheckSameOrder(const CPos& A, const CPos& B, const CPos& pt, const CPos& norm);
    bool isPointInMesh(const CPos& p);
    bool isPointInMesh(const CPos& p, const Vec6& b);
    bool isPointInMesh(const PointID& pid, const Vec6& b);
    bool isEdgeInMesh(const Edge& e, const Vec6& b);
    bool isTriangleInMesh(const Triangle& t, const Vec6& b);
    bool isTetrahedronInMesh(const Tetra& t, const Vec6& b);

public:
    //Input
    // Global mesh
    Data<VecCoord> f_X0;
    Data<helper::vector<Edge> > f_edges;
    Data<helper::vector<Triangle> > f_triangles;
    Data<helper::vector<Tetra> > f_tetrahedra;
    // ROI mesh
    Data<VecCoord> f_X0_i;
    Data<helper::vector<Edge> > f_edges_i;
    Data<helper::vector<Triangle> > f_triangles_i;

    Data<bool> f_computeEdges;
    Data<bool> f_computeTriangles;
    Data<bool> f_computeTetrahedra;
    Data<bool> f_computeTemplateTriangles;

    //Output
    Data<Vec6> f_box;
    Data<SetIndex> f_indices;
    Data<SetIndex> f_edgeIndices;
    Data<SetIndex> f_triangleIndices;
    Data<SetIndex> f_tetrahedronIndices;
    Data<VecCoord > f_pointsInROI;
    Data<helper::vector<Edge> > f_edgesInROI;
    Data<helper::vector<Triangle> > f_trianglesInROI;
    Data<helper::vector<Tetra> > f_tetrahedraInROI;

    Data<VecCoord > f_pointsOutROI;
    Data<helper::vector<Edge> > f_edgesOutROI;
    Data<helper::vector<Triangle> > f_trianglesOutROI;
    Data<helper::vector<Tetra> > f_tetrahedraOutROI;
    Data<SetIndex> f_indicesOut;
    Data<SetIndex> f_edgeOutIndices;
    Data<SetIndex> f_triangleOutIndices;
    Data<SetIndex> f_tetrahedronOutIndices;

    //Parameter
    Data<bool> p_drawOut;
    Data<bool> p_drawMesh;
    Data<bool> p_drawBox;
    Data<bool> p_drawPoints;
    Data<bool> p_drawEdges;
    Data<bool> p_drawTriangles;
    Data<bool> p_drawTetrahedra;
    Data<double> _drawSize;
    Data<bool> p_doUpdate;
    Data<bool> first;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MESHROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Vec3dTypes>;
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Rigid3dTypes>;
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Vec6dTypes>; //Phuoc
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Vec3fTypes>;
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Rigid3fTypes>;
extern template class SOFA_ENGINE_API MeshROI<defaulttype::Vec6fTypes>; //Phuoc
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
