/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

    virtual ~MeshROI() {}
public:

    virtual void init() override;
    virtual void reinit() override;
    virtual void update() override;
    virtual void draw(const core::visual::VisualParams*) override;

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

    static std::string templateName(const MeshROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    bool checkSameOrder(const CPos& A, const CPos& B, const CPos& pt, const CPos& norm);
    bool isPointInMesh(const CPos& p);
    bool isPointInIndices(const unsigned int& i);
    bool isPointInBoundingBox(const CPos& p);
    bool isEdgeInMesh(const Edge& e);
    bool isTriangleInMesh(const Triangle& t);
    bool isTetrahedronInMesh(const Tetra& t);


    void compute();
    void checkInputData();
    void computeBoundingBox();

public:
    //Input
    // Global mesh
    Data<VecCoord> d_X0;
    Data<helper::vector<Edge> > d_edges;
    Data<helper::vector<Triangle> > d_triangles;
    Data<helper::vector<Tetra> > d_tetrahedra;
    // ROI mesh
    Data<VecCoord> d_X0_i;
    Data<helper::vector<Edge> > d_edges_i;
    Data<helper::vector<Triangle> > d_triangles_i;

    Data<bool> d_computeEdges;
    Data<bool> d_computeTriangles;
    Data<bool> d_computeTetrahedra;
    Data<bool> d_computeTemplateTriangles;

    //Output
    Data<Vec6> d_box;
    Data<SetIndex> d_indices;
    Data<SetIndex> d_edgeIndices;
    Data<SetIndex> d_triangleIndices;
    Data<SetIndex> d_tetrahedronIndices;
    Data<VecCoord > d_pointsInROI;
    Data<helper::vector<Edge> > d_edgesInROI;
    Data<helper::vector<Triangle> > d_trianglesInROI;
    Data<helper::vector<Tetra> > d_tetrahedraInROI;

    Data<VecCoord > d_pointsOutROI;
    Data<helper::vector<Edge> > d_edgesOutROI;
    Data<helper::vector<Triangle> > d_trianglesOutROI;
    Data<helper::vector<Tetra> > d_tetrahedraOutROI;
    Data<SetIndex> d_indicesOut;
    Data<SetIndex> d_edgeOutIndices;
    Data<SetIndex> d_triangleOutIndices;
    Data<SetIndex> d_tetrahedronOutIndices;

    //Parameter
    Data<bool> d_drawOut;
    Data<bool> d_drawMesh;
    Data<bool> d_drawBox;
    Data<bool> d_drawPoints;
    Data<bool> d_drawEdges;
    Data<bool> d_drawTriangles;
    Data<bool> d_drawTetrahedra;
    Data<double> d_drawSize;
    Data<bool> d_doUpdate;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_MESHROI_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Rigid3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Vec6dTypes>; //Phuoc
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Rigid3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API MeshROI<defaulttype::Vec6fTypes>; //Phuoc
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
