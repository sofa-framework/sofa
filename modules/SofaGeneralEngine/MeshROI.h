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
    Data<VecCoord> d_X0; ///< Rest position coordinates of the degrees of freedom
    Data<helper::vector<Edge> > d_edges; ///< Edge Topology
    Data<helper::vector<Triangle> > d_triangles; ///< Triangle Topology
    Data<helper::vector<Tetra> > d_tetrahedra; ///< Tetrahedron Topology
    // ROI mesh
    Data<VecCoord> d_X0_i; ///< ROI position coordinates of the degrees of freedom
    Data<helper::vector<Edge> > d_edges_i; ///< ROI Edge Topology
    Data<helper::vector<Triangle> > d_triangles_i; ///< ROI Triangle Topology

    Data<bool> d_computeEdges; ///< If true, will compute edge list and index list inside the ROI.
    Data<bool> d_computeTriangles; ///< If true, will compute triangle list and index list inside the ROI.
    Data<bool> d_computeTetrahedra; ///< If true, will compute tetrahedra list and index list inside the ROI.
    Data<bool> d_computeTemplateTriangles; ///< Compute with the mesh (not only bounding box)

    //Output
    Data<Vec6> d_box; ///< Bounding box defined by xmin,ymin,zmin, xmax,ymax,zmax
    Data<SetIndex> d_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> d_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> d_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> d_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI
    Data<VecCoord > d_pointsInROI; ///< Points contained in the ROI
    Data<helper::vector<Edge> > d_edgesInROI; ///< Edges contained in the ROI
    Data<helper::vector<Triangle> > d_trianglesInROI; ///< Triangles contained in the ROI
    Data<helper::vector<Tetra> > d_tetrahedraInROI; ///< Tetrahedra contained in the ROI

    Data<VecCoord > d_pointsOutROI; ///< Points not contained in the ROI
    Data<helper::vector<Edge> > d_edgesOutROI; ///< Edges not contained in the ROI
    Data<helper::vector<Triangle> > d_trianglesOutROI; ///< Triangles not contained in the ROI
    Data<helper::vector<Tetra> > d_tetrahedraOutROI; ///< Tetrahedra not contained in the ROI
    Data<SetIndex> d_indicesOut; ///< Indices of the points not contained in the ROI
    Data<SetIndex> d_edgeOutIndices; ///< Indices of the edges not contained in the ROI
    Data<SetIndex> d_triangleOutIndices; ///< Indices of the triangles not contained in the ROI
    Data<SetIndex> d_tetrahedronOutIndices; ///< Indices of the tetrahedra not contained in the ROI

    //Parameter
    Data<bool> d_drawOut; ///< Draw the data not contained in the ROI
    Data<bool> d_drawMesh; ///< Draw Mesh used for the ROI
    Data<bool> d_drawBox; ///< Draw the Bounding box around the mesh used for the ROI
    Data<bool> d_drawPoints; ///< Draw Points
    Data<bool> d_drawEdges; ///< Draw Edges
    Data<bool> d_drawTriangles; ///< Draw Triangles
    Data<bool> d_drawTetrahedra; ///< Draw Tetrahedra
    Data<double> d_drawSize; ///< rendering size for mesh and topological elements
    Data<bool> d_doUpdate; ///< Update the computation (not only at the init)
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
