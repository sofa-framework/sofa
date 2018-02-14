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
#ifndef SOFA_COMPONENT_ENGINE_SUBSETTOPOLOGY_H
#define SOFA_COMPONENT_ENGINE_SUBSETTOPOLOGY_H
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
class SubsetTopology : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(SubsetTopology,DataTypes),core::DataEngine);
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<6,Real> Vec6;
    typedef core::topology::BaseMeshTopology::SetIndex SetIndex;
    typedef typename DataTypes::CPos CPos;

    typedef defaulttype::Vec<3,Real> Vec3;
    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Quad Quad;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;
    typedef core::topology::BaseMeshTopology::Hexa Hexa;

protected:

    SubsetTopology();

    ~SubsetTopology() {}
public:
    void init() override;

    void reinit() override;

    void update() override;

    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible=false ) override;

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

    static std::string templateName(const SubsetTopology<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    bool isPointInROI(const CPos& p, unsigned int idROI);
    bool isPointInROI(const PointID& pid, unsigned int idROI);
    bool isEdgeInROI(const Edge& e, unsigned int idROI);
    bool isTriangleInROI(const Triangle& t, unsigned int idROI);
    bool isQuadInROI(const Quad& t, unsigned int idROI);
    bool isTetrahedronInROI(const Tetra& t, unsigned int idROI);
    bool isHexahedronInROI(const Hexa& t, unsigned int idROI);

    void findVertexOnBorder(const Triangle& t, unsigned int idROI);
    void findVertexOnBorder(const Tetra& t, unsigned int idROI);

	bool isPointChecked(unsigned int id, sofa::helper::vector<bool>& pointChecked);

public:
    enum ROIType
    {
        //boxROI
        BOX = 0,
        //sphereROI
        SPHERE = 1
    };

    //Input
    //For cube
    Data< helper::vector<Vec6> > boxes; ///< Box defined by xmin,ymin,zmin, xmax,ymax,zmax

    //For sphere
    Data< helper::vector<Vec3> > centers; ///< Center(s) of the sphere(s)
    Data< helper::vector<Real> > radii; ///< Radius(i) of the sphere(s)
    Data< Vec3 > direction; ///< Edge direction(if edgeAngle > 0)
    Data< Vec3 > normal; ///< Normal direction of the triangles (if triAngle > 0)
    Data< Real > edgeAngle; ///< Max angle between the direction of the selected edges and the specified direction
    Data< Real > triAngle; ///< Max angle between the normal of the selected triangle and the specified normal direction

    Data<VecCoord> f_X0; ///< Rest position coordinates of the degrees of freedom
    Data<helper::vector<Edge> > f_edges; ///< Edge Topology
    Data<helper::vector<Triangle> > f_triangles; ///< Triangle Topology
    Data<helper::vector<Quad> > f_quads; ///< Quad Topology
    Data<helper::vector<Tetra> > f_tetrahedra; ///< Tetrahedron Topology
    Data<helper::vector<Hexa> > f_hexahedra; ///< Hexahedron Topology
    Data<SetIndex> d_tetrahedraInput; ///< Indices of the tetrahedra to keep

    //Output
    Data<SetIndex> f_indices; ///< Indices of the points contained in the ROI
    Data<SetIndex> f_edgeIndices; ///< Indices of the edges contained in the ROI
    Data<SetIndex> f_triangleIndices; ///< Indices of the triangles contained in the ROI
    Data<SetIndex> f_quadIndices; ///< Indices of the quads contained in the ROI
    Data<SetIndex> f_tetrahedronIndices; ///< Indices of the tetrahedra contained in the ROI
    Data<SetIndex> f_hexahedronIndices; ///< Indices of the hexahedra contained in the ROI
    Data<VecCoord > f_pointsInROI; ///< Points contained in the ROI
    Data<VecCoord > f_pointsOutROI; ///< Points out of the ROI
    Data<helper::vector<Edge> > f_edgesInROI; ///< Edges contained in the ROI
    Data<helper::vector<Edge> > f_edgesOutROI; ///< Edges out of the ROI
    Data<helper::vector<Triangle> > f_trianglesInROI; ///< Triangles contained in the ROI
    Data<helper::vector<Triangle> > f_trianglesOutROI; ///< Triangles out of the ROI
    Data<helper::vector<Quad> > f_quadsInROI; ///< Quads contained in the ROI
    Data<helper::vector<Quad> > f_quadsOutROI; ///< Quads out of the ROI
    Data<helper::vector<Tetra> > f_tetrahedraInROI; ///< Tetrahedra contained in the ROI
    Data<helper::vector<Tetra> > f_tetrahedraOutROI; ///< Tetrahedra out of the ROI
    Data<helper::vector<Hexa> > f_hexahedraInROI; ///< Hexahedra contained in the ROI
    Data<helper::vector<Hexa> > f_hexahedraOutROI; ///< Hexahedra out of the ROI
    Data<unsigned int> f_nbrborder; ///< If localIndices option is activated, will give the number of vertices on the border of the ROI (being the n first points of each output Topology). 

    //Parameter
    Data<bool> p_localIndices; ///< If true, will compute local dof indices in topological elements
    Data<bool> p_drawROI; ///< Draw ROI
    Data<bool> p_drawPoints; ///< Draw Points
    Data<bool> p_drawEdges; ///< Draw Edges
    Data<bool> p_drawTriangles; ///< Draw Triangles
    Data<bool> p_drawTetrahedra; ///< Draw Tetrahedra
    Data<double> _drawSize; ///< rendering size for box and topological elements

    ROIType typeROI;
    sofa::helper::vector<unsigned int> localIndices;
    sofa::helper::vector<unsigned int> listOnBorder;

};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_SUBSETTOPOLOGY_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API SubsetTopology<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API SubsetTopology<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API SubsetTopology<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API SubsetTopology<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
