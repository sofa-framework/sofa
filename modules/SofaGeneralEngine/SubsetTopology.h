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
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

    void computeBBox(const core::ExecParams* params, bool onlyVisible=false );

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
    Data< helper::vector<Vec6> > boxes;

    //For sphere
    Data< helper::vector<Vec3> > centers;
    Data< helper::vector<Real> > radii;
    Data< Vec3 > direction;
    Data< Vec3 > normal;
    Data< Real > edgeAngle;
    Data< Real > triAngle;

    Data<VecCoord> f_X0;
    Data<helper::vector<Edge> > f_edges;
    Data<helper::vector<Triangle> > f_triangles;
    Data<helper::vector<Quad> > f_quads;
    Data<helper::vector<Tetra> > f_tetrahedra;
    Data<helper::vector<Hexa> > f_hexahedra;
    Data<SetIndex> d_tetrahedraInput;

    //Output
    Data<SetIndex> f_indices;
    Data<SetIndex> f_edgeIndices;
    Data<SetIndex> f_triangleIndices;
    Data<SetIndex> f_quadIndices;
    Data<SetIndex> f_tetrahedronIndices;
    Data<SetIndex> f_hexahedronIndices;
    Data<VecCoord > f_pointsInROI;
    Data<VecCoord > f_pointsOutROI;
    Data<helper::vector<Edge> > f_edgesInROI;
    Data<helper::vector<Edge> > f_edgesOutROI;
    Data<helper::vector<Triangle> > f_trianglesInROI;
    Data<helper::vector<Triangle> > f_trianglesOutROI;
    Data<helper::vector<Quad> > f_quadsInROI;
    Data<helper::vector<Quad> > f_quadsOutROI;
    Data<helper::vector<Tetra> > f_tetrahedraInROI;
    Data<helper::vector<Tetra> > f_tetrahedraOutROI;
    Data<helper::vector<Hexa> > f_hexahedraInROI;
    Data<helper::vector<Hexa> > f_hexahedraOutROI;
    Data<unsigned int> f_nbrborder;

    //Parameter
    Data<bool> p_localIndices;
    Data<bool> p_drawROI;
    Data<bool> p_drawPoints;
    Data<bool> p_drawEdges;
    Data<bool> p_drawTriangles;
    Data<bool> p_drawTetrahedra;
    Data<double> _drawSize;

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
