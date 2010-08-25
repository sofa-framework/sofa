/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_H
#define SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/loader/MeshLoader.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace core::behavior;
using namespace core::objectmodel;

/**
 * This class find all the points/edges/triangles/tetrahedra located inside a given box.
 */
template <class DataTypes>
class ValuesFromPositions : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ValuesFromPositions,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef helper::vector<Real> VecReal;
    typedef typename DataTypes::CPos CPos;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;

public:

    ValuesFromPositions();

    ~ValuesFromPositions() {}

    void init();

    void reinit();

    void update();

    void draw();

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
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
    static void create(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        core::objectmodel::BaseObject::create(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const ValuesFromPositions<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }


protected:
    struct TempData
    {
        CPos dir;
        Real bmin, bmax;
        VecReal inputValues;
        const VecCoord* x0;
    };
    Real valueFromPosition(const CPos& p, const TempData& data);
    Real valueFromPoint(const PointID& pid, const TempData& data);
    Real valueFromEdge(const Edge& e, const TempData& data);
    Real valueFromTriangle(const Triangle& t, const TempData& data);
    Real valueFromTetrahedron(const Tetra &t, const TempData& data);
public:
    //Input
    Data<VecReal> f_inputValues;
    Data<CPos> f_direction;
    Data<VecCoord> f_X0;
    Data<helper::vector<Edge> > f_edges;
    Data<helper::vector<Triangle> > f_triangles;
    Data<helper::vector<Tetra> > f_tetrahedra;

    //Output
    Data<VecReal> f_values;
    Data<VecReal> f_edgeValues;
    Data<VecReal> f_triangleValues;
    Data<VecReal> f_tetrahedronValues;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_ENGINE_API ValuesFromPositions<defaulttype::Vec3dTypes>;
template class SOFA_COMPONENT_ENGINE_API ValuesFromPositions<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_ENGINE_API ValuesFromPositions<defaulttype::Vec3fTypes>;
template class SOFA_COMPONENT_ENGINE_API ValuesFromPositions<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
