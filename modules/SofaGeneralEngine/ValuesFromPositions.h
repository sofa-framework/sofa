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
#ifndef SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_H
#define SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_H
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
#include <sofa/helper/OptionsGroup.h>

namespace sofa
{

namespace component
{

namespace engine
{

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
    typedef defaulttype::Vec<3, Real> Vec3;

    typedef unsigned int PointID;
    typedef core::topology::BaseMeshTopology::Edge Edge;
    typedef core::topology::BaseMeshTopology::Triangle Triangle;
    typedef core::topology::BaseMeshTopology::Tetra Tetra;

protected:

    ValuesFromPositions();

    ~ValuesFromPositions() {}
public:
    void init();

    void reinit();

    void update();

    void draw(const core::visual::VisualParams* vparams);

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

    void updateValues(TempData& _data);
    Real valueFromPosition(const CPos& p, const TempData& data);
    Real valueFromPoint(const PointID& pid, const TempData& data);
    Real valueFromEdge(const Edge& e, const TempData& data);
    Real valueFromTriangle(const Triangle& t, const TempData& data);
    Real valueFromTetrahedron(const Tetra &t, const TempData& data);

    void updateVectors(TempData& _data);
    Vec3 vectorFromPosition(const CPos& p, const TempData& data);
    Vec3 vectorFromPoint(const PointID& pid, const TempData& data);
    Vec3 vectorFromEdge(const Edge& e, const TempData& data);
    Vec3 vectorFromTriangle(const Triangle& t, const TempData& data);
    Vec3 vectorFromTetrahedron(const Tetra &t, const TempData& data);

public:
    //Input
    Data<VecReal> f_inputValues;
    Data<CPos> f_direction;
    Data<VecCoord> f_X0;
    Data<helper::vector<Edge> > f_edges;
    Data<helper::vector<Triangle> > f_triangles;
    Data<helper::vector<Tetra> > f_tetrahedra;

    //Output scalars
    Data<VecReal> f_values;
    Data<VecReal> f_edgeValues;
    Data<VecReal> f_triangleValues;
    Data<VecReal> f_tetrahedronValues;

    //Output vectors
    Data<sofa::helper::vector<Vec3> > f_pointVectors;
    Data<sofa::helper::vector<Vec3> > f_edgeVectors;
    Data<sofa::helper::vector<Vec3> > f_triangleVectors;
    Data<sofa::helper::vector<Vec3> > f_tetrahedronVectors;

    // parameters
    sofa::core::objectmodel::Data< sofa::helper::OptionsGroup > p_fieldType;
    Data <bool> p_drawVectors;
    Data <float> p_vectorLength;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_ENGINE_VALUESFROMPOSITIONS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_ENGINE_API ValuesFromPositions<defaulttype::Vec3dTypes>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromPositions<defaulttype::Rigid3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_ENGINE_API ValuesFromPositions<defaulttype::Vec3fTypes>;
extern template class SOFA_GENERAL_ENGINE_API ValuesFromPositions<defaulttype::Rigid3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif
