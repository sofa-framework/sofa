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
#ifndef SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_H
#define SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <sofa/defaulttype/Vec.h>
#include <sofa/core/DataEngine.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/component/topology/PointSubset.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/helper/gl/BasicShapes.h>

namespace sofa
{

namespace component
{

namespace engine
{

using namespace core::componentmodel::behavior;
using namespace core::componentmodel::topology;
using namespace core::objectmodel;

/**
 * This class find all the points located inside a given sphere.
 */
template <class DataTypes>
class PrimitivesInSphereROI : public core::DataEngine
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PrimitivesInSphereROI,DataTypes),core::DataEngine);
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::Vec<3,Real> Vec3;
    typedef helper::vector<BaseMeshTopology::EdgeID> SetEdge;
    typedef helper::vector<BaseMeshTopology::TriangleID> SetTriangle;
    typedef topology::PointSubset SetIndex;

public:

    PrimitivesInSphereROI();

    ~PrimitivesInSphereROI() {}

    void init();

    void reinit();

    void update();

    void draw();

    bool addBBox(double* /*minBBox*/, double* /*maxBBox*/) { return false; };

    /// Pre-construction check method called by ObjectFactory.
    /// Check that DataTypes matches the MechanicalState.
    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
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

    static std::string templateName(const PrimitivesInSphereROI<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    Data<bool> isVisible;
    Data< helper::vector<Vec3> > centers;
    Data< helper::vector<Real> > radii;

    Data< Vec3 > direction;
    Data< Vec3 > normal;
    Data< Real > edgeAngle;
    Data< Real > triAngle;

    Data<VecCoord> f_X0;
    Data< helper::vector<BaseMeshTopology::Edge> > f_edges;
    Data< helper::vector<BaseMeshTopology::Triangle> > f_triangles;
    Data<SetIndex> f_pointIndices;
    Data<SetEdge> f_edgeIndices;
    Data<SetTriangle> f_triangleIndices;
    Data<double> _drawSize;
    const VecCoord* x0;
    unsigned int pointSize;

protected:

    bool containsPoint(const Vec3& c, const Real& r, const Coord& p);
    bool containsEdge(const Vec3& c, const Real& r, const BaseMeshTopology::Edge& edge);
    bool containsTriangle(const Vec3& c, const Real& r, const BaseMeshTopology::Triangle& triangle);
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_ENGINE_API PrimitivesInSphereROI<defaulttype::Vec3dTypes>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_ENGINE_API PrimitivesInSphereROI<defaulttype::Vec3fTypes>;
#endif //SOFA_DOUBLE
#endif

} // namespace engine

} // namespace component

} // namespace sofa

#endif //SOFA_COMPONENT_ENGINE_PRIMITIVESINSPHEREROI_H
