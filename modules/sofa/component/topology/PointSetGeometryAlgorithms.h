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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_H

#include <sofa/core/componentmodel/topology/BaseTopology.h>
#include <sofa/core/componentmodel/topology/BaseMeshTopology.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/component.h>
#include <sofa/helper/system/glut.h>

namespace sofa
{

namespace component
{

namespace topology
{
using core::componentmodel::topology::BaseMeshTopology;
using core::componentmodel::behavior::MechanicalState;
typedef BaseMeshTopology::PointID PointID;

/**
* A class that can perform some geometric computation on a set of points.
*/
template<class DataTypes>
class PointSetGeometryAlgorithms : public core::componentmodel::topology::GeometryAlgorithms
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointSetGeometryAlgorithms,DataTypes),core::componentmodel::topology::GeometryAlgorithms);

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::CPos CPos;
    enum { NC = CPos::static_size };

    enum Angle {ACUTE, RIGHT, OBTUSE};

    Angle computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const;

    PointSetGeometryAlgorithms()
        : GeometryAlgorithms()
        ,debugViewIndicesScale (core::objectmodel::Base::initData(&debugViewIndicesScale, (float) 0.0001, "debugViewIndicesScale", "Debug : scale for view topology indices"))
        ,debugViewPointIndices (core::objectmodel::Base::initData(&debugViewPointIndices, (bool) false, "debugViewPointIndices", "Debug : view Point indices"))
    {
    }

    virtual ~PointSetGeometryAlgorithms() {}

    virtual void init();

    virtual void reinit();

    void draw();

    void computeIndicesScale();

    /** return the centroid of the set of points */
    Coord getPointSetCenter() const;

    /** return the centre and a radius of a sphere enclosing the  set of points (may not be the smalled one) */
    void getEnclosingSphere(Coord &center, Real &radius) const;

    /** return the axis aligned bounding box : index 0 = xmin, index 1=ymin,
    index 2 = zmin, index 3 = xmax, index 4 = ymax, index 5=zmax */
    void getAABB(Real bb[6]) const;

    /** \brief Returns the axis aligned bounding box */
    void getAABB(CPos& minCoord, CPos& maxCoord) const;

    const Coord& getPointPosition(const PointID pointId) const;

    const Coord& getPointRestPosition(const PointID pointId) const;

    /** \brief Returns the object where the mechanical DOFs are stored */
    sofa::core::componentmodel::behavior::MechanicalState<DataTypes> *getDOF() const { return object;	}

    float PointIndicesScale;

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (dynamic_cast<sofa::core::componentmodel::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const PointSetGeometryAlgorithms<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

protected:
    /** the object where the mechanical DOFs are stored */
    sofa::core::componentmodel::behavior::MechanicalState<DataTypes> *object;
    sofa::core::componentmodel::topology::BaseMeshTopology* m_topology;
    Data<float> debugViewIndicesScale;
    Data<bool> debugViewPointIndices;
};

#if defined(WIN32) && !defined(SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_CPP)
#pragma warning(disable : 4231)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
extern template class SOFA_COMPONENT_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETGEOMETRYALGORITHMS_H
