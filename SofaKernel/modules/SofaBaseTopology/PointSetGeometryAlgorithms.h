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
#ifndef SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_H
#define SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_H
#include "config.h"

#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
namespace sofa
{

namespace component
{

namespace topology
{

/**
 * A class that can perform some geometric computation on a set of points.
 */
template<class DataTypes>
class PointSetGeometryAlgorithms : public core::topology::GeometryAlgorithms
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(PointSetGeometryAlgorithms,DataTypes),core::topology::GeometryAlgorithms);

    typedef core::topology::BaseMeshTopology::PointID PointID;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::CPos CPos;
    enum { NC = CPos::static_size };

    enum Angle {ACUTE, RIGHT, OBTUSE};

    Angle computeAngle(PointID ind_p0, PointID ind_p1, PointID ind_p2) const;

protected:
    PointSetGeometryAlgorithms();


    virtual ~PointSetGeometryAlgorithms() override {}
public:
    virtual void init() override;

    virtual void reinit() override;

    void draw(const core::visual::VisualParams* vparams) override;

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
    sofa::core::behavior::MechanicalState<DataTypes> *getDOF() const { return object;	}

    //float PointIndicesScale;
    float getIndicesScale() const;

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getMechanicalState() && dynamic_cast<sofa::core::behavior::MechanicalState<DataTypes>*>(context->getMechanicalState()) == NULL)
            return false;
        return BaseObject::canCreate(obj, context, arg);
    }

    virtual std::string getTemplateName() const override
    {
        return templateName(this);
    }

    static std::string templateName(const PointSetGeometryAlgorithms<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    /** \brief Called by the MechanicalObject state change callback to initialize added
     * points according to the topology (topology element & local coordinates) 
     */
    void initPointsAdded(const helper::vector< unsigned int > &indices, const helper::vector< core::topology::PointAncestorElem > &ancestorElems
        , const helper::vector< core::VecCoordId >& coordVecs, const helper::vector< core::VecDerivId >& derivVecs ) override;

    /** \brief Process the added point initialization according to the topology and local coordinates.
    */
    virtual void initPointAdded(unsigned int indice, const core::topology::PointAncestorElem &ancestorElem, const helper::vector< VecCoord* >& coordVecs, const helper::vector< VecDeriv* >& derivVecs);

protected:
    /** the object where the mechanical DOFs are stored */
    sofa::core::behavior::MechanicalState<DataTypes> *object;
    sofa::core::topology::BaseMeshTopology* m_topology;
    Data<float> d_showIndicesScale;
    Data<bool> d_showPointIndices;
    /// Tage of the Mechanical State associated with the vertex position
    Data<std::string> d_tagMechanics;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec2dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec1dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid3dTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid2dTypes>;
#endif

#ifndef SOFA_DOUBLE
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec2fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Vec1fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid3fTypes>;
extern template class SOFA_BASE_TOPOLOGY_API PointSetGeometryAlgorithms<defaulttype::Rigid2fTypes>;
#endif
#endif

} // namespace topology

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENTS_POINTSETGEOMETRYALGORITHMS_H
