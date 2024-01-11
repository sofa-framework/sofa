/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once
#include <sofa/component/topology/container/dynamic/config.h>

#include <sofa/core/topology/BaseTopology.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/core/State.h>

#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa::component::topology::container::dynamic
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


    ~PointSetGeometryAlgorithms() override {}
public:
    void init() override;

    void reinit() override;

    void draw(const core::visual::VisualParams* vparams) override;

    void computeBBox(const core::ExecParams* params, bool onlyVisible=false) override;

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

    /** \brief Returns the object where the DOFs are stored */
    sofa::core::State<DataTypes> *getDOF() const { return object;	}

    //float PointIndicesScale;
    float getIndicesScale() const;

    template<class T>
    static bool canCreate(T*& obj, core::objectmodel::BaseContext* context, core::objectmodel::BaseObjectDescription* arg)
    {
        if (context->getState() && dynamic_cast<sofa::core::State<DataTypes>*>(context->getState()) == nullptr)
        {
            arg->logError(std::string("No state with the datatype '") + DataTypes::Name() +
                          "' found in the context node.");
            return false;
        }
        return BaseObject::canCreate(obj, context, arg);
    }

    /** \brief Called by the state change callback to initialize added
     * points according to the topology (topology element & local coordinates) 
     */
    void initPointsAdded(const type::vector< sofa::Index > &indices, const type::vector< core::topology::PointAncestorElem > &ancestorElems
        , const type::vector< core::VecCoordId >& coordVecs, const type::vector< core::VecDerivId >& derivVecs ) override;

    /** \brief Process the added point initialization according to the topology and local coordinates.
    */
    virtual void initPointAdded(PointID indice, const core::topology::PointAncestorElem &ancestorElem, const type::vector< VecCoord* >& coordVecs, const type::vector< VecDeriv* >& derivVecs);

protected:
    /** the object where the mechanical DOFs are stored */
    sofa::core::State<DataTypes> *object;
    sofa::core::topology::BaseMeshTopology* m_topology;
    Data<float> d_showIndicesScale; ///< Debug : scale for view topology indices
    Data<bool> d_showPointIndices; ///< Debug : view Point indices
    /// Tage of the Mechanical State associated with the vertex position
    Data<std::string> d_tagMechanics;

    /// Link to be set to the topology container in the component graph.
    SingleLink<PointSetGeometryAlgorithms<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

    /// Return true if the visibility parameters are showing the object in any way whatsoever, false otherwise
    virtual bool mustComputeBBox() const;

};

#if !defined(SOFA_COMPONENT_TOPOLOGY_POINTSETGEOMETRYALGORITHMS_CPP)
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<defaulttype::Rigid3Types>;
extern template class SOFA_COMPONENT_TOPOLOGY_CONTAINER_DYNAMIC_API PointSetGeometryAlgorithms<defaulttype::Rigid2Types>;


#endif

} //namespace sofa::component::topology::container::dynamic
