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
#include <sofa/component/mechanicalload/config.h>

#include <sofa/core/behavior/ForceField.h>
#include <sofa/core/topology/TopologySubsetIndices.h>

namespace sofa::component::mechanicalload
{

template<class DataTypes>
class LinearForceFieldInternalData
{
public:
};

/** Apply forces changing to given degres of freedom. Some keyTimes are given
* and the force to be applied is linearly interpolated between keyTimes. */
template<class DataTypes>
class LinearForceField : public core::behavior::ForceField<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(LinearForceField, DataTypes), SOFA_TEMPLATE(core::behavior::ForceField, DataTypes));

    typedef core::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef core::objectmodel::Data<VecCoord> DataVecCoord;
    typedef core::objectmodel::Data<VecDeriv> DataVecDeriv;
    typedef type::vector<unsigned int> SetIndexArray;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;

protected:

    LinearForceFieldInternalData<DataTypes> *data;
    friend class LinearForceFieldInternalData<DataTypes>;

public:
    /// concerned DOFs
    SetIndex points;

    /// applied force for all the points
    Data< Real > d_force;

    /// the key frames when the forces are defined by the user
    Data< type::vector< Real > > d_keyTimes;

    /// forces corresponding to the key frames
    Data< VecDeriv > d_keyForces;

    /// for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< SReal > d_arrowSizeCoef;

    /// Link to be set to the topology container in the component graph.
    SingleLink<LinearForceField<DataTypes>, sofa::core::topology::BaseMeshTopology, BaseLink::FLAG_STOREPATH | BaseLink::FLAG_STRONGLINK> l_topology;

protected:
    LinearForceField();
    ~LinearForceField() override { delete data; }

public:
    /// methods to add/remove some indices, keyTimes, keyForces
    void addPoint(unsigned index);
    void removePoint(unsigned int index);
    void clearPoints();

    /**
    * Add a new key force.
    * Key force should be added in classified order.
    *
    * @param time  the simulation time you want to set a movement (in sec)
    * @param force the corresponding force
    */
    void addKeyForce(Real time, Deriv force);
    void clearKeyForces();

    void init() override;

    // ForceField methods
    /// Add the forces
    void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    /// Compute the force derivative
    void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override;

    void addKToMatrix(sofa::linearalgebra::BaseMatrix * matrix, SReal kFact, unsigned int &offset) override;

    void buildStiffnessMatrix(core::behavior::StiffnessMatrix* matrix) override;

    void buildDampingMatrix(core::behavior::DampingMatrix* /*matrix*/) final;

    SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

private :
    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;

    /// the forces corresponding to the surrounding key times
    Deriv prevF, nextF;

    /// initial constrained DOFs position
    //VecCoord x0;

}; // definition of the LinearForceField class


template <>
void SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<defaulttype::Rigid3Types>::init();

template <>
SReal SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<defaulttype::Rigid3Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;

template <>
SReal SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<defaulttype::Rigid2Types>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;



#if !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)
extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<sofa::defaulttype::Vec3Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<sofa::defaulttype::Vec6Types>;
extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<sofa::defaulttype::Rigid3Types>;
// extern template class SOFA_COMPONENT_MECHANICALLOAD_API LinearForceField<Rigid2Types>;


#endif //  !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)

} // namespace sofa::component::mechanicalload
