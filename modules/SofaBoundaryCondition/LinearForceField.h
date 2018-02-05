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
#ifndef SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H
#include "config.h"

#include <sofa/core/behavior/ForceField.h>
#include <SofaBaseTopology/TopologySubsetData.h>

namespace sofa
{

namespace component
{

namespace forcefield
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
    typedef helper::vector<unsigned int> SetIndexArray;
    typedef sofa::component::topology::PointSubsetData< SetIndexArray > SetIndex;

protected:

    LinearForceFieldInternalData<DataTypes> *data;
    friend class LinearForceFieldInternalData<DataTypes>;

public:
    /// concerned DOFs
    SetIndex points;

    /// applied force for all the points
    Data< Real > d_force;

    /// the key frames when the forces are defined by the user
    Data< helper::vector< Real > > d_keyTimes;

    /// forces corresponding to the key frames
    Data< VecDeriv > d_keyForces;

    /// for drawing. The sign changes the direction, 0 doesn't draw arrow
    Data< SReal > d_arrowSizeCoef;
protected:
    LinearForceField();
    virtual ~LinearForceField() override { delete data; }

public:
    void draw(const core::visual::VisualParams* vparams) override;

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

    virtual void init() override;

    // ForceField methods
    /// Add the forces
    virtual void addForce (const core::MechanicalParams* mparams, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& v) override;

    /// Compute the force derivative
    virtual void addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /* d_df */, const DataVecDeriv& /* d_dx */) override
    {
        //TODO: remove this line (avoid warning message) ...
        mparams->setKFactorUsed(true);
    }

    virtual void addKToMatrix(sofa::defaulttype::BaseMatrix * matrix, SReal kFact, unsigned int &offset) override;

    virtual SReal getPotentialEnergy(const core::MechanicalParams* mparams, const DataVecCoord& x) const override;

private :
    /// the key times surrounding the current simulation time (for interpolation)
    Real prevT, nextT;

    /// the forces corresponding to the surrounding key times
    Deriv prevF, nextF;

    /// initial constrained DOFs position
    //VecCoord x0;

protected:
    sofa::core::topology::BaseMeshTopology* topology;

}; // definition of the LinearForceField class


#ifndef SOFA_FLOAT
template <>
SReal LinearForceField<defaulttype::Rigid3dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;
template <>
SReal LinearForceField<defaulttype::Rigid2dTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;
#endif

#ifndef SOFA_DOUBLE
template <>
SReal LinearForceField<defaulttype::Rigid3fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;
template <>
SReal LinearForceField<defaulttype::Rigid2fTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord& ) const;
#endif

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)

#ifndef SOFA_FLOAT
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec3dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec2dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec1dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec6dTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Rigid3dTypes>;
// extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid2dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec3fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec2fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec1fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Vec6fTypes>;
extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<sofa::defaulttype::Rigid3fTypes>;
// extern template class SOFA_BOUNDARY_CONDITION_API LinearForceField<Rigid2fTypes>;
#endif

#endif // defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_CPP)


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_LINEARFORCEFIELD_H
