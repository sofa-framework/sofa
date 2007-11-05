// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_H
#define SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_H

#include <sofa/component/forcefield/SpringForceField.h>
#include <sofa/defaulttype/Mat.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/** SpringForceField able to evaluate and apply its stiffness.
This allows to perform implicit integration.
Stiffness is evaluated and stored by the addForce method.
When explicit integration is used, SpringForceField is slightly more efficient.
*/
template<class DataTypes>
class StiffSpringForceField : public SpringForceField<DataTypes>
{
public:
    typedef SpringForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef typename Inherit::Spring Spring;
    typedef core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;
    enum { N=Coord::static_size };
    typedef defaulttype::Mat<N,N,Real> Mat;

protected:
    sofa::helper::vector<Mat>  dfdx;
    double m_potentialEnergy;

    /// Accumulate the spring force and compute and store its stiffness
    void addSpringForce(double& potentialEnergy, VecDeriv& f1, const VecCoord& p1, const VecDeriv& v1, VecDeriv& f2, const VecCoord& p2, const VecDeriv& v2, int i, const Spring& spring);

    /// Apply the stiffness, i.e. accumulate df given dx
    void addSpringDForce(VecDeriv& df1, const VecDeriv& dx1, VecDeriv& df2, const VecDeriv& dx2, int i, const Spring& spring);

public:
    StiffSpringForceField(MechanicalState* object1, MechanicalState* object2, double ks=100.0, double kd=5.0)
        : SpringForceField<DataTypes>(object1, object2, ks, kd)
    {
    }

    StiffSpringForceField(double ks=100.0, double kd=5.0)
        : SpringForceField<DataTypes>(ks, kd)
    {
    }

    virtual void init();

    /// Accumulate f corresponding to x,v
    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    /// Accumulate df corresponding to dx
    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);
};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
