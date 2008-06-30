#ifndef SOFA_COMPONENT_CONSTANTFORCEFIELD_H
#define SOFA_COMPONENT_CONSTANTFORCEFIELD_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/vector.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

/** Apply constant forces to given degrees of freedom.  */
template<class DataTypes>
class ConstantForceField : public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef core::componentmodel::behavior::ForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type Real;
    typedef helper::vector<unsigned> VecIndex;
public:

    Data< VecIndex > points;
    Data< VecDeriv > forces;

    ConstantForceField();

    /// Add the forces
    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    /// Constant force has null variation
    virtual void addDForce (VecDeriv& , const VecDeriv& ) {}


    virtual double getPotentialEnergy(const VecCoord& x);


    void draw();
    bool addBBox(double* minBBox, double* maxBBox);

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
