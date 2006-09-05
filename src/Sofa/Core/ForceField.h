#ifndef SOFA_CORE_FORCEFIELD_H
#define SOFA_CORE_FORCEFIELD_H

#include "BasicForceField.h"
#include "MechanicalModel.h"

namespace Sofa
{

namespace Core
{

template<class TDataTypes>
class ForceField : public BasicForceField
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ForceField(MechanicalModel<DataTypes> *mm = NULL);

    virtual ~ForceField();

    virtual void init();

    virtual void addForce();

    virtual void addDForce();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v) = 0;

    virtual void addDForce (VecDeriv& df, const VecCoord& x, const VecDeriv& v, const VecDeriv& dx) = 0;

protected:
    MechanicalModel<DataTypes> *mmodel;
};

} // namespace Core

} // namespace Sofa

#endif
