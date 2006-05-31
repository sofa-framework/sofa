#ifndef SOFA_CORE_CONSTRAINT_H
#define SOFA_CORE_CONSTRAINT_H

#include "BasicConstraint.h"
#include "MechanicalModel.h"

namespace Sofa
{

namespace Core
{

template<class DataTypes>
class Constraint : public BasicConstraint
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    Constraint(MechanicalModel<DataTypes> *mm = NULL);

    virtual ~Constraint();

    virtual void init();

    virtual void applyConstraint(); ///< project dx to constrained space

    virtual void applyConstraint(VecDeriv& dx) = 0; ///< project dx to constrained space

protected:
    MechanicalModel<DataTypes> *mmodel;
};

} // namespace Core

} // namespace Sofa

#endif
