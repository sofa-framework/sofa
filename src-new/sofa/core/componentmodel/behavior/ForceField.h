#ifndef SOFA_CORE_FORCEFIELD_H
#define SOFA_CORE_FORCEFIELD_H

#include "BasicForceField.h"
#include "MechanicalModel.h"
#include "Sofa-old/Components/Common/SofaBaseMatrix.h"
#include "Sofa-old/Components/Common/SofaBaseVector.h"

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

    virtual double getPotentialEnergy();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v) = 0;

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx) = 0;

    virtual double getPotentialEnergy(const VecCoord& x) =0;

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeMatrix(Sofa::Components::Common::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void computeVector(Sofa::Components::Common::SofaBaseVector *, unsigned int & ) {};

    virtual void matResUpdatePosition(Sofa::Components::Common::SofaBaseVector *, unsigned int & ) {};

protected:
    MechanicalModel<DataTypes> *mmodel;
};

} // namespace Core

} // namespace Sofa

#endif
