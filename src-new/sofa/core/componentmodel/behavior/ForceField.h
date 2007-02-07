#ifndef SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_H
#define SOFA_CORE_COMPONENTMODEL_BEHAVIOR_FORCEFIELD_H

#include <sofa/core/componentmodel/behavior/BaseForceField.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>

namespace sofa
{

namespace core
{

namespace componentmodel
{

namespace behavior
{

template<class TDataTypes>
class ForceField : public BaseForceField
{
public:
    typedef TDataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

    ForceField(MechanicalState<DataTypes> *mm = NULL);

    virtual ~ForceField();

    virtual void init();

    virtual void addForce();

    virtual void addDForce();

    virtual double getPotentialEnergy();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v) = 0;

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx) = 0;

    virtual double getPotentialEnergy(const VecCoord& x) =0;

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeMatrix(sofa::defaulttype::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void computeVector(sofa::defaulttype::SofaBaseVector *, unsigned int & ) {};

    virtual void matResUpdatePosition(sofa::defaulttype::SofaBaseVector *, unsigned int & ) {};

protected:
    MechanicalState<DataTypes> *mstate;
};

} // namespace behavior

} // namespace componentmodel

} // namespace core

} // namespace sofa

#endif
