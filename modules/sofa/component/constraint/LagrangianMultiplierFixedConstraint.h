#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERFIXEDCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/ForceField.h>
#include <sofa/component/constraint/LagrangianMultiplierConstraint.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class LagrangianMultiplierFixedConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::componentmodel::behavior::ForceField<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;

protected:

    struct PointConstraint
    {
        int indice;   ///< index of the constrained point
        Coord pos;    ///< constrained position of the point
    };

    sofa::helper::vector<PointConstraint> constraints;

public:

    LagrangianMultiplierFixedConstraint()
    {
    }

    void clear(int reserve = 0)
    {
        constraints.clear();
        if (reserve)
            constraints.reserve(reserve);
        this->lambda->resize(0);
    }

    void addConstraint(int indice, const Coord& pos);

    virtual void init();

    virtual void addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

    virtual void addDForce (VecDeriv& df, const VecDeriv& dx);

    virtual double getPotentialEnergy(const VecCoord& x);

    void draw();

    /// this constraint is holonomic
    bool isHolonomic() {return true;}
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
