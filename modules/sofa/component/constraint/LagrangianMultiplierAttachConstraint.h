#ifndef SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LAGRANGIANMULTIPLIERATTACHCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/PairInteractionForceField.h>
#include <sofa/component/constraint/LagrangianMultiplierConstraint.h>
#include <sofa/core/VisualModel.h>
#include <vector>


namespace sofa
{

namespace component
{

namespace constraint
{

template<class DataTypes>
class LagrangianMultiplierAttachConstraint : public LagrangianMultiplierConstraint<DataTypes>, public core::componentmodel::behavior::PairInteractionForceField<DataTypes>, public core::VisualModel
{
public:
    typedef typename core::componentmodel::behavior::PairInteractionForceField<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef defaulttype::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;
    typedef typename core::componentmodel::behavior::MechanicalState<DataTypes> MechanicalState;

protected:
    struct ConstraintData
    {
        int m1, m2;   ///< the two attached points
    };

    std::vector<ConstraintData> constraints;

public:

    LagrangianMultiplierAttachConstraint(MechanicalState* m1=NULL, MechanicalState* m2=NULL)
        : Inherit(m1, m2)
    {
    }

    void clear(int reserve = 0)
    {
        constraints.clear();
        if (reserve)
            constraints.reserve(reserve);
        this->lambda->resize(0);
    }

    void addConstraint(int m1, int m2);

    virtual void addForce(VecDeriv& f1, VecDeriv& f2, const VecCoord& x1, const VecCoord& x2, const VecDeriv& v1, const VecDeriv& v2);

    virtual void addDForce(VecDeriv& df1, VecDeriv& df2, const VecDeriv& dx1, const VecDeriv& dx2);

    virtual double getPotentialEnergy(const VecCoord&, const VecCoord&);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }

};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
