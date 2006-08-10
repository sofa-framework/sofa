#ifndef SOFA_COMPONENTS_LagrangianMultiplierFixedConstraint_H
#define SOFA_COMPONENTS_LagrangianMultiplierFixedConstraint_H

#include "Sofa/Core/InteractionForceField.h"
#include "LagrangianMultiplierConstraint.h"
#include "Sofa/Abstract/VisualModel.h"

#include <vector>

namespace Sofa
{

namespace Components
{

template<class DataTypes>
class LagrangianMultiplierFixedConstraint : public LagrangianMultiplierConstraint<DataTypes>, public Core::ForceField<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef Common::StdVectorTypes<Real, Real, Real> LMTypes;
    typedef typename LMTypes::VecCoord LMVecCoord;
    typedef typename LMTypes::VecDeriv LMVecDeriv;

protected:

    struct PointConstraint
    {
        int indice;   ///< index of the constrained point
        Coord pos;    ///< constrained position of the point
    };

    std::vector<PointConstraint> constraints;

public:

    LagrangianMultiplierFixedConstraint(Core::MechanicalModel<DataTypes>* object)
        : Core::ForceField<DataTypes>(object)
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

    virtual void addDForce (VecDeriv& df, const VecCoord& x, const VecDeriv& v, const VecDeriv& dx);

    // -- VisualModel interface
    void draw();
    void initTextures() { }
    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
