#ifndef SOFA_COMPONENTS_FIXEDCONSTRAINT_H
#define SOFA_COMPONENTS_FIXEDCONSTRAINT_H

#include "Sofa/Core/Constraint.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

#include <set>

namespace Sofa
{

namespace Components
{

template <class DataTypes>
class FixedConstraint : public Core::Constraint, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
protected:
    Core::MechanicalModel<DataTypes>* mmodel;

    std::set<int> indices;
public:
    FixedConstraint();

    FixedConstraint(Core::MechanicalModel<DataTypes>* mmodel);

    ~FixedConstraint();

    void setMechanicalModel(Core::MechanicalModel<DataTypes>* mm);

    Core::Constraint* addConstraint(int index);
    Core::Constraint*  removeConstraint(int index);

    // -- Constraint interface
    void applyConstraint();

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
