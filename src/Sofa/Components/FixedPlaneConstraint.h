#ifndef SOFA_COMPONENTS_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENTS_FIXEDPLANECONSTRAINT_H

#include "Sofa/Core/Constraint.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

#include <set>

namespace Sofa
{

namespace Components
{

template <class DataTypes>
class FixedPlaneConstraint : public Core::Constraint<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;

protected:
    std::set<int> indices;
    /// direction on which the constraint applies
    Coord direction;
    /// whether to nail a point or to allow sliding along a plane of a known normal
    bool alongDirection;

public:
    FixedPlaneConstraint();

    FixedPlaneConstraint(Core::MechanicalModel<DataTypes>* mmodel);

    ~FixedPlaneConstraint();

    FixedPlaneConstraint<DataTypes>* addConstraint(int index);
    FixedPlaneConstraint<DataTypes>* removeConstraint(int index);

    // -- Constraint interface
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)


    void setDirection (Coord dir);
    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
