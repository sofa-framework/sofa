#ifndef SOFA_COMPONENTS_FIXEDCONSTRAINT_H
#define SOFA_COMPONENTS_FIXEDCONSTRAINT_H

#include "Sofa/Core/Constraint.h"
#include "Sofa/Core/MechanicalModel.h"
#include "Sofa/Abstract/VisualModel.h"

#include "Sofa/Components/Common/SofaBaseMatrix.h"
#include "Sofa/Components/Common/SofaBaseVector.h"
#include "Sofa/Components/Common/vector.h"

#include <set>

namespace Sofa
{

namespace Components
{

using Common::vector;
using Common::DataField;

template <class DataTypes>
class FixedConstraint : public Core::Constraint<DataTypes>, public Abstract::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef vector<int> SetIndex;

protected:
    //SetIndex indices;

public:
    FixedConstraint();
    virtual const char* getTypeName() const { return "FixedConstraint"; }
    DataField<SetIndex> f_indices;


    FixedConstraint(Core::MechanicalModel<DataTypes>* mmodel);

    virtual ~FixedConstraint();

    FixedConstraint<DataTypes>* addConstraint(int index);
    FixedConstraint<DataTypes>* removeConstraint(int index);

    // -- Constraint interface
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    void applyConstraint(Components::Common::SofaBaseMatrix *mat, unsigned int &offset);
    void applyConstraint(Components::Common::SofaBaseVector *vect, unsigned int &offset);

    // -- VisualModel interface

    virtual void draw();

    void initTextures() { }

    void update() { }
};

} // namespace Components

} // namespace Sofa

#endif
