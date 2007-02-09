#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_FIXEDCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/SofaBaseMatrix.h>
#include <sofa/defaulttype/SofaBaseVector.h>
#include <sofa/helper/vector.h>
#include <sofa/component/topology/PointSubset.h>
#include <set>

namespace sofa
{

namespace component
{

namespace constraint
{

using helper::vector;
using core::objectmodel::DataField;
using namespace sofa::core::objectmodel;

/// This class can be overridden if needed for additionnal storage within template specializations.
template <class DataTypes>
class FixedConstraintInternalData
{
};

template <class DataTypes>
class FixedConstraint : public core::componentmodel::behavior::Constraint<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;
    //typedef std::set<int> SetIndex;

protected:
    //SetIndex indices;

    FixedConstraintInternalData<DataTypes> data;

public:
    FixedConstraint();
    //virtual const char* getTypeName() const { return "FixedConstraint"; }
    DataField<SetIndex> f_indices;


    FixedConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate);

    virtual ~FixedConstraint();

    FixedConstraint<DataTypes>* addConstraint(unsigned int index);
    FixedConstraint<DataTypes>* removeConstraint(unsigned int index);

    // -- Constraint interface
    void init();
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    void applyConstraint(defaulttype::SofaBaseMatrix *mat, unsigned int &offset);
    void applyConstraint(defaulttype::SofaBaseVector *vect, unsigned int &offset);

    // handle topological changes
    virtual void handleEvent( Event* );

    // -- VisualModel interface

    virtual void draw();

    void initTextures() { }

    void update() { }
};

} // namespace constraint

} // namespace component

} // namespace sofa


#endif
