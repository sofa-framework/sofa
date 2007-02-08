#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_H

#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/VisualModel.h>
#include <set>


namespace sofa
{

namespace component
{

namespace constraint
{

template <class DataTypes>
class FixedPlaneConstraint : public core::componentmodel::behavior::Constraint<DataTypes>, public core::VisualModel
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename Coord::value_type   Real    ;

protected:
    std::set<int> indices; // the set of vertex indices
    /// direction on which the constraint applies
    Coord direction;
    /// whether vertices should be selected from 2 parallel planes
    bool selectVerticesFromPlanes;

    Real dmin; // coordinates min of the plane for the vertex selection
    Real dmax;// coordinates max of the plane for the vertex selection
public:
    FixedPlaneConstraint();

    FixedPlaneConstraint(core::componentmodel::behavior::MechanicalState<DataTypes>* mstate);

    ~FixedPlaneConstraint();

    FixedPlaneConstraint<DataTypes>* addConstraint(int index);
    FixedPlaneConstraint<DataTypes>* removeConstraint(int index);

    // -- Constraint interface
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    virtual void init();

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax) {dmin=_dmin; dmax=_dmax; selectVerticesFromPlanes=true;}

    // -- VisualModel interface

    void draw();

    void initTextures() { }

    void update() { }
protected:
    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,direction);
        if ((d>dmin)&& (d<dmax))
            return true;
        else
            return false;
    }
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
