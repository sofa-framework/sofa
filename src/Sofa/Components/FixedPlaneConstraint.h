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

    FixedPlaneConstraint(Core::MechanicalModel<DataTypes>* mmodel);

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

} // namespace Components

} // namespace Sofa

#endif
