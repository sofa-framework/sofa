/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_FIXEDPLANECONSTRAINT_H

#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <set>
#include <sofa/component/topology/PointSubset.h>


namespace sofa
{

namespace component
{

namespace constraint
{

template <class DataTypes>
class FixedPlaneConstraint : public core::componentmodel::behavior::Constraint<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef core::componentmodel::behavior::Constraint<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef topology::PointSubset SetIndex;
    typedef typename Coord::value_type   Real    ;

protected:
    Data<SetIndex> indices; // the set of vertex indices
    /// direction on which the constraint applies
    Data<Coord> direction;
    /// whether vertices should be selected from 2 parallel planes
    bool selectVerticesFromPlanes;

    Data<Real> dmin; // coordinates min of the plane for the vertex selection
    Data<Real> dmax;// coordinates max of the plane for the vertex selection
public:
    FixedPlaneConstraint();

    ~FixedPlaneConstraint();

    FixedPlaneConstraint<DataTypes>* addConstraint(int index);

    FixedPlaneConstraint<DataTypes>* removeConstraint(int index);
    // Handle topological changes
    virtual void handleTopologyChange();
    // -- Constraint interface
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    virtual void init();

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax) {dmin=_dmin; dmax=_dmax; selectVerticesFromPlanes=true;}

    void draw();
protected:

    // Define TestNewPointFunction
    static bool FPCTestNewPointFunction(int, void*, const helper::vector< unsigned int > &, const helper::vector< double >& );

    // Define RemovalFunction
    static void FPCRemovalFunction ( int , void*);

    bool isPointInPlane(Coord p)
    {
        Real d=dot(p,direction.getValue());
        if ((d>dmin.getValue())&& (d<dmax.getValue()))
            return true;
        else
            return false;
    }
};

} // namespace constraint

} // namespace component

} // namespace sofa

#endif
