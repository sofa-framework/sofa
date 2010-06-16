/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H
#define SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_H

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <set>
#include <sofa/component/topology/PointSubset.h>


namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{

/// This class can be overriden if needed for additionnal storage within template specilizations.
template <class DataTypes>
class FixedPlaneConstraintInternalData
{
};

template <class DataTypes>
class FixedPlaneConstraint : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(FixedPlaneConstraint,DataTypes),SOFA_TEMPLATE(sofa::core::behavior::ProjectiveConstraintSet, DataTypes));

    typedef core::behavior::ProjectiveConstraintSet<DataTypes> Inherit;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::SparseVecDeriv SparseVecDeriv;

    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef topology::PointSubset SetIndex;
    typedef typename Coord::value_type   Real    ;

protected:
    FixedPlaneConstraintInternalData<DataTypes> data;
    friend class FixedPlaneConstraintInternalData<DataTypes>;

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
    template <class DataDeriv>
    void projectResponseT(DataDeriv& dx);

    void projectResponse(VecDeriv& dx);
    void projectResponse(SparseVecDeriv& dx);
    virtual void projectVelocity(VecDeriv& /*dx*/) {} ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& /*x*/) {} ///< project x to constrained space (x models a position)

    virtual void init();

    void setDirection (Coord dir);
    void selectVerticesAlongPlane();
    void setDminAndDmax(const Real _dmin,const Real _dmax) {dmin=_dmin; dmax=_dmax; selectVerticesFromPlanes=true;}

    void draw();
protected:

    sofa::core::topology::BaseMeshTopology* topology;

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

#if defined(WIN32) && !defined(SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_FIXEDPLANECONSTRAINT_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API FixedPlaneConstraint<defaulttype::Rigid3dTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API FixedPlaneConstraint<defaulttype::Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API FixedPlaneConstraint<defaulttype::Rigid3fTypes>;
extern template class SOFA_COMPONENT_PROJECTIVECONSTRAINTSET_API FixedPlaneConstraint<defaulttype::Vec3fTypes>;
#endif
#endif

} // namespace projectiveconstraintset

} // namespace component

} // namespace sofa

#endif
