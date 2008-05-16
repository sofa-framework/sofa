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
#ifndef SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_H
#define SOFA_COMPONENT_CONSTRAINT_LINEARMOVEMENTCONSTRAINT_H

#include <sofa/core/componentmodel/behavior/Constraint.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>
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
using core::objectmodel::Data;
using namespace sofa::core::objectmodel;
using namespace sofa::defaulttype;

/** impose a movement to given DOFs
	The movement between 2 key times is linearly interpolated
*/
template <class DataTypes>
class LinearMovementConstraint : public core::componentmodel::behavior::Constraint<DataTypes>, public virtual core::objectmodel::BaseObject
{
public:
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;
    typedef topology::PointSubset SetIndex;
    typedef helper::vector<unsigned int> SetIndexArray;

public:
    Data<SetIndex> m_indices;
    Data<helper::vector<Real> > m_keyTimes;
    Data<VecDeriv > m_keyMovements;

    Real prevT, nextT;
    Deriv prevM, nextM;
    VecCoord x0;

    LinearMovementConstraint();

    virtual ~LinearMovementConstraint();

    void clearIndices();
    void addIndex(unsigned int index);
    void removeIndex(unsigned int index);
    void clearKeyMovements();
    void addKeyMovement(Real time, Deriv movement);
    //void removeTranslation();

    // -- Constraint interface
    void init();
    void projectResponse(VecDeriv& dx);
    virtual void projectVelocity(VecDeriv& dx); ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition(VecCoord& x); ///< project x to constrained space (x models a position)

    //void applyConstraint(defaulttype::BaseMatrix *mat, unsigned int &offset);
    //void applyConstraint(defaulttype::BaseVector *vect, unsigned int &offset);

    // Handle topological changes
    virtual void handleTopologyChange();

    virtual void draw();

protected:

    // Define TestNewPointFunction
    static bool FCTestNewPointFunction(int, void*, const sofa::helper::vector< unsigned int > &, const sofa::helper::vector< double >& );

    // Define RemovalFunction
    static void FCRemovalFunction ( int , void*);

};

} // namespace constraint

} // namespace component

} // namespace sofa


#endif
