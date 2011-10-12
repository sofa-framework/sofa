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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_CORE_BEHAVIOR_BASECONSTRAINT_H
#define SOFA_CORE_BEHAVIOR_BASECONSTRAINT_H

#include <sofa/core/behavior/BaseConstraintSet.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/MultiMatrixAccessor.h>

#include <sofa/defaulttype/BaseMatrix.h>
#include <sofa/defaulttype/BaseVector.h>

#include <vector>

namespace sofa
{

namespace core
{

namespace behavior
{
#ifdef SOFA_DEV

/**
 *  \brief Object computing a constraint resolution within a Gauss-Seidel algorithm
 */

class ConstraintResolution
{
public:
    ConstraintResolution()
        : nbLines(1), tolerance(0.0) {}

    virtual ~ConstraintResolution() {}

    /// The resolution object can do precomputation with the compliance matrix, and give an initial guess.
    virtual void init(int /*line*/, double** /*w*/, double* /*force*/) {}

    /// The resolution object can provide an initial guess
    virtual void initForce(int /*line*/, double* /*force*/) {}

    /// Resolution of the constraint for one Gauss-Seidel iteration
    virtual void resolution(int line, double** w, double* d, double* force, double * /*dFree*/)
    {
        resolution(line, w, d, force);
    };

    /// Resolution of the constraint for one Gauss-Seidel iteration
    /// @deprecated
    virtual void resolution(int /*line*/, double** /*w*/, double* /*d*/, double* /*force*/) {};

    /// Called after Gauss-Seidel last iteration, in order to store last computed forces for the inital guess
    virtual void store(int /*line*/, double* /*force*/, bool /*convergence*/) {}

    /// Number of dof used by this particular constraint. To be modified in the object's constructor.
    unsigned int nbLines;

    /// Custom tolerance, used for the convergence of this particular constraint instead of the global tolerance
    double tolerance;
};
#endif // SOFA_DEV

/**
 *  \brief Component computing constraints within a simulated body.
 *
 *  This class define the abstract API common to all constraints.
 *  A BaseConstraint computes constraints applied to one or more simulated body
 *  given its current position and velocity.
 *
 *  Constraints can be internal to a given body (attached to one MechanicalState,
 *  see the Constraint class), or link several bodies together (such as contacts,
 *  see the InteractionConstraint class).
 *
 */
class SOFA_CORE_API BaseConstraint : public BaseConstraintSet
{
public:
    SOFA_ABSTRACT_CLASS(BaseConstraint, BaseConstraintSet);

protected:
    virtual ~BaseConstraint() { }
public:
    /// Get the ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.
    int getGroup() const { return group.getValue(); }

    /// Set the ID of the group containing this constraint. This ID is used to specify which constraints are solved by which solver, by specifying in each solver which groups of constraints it should handle.
    void setGroup(int g) { group.setValue(g); }

    typedef long long PersistentID;
    typedef helper::vector<PersistentID> VecPersistentID;
    typedef defaulttype::Vec<3,int> ConstCoord;
    typedef helper::vector<ConstCoord> VecConstCoord;
    typedef defaulttype::Vec<3,double> ConstDeriv;
    typedef helper::vector<ConstDeriv> VecConstDeriv;
    typedef double ConstArea;
    typedef helper::vector<ConstArea> VecConstArea;

    class ConstraintBlockInfo
    {
    public:
        BaseConstraint* parent;
        int const0; ///< index of first constraint
        int nbLines; ///< how many dofs (i.e. lines in the matrix) are used by each constraint
        int nbGroups; ///< how many groups of constraints are active
        bool hasId; ///< true if this constraint has persistent ID information
        bool hasPosition; ///< true if this constraint has coordinates information
        bool hasDirection; ///< true if this constraint has direction information
        bool hasArea; ///< true if this constraint has area information
        int offsetId; ///< index of first constraint group info in vector of persistent ids and coordinates
        int offsetPosition; ///< index of first constraint group info in vector of coordinates
        int offsetDirection; ///< index of first constraint info in vector of directions
        int offsetArea; ///< index of first constraint group info in vector of areas
        ConstraintBlockInfo() : parent(NULL), const0(0), nbLines(1), nbGroups(0), hasId(false), hasPosition(false), hasDirection(false), hasArea(false), offsetId(0), offsetPosition(0), offsetDirection(0), offsetArea(0)
        {}
    };
    typedef helper::vector<ConstraintBlockInfo> VecConstraintBlockInfo;

    /// Get information for each constraint: pointer to parent BaseConstraint, unique persistent ID, 3D position
    virtual void getConstraintInfo(VecConstraintBlockInfo& /*blocks*/, VecPersistentID& /*ids*/, VecConstCoord& /*positions*/, VecConstDeriv& /*directions*/, VecConstArea& /*areas*/) {}

#ifdef SOFA_DEV
    /// Add the corresponding ConstraintResolution using the offset parameter
    virtual void getConstraintResolution(std::vector<ConstraintResolution*>& /*resTab*/, unsigned int& /*offset*/) {};
#endif //SOFA_DEV
};

} // namespace behavior

} // namespace core

} // namespace sofa

#endif
