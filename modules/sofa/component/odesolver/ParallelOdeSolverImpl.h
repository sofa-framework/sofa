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
#ifndef SOFA_SMP_PARALLELODESOLVERIMPL_H
#define SOFA_SMP_PARALLELODESOLVERIMPL_H

#include <sofa/core/componentmodel/behavior/OdeSolver.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>
#include <sofa/simulation/common/ParallelSolverImpl.h>
#ifdef SOFA_SMP
#include <sofa/core/componentmodel/behavior/ParallelMultivector.h>
using namespace sofa::defaulttype::SharedTypes;
#endif
namespace sofa
{

namespace component
{
namespace odesolver
{


class ParallelOdeSolverImpl : virtual public component::odesolver::OdeSolverImpl, virtual public sofa::simulation::common::ParallelSolverImpl
{
public:
    typedef sofa::core::componentmodel::behavior::ParallelMultiVector<ParallelOdeSolverImpl> MultiVector;
    typedef simulation::common::ParallelSolverImpl::VecId VecId;
    /*      typedef sofa::core::componentmodel::behavior::MultiVector<OdeSolverImpl> MultiVector;
        typedef sofa::core::componentmodel::behavior::MultiMatrix<OdeSolverImpl> MultiMatrix;
        typedef sofa::core::componentmodel::behavior::MechanicalMatrix MechanicalMatrix;

        /// Propagate the given state (time, position and velocity) through all mappings
        virtual void propagatePositionAndVelocity(double t, VecId x, VecId v);
        /// Compute the acceleration corresponding to the given state (time, position and velocity)
        virtual void computeAcc(double t, VecId a, VecId x, VecId v);
        virtual void computeContactAcc(double t, VecId a, VecId x, VecId v);

        /// @name Matrix operations using LinearSolver components
        /// @{

        virtual void m_resetSystem();
        virtual void m_setSystemMBKMatrix(double mFact, double bFact, double kFact);
        virtual void m_setSystemRHVector(VecId v);
        virtual void m_setSystemLHVector(VecId v);
        virtual void m_solveSystem();
        virtual void m_print( std::ostream& out );

        /// @} */
};

}
} // namespace simulation

} // namespace sofa

#endif
