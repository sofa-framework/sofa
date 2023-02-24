/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/core/config.h>

namespace sofa::core::behavior
{

/**
 *  \brief Object computing a constraint resolution within a Gauss-Seidel algorithm
 */
class SOFA_CORE_API ConstraintResolution
{
public:
    ConstraintResolution(unsigned int nbLines, SReal tolerance = 0.0);

    virtual ~ConstraintResolution();

    /// The resolution object can do precomputation with the compliance matrix, and give an initial guess.
    virtual void init(int /*line*/, SReal** /*w*/, SReal* /*force*/);

    /// The resolution object can provide an initial guess
    virtual void initForce(int /*line*/, SReal* /*force*/);

    /// Resolution of the constraint for one Gauss-Seidel iteration
    virtual void resolution(int line, SReal** w, SReal* d, SReal* force, SReal* dFree);

    /// Called after Gauss-Seidel last iteration, in order to store last computed forces for the inital guess
    virtual void store(int /*line*/, SReal* /*force*/, bool /*convergence*/);

    void setNbLines(unsigned int nbLines)
    {
        m_nbLines = nbLines;
    }

    unsigned int getNbLines() const
    {
        return m_nbLines;
    }

    void setTolerance(SReal tolerance)
    {
        m_tolerance = tolerance;
    }

    SReal getTolerance() const
    {
        return m_tolerance;
    }

private:
    /// Number of dof used by this particular constraint. To be modified in the object's constructor.
    unsigned int m_nbLines;

    /// Custom tolerance, used for the convergence of this particular constraint instead of the global tolerance
    SReal m_tolerance;
};

} // namespace sofa

