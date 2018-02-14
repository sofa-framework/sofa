/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaPreconditioner/WarpPreconditioner.inl>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/accessor.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>

#include <iostream>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{


SOFA_DECL_CLASS(WarpPreconditioner)

int WarpPreconditionerClass = core::RegisterObject("Linear system solver wrapping another (precomputed) linear solver by a per-node rotation matrix")
#ifndef SOFA_FLOAT
.add< WarpPreconditioner< RotationMatrix<double>, FullVector<double>, NoThreadManager > >()
#endif
#ifndef SOFA_DOUBLE
.add< WarpPreconditioner< RotationMatrix<float>, FullVector<float>,NoThreadManager > > ()
#endif
;
;

} // namespace linearsolver

} // namespace component

} // namespace sofa

