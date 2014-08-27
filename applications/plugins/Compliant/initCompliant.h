/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITCompliant_H
#define INITCompliant_H


#include <sofa/helper/system/config.h>
#include <sofa/simulation/common/Node.h>

#ifdef SOFA_BUILD_Compliant
#define SOFA_Compliant_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_Compliant_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/** \mainpage
  Simulation of deformable object using a formulation similar to the KKT system for hard constraints, regularized using a compliance matrix.

  Provides implicit time integration combined with constraint stabilization.

  See Compliant/doc/compliant.pdf and Compliant/doc/compliant-reference.pdf for more detail.

  See also class sofa::component::odesolver::CompliantImplicitSolver

  A test suite is available, see page \ref Page_CompliantTestSuite

  @author Francois Faure, Maxime Tournier, Matthieu Nesme, Benjamin Gilles

  */

namespace sofa
{
	SOFA_Compliant_API void initCompliant();


}

#endif // INITCompliant_H
