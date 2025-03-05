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
//
// C++ Interface: GetAssembledSizeVisitor
//
// Description:
//
//
// Author: Francois Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_GetAssembledSizeVisitor_H
#define SOFA_SIMULATION_GetAssembledSizeVisitor_H



#include <SceneCreator/config.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/linearalgebra/BaseVector.h>


namespace sofa::simulation
{

/** Compute the size of the assembled position vector and velocity vector.
  Only the independent DOFs are considered.
  The two values may be different, such as for rigid objects.
    Francois Faure, 2013
*/
 class SOFA_SCENECREATOR_API GetAssembledSizeVisitor: public Visitor
{
public:
    GetAssembledSizeVisitor( const sofa::core::ExecParams* params= sofa::core::mechanicalparams::castToExecParams(core::mechanicalparams::defaultInstance()));
    ~GetAssembledSizeVisitor() override;

    Result processNodeTopDown( simulation::Node*  ) override;
    const char* getClassName() const override { return "GetAssembledSizeVisitor"; }

    unsigned positionSize() const { return xsize; }
    unsigned velocitySize() const { return vsize; }
    void setIndependentOnly( bool );

protected:
    std::size_t xsize;
    std::size_t vsize;
    bool independentOnly;
};

} // namespace sofa::simulation


#endif
