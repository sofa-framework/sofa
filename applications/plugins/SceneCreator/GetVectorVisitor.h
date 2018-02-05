/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
// C++ Interface: GetVectorVisitor
//
// Description:
//
//
// Author: Francois Faure, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_SIMULATION_GetVectorVisitor_H
#define SOFA_SIMULATION_GetVectorVisitor_H

#if !defined(__GNUC__) || (__GNUC__ > 3 || (_GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#include <SceneCreator/config.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/defaulttype/BaseVector.h>
#include <Eigen/Dense>


namespace sofa
{

namespace simulation
{

/** Copy a given MultiVector (generally spread across the MechanicalStates) to a BaseVector
    Only the independent DOFs are used.
    Francois Faure, 2013
*/
class SOFA_SCENECREATOR_API GetVectorVisitor: public Visitor
{
public:
//    typedef Eigen::Matrix<SReal, Eigen::Dynamic, 1> Vector;
    typedef defaulttype::BaseVector Vector;
    GetVectorVisitor( const sofa::core::ExecParams* params, Vector* vec, core::ConstVecId src );
    virtual ~GetVectorVisitor();

    virtual Result processNodeTopDown( simulation::Node*  );
    virtual const char* getClassName() const { return "GetVectorVisitor"; }

    /// If true, process the independent nodes only
    void setIndependentOnly( bool );

protected:
    Vector* vec;
    core::ConstVecId src;
    unsigned offset;
    bool independentOnly;

};

} // namespace simulation
} // namespace sofa

#endif
