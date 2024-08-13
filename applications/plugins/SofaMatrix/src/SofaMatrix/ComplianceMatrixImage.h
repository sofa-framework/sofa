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
#include <SofaMatrix/config.h>

#include <sofa/core/objectmodel/BaseObject.h>
#include <SofaMatrix/BaseMatrixImageProxy.h>
#include <sofa/component/constraint/lagrangian/solver/ConstraintSolverImpl.h>

namespace sofa::component::constraintset
{

/**
 * Component to convert a BaseMatrix from the constraint solver into an image that can be visualized in the GUI.
 * Use ComplianceMatrixExporter in order to save an image on the disk.
 *
 * Note that the compliance matrix is dense. It means all the entries will proably be non-zero
 */
class SOFA_SOFAMATRIX_API ComplianceMatrixImage : public core::objectmodel::BaseObject
{
public:
    SOFA_CLASS(ComplianceMatrixImage, core::objectmodel::BaseObject);

protected:

    ComplianceMatrixImage();
    ~ComplianceMatrixImage() override;

    void init() override;
    void handleEvent(core::objectmodel::Event *event) override;

    Data< type::BaseMatrixImageProxy > d_bitmap; ///< Visualization of the representation of the matrix as a binary image. White pixels are zeros, black pixels are non-zeros.
    SingleLink<ComplianceMatrixImage, sofa::component::constraint::lagrangian::solver::ConstraintSolverImpl, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_constraintSolver;
};

} //namespace sofa::component::constraintset