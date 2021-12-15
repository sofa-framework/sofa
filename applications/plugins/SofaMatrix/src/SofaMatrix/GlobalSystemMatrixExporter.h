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
#include <sofa/simulation/BaseSimulationExporter.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/OptionsGroup.h>

namespace sofa::component::linearsolver
{

/**
 * @brief Exports the global system matrix of the current context into a file. The exporter allows to write the file
 * under several formats.
 *
 * The class is designed so more file format can be supported.
 */
class SOFA_SOFAMATRIX_API GlobalSystemMatrixExporter : public sofa::simulation::BaseSimulationExporter
{
public:
    SOFA_CLASS(GlobalSystemMatrixExporter, sofa::simulation::BaseSimulationExporter);

    bool write() override;
    void doInit() override;

protected:
    Data<sofa::helper::OptionsGroup> d_fileFormat;

    GlobalSystemMatrixExporter();

    SingleLink<GlobalSystemMatrixExporter, sofa::core::behavior::LinearSolver, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> l_linearSolver;
};
}