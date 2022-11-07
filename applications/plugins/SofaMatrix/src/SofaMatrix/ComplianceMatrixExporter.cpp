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
#include <SofaMatrix/ComplianceMatrixExporter.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream>
#include <sofa/defaulttype/MatrixExporter.h>

namespace sofa::component::constraintset
{

int ComplianceMatrixExporterClass = core::RegisterObject("Export the compliance matrix from a constraint solver.")
        .add<ComplianceMatrixExporter>();

ComplianceMatrixExporter::ComplianceMatrixExporter()
: Inherit1()
, d_fileFormat(initData(&d_fileFormat, sofa::defaulttype::matrixExporterOptionsGroup, "format", "File format"))
, d_precision(initData(&d_precision, 6, "precision", "Number of digits used to write an entry of the matrix, default is 6"))
, l_constraintSolver(initLink("constraintSolver", "Constraint solver used to export its compliance matrix"))
{
    d_exportAtBegin.setReadOnly(true);
    d_exportAtEnd.setReadOnly(true);

    d_exportAtBegin.setDisplayed(false);
    d_exportAtEnd.setDisplayed(false);
}

void ComplianceMatrixExporter::doInit()
{
    if (!l_constraintSolver)
    {
        l_constraintSolver.set(this->getContext()->template get<sofa::component::constraint::lagrangian::solver::ConstraintSolverImpl>());
    }

    if (!l_constraintSolver)
    {
        msg_error() << "No constraint solver found in the current context, whereas it is required. This component exports the matrix from a constraint solver.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

bool ComplianceMatrixExporter::write()
{
    if (l_constraintSolver)
    {
        if (auto* constraintProblem = l_constraintSolver->getConstraintProblem())
        {
            const std::string basename = getOrCreateTargetPath(d_filename.getValue(),
                                                               d_exportEveryNbSteps.getValue());

            const auto selectedExporter = d_fileFormat.getValue().getSelectedItem();
            const auto exporter = sofa::defaulttype::matrixExporterMap.find(selectedExporter);
            if (exporter != sofa::defaulttype::matrixExporterMap.end())
            {
                const std::string filename = basename + "." + exporter->first;
                msg_info() << "Writing compliance matrix from constraint solver '" << l_constraintSolver->getName() << "' in " << filename;
                return exporter->second(filename, &constraintProblem->W, d_precision.getValue());
            }
        }
        else
        {
            msg_warning() << "Matrix cannot be exported, probably because the constraint problem is not yet built";
        }
    }
    return false;
}

} // namespace sofa::component::constraintset