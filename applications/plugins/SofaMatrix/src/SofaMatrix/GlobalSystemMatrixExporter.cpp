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
#include <SofaMatrix/GlobalSystemMatrixExporter.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream>
#include <sofa/defaulttype/MatrixExporter.h>

namespace sofa::component::linearsolver
{

int GlobalSystemMatrixExporterClass = core::RegisterObject("Export the global system matrix from a linear solver.")
        .add<GlobalSystemMatrixExporter>();

GlobalSystemMatrixExporter::GlobalSystemMatrixExporter()
: Inherit1()
, d_fileFormat(initData(&d_fileFormat, sofa::defaulttype::matrixExporterOptionsGroup, "format", "File format"))
, d_precision(initData(&d_precision, 6, "precision", "Number of digits used to write an entry of the matrix, default is 6"))
, l_linearSolver(initLink("linearSolver", "Linear solver used to export its matrix"))
{
    d_exportAtBegin.setReadOnly(true);
    d_exportAtEnd.setReadOnly(true);

    d_exportAtBegin.setDisplayed(false);
    d_exportAtEnd.setDisplayed(false);
}

void GlobalSystemMatrixExporter::doInit()
{
    if (!l_linearSolver)
    {
        l_linearSolver.set(this->getContext()->template get<sofa::core::behavior::LinearSolver>());
    }

    if (!l_linearSolver)
    {
        msg_error() << "No linear solver found in the current context, whereas it is required. This component exports the matrix from a linear solver.";
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

bool GlobalSystemMatrixExporter::write()
{
    if (l_linearSolver)
    {
        if (l_linearSolver->getSystemBaseMatrix())
        {
            const std::string basename = getOrCreateTargetPath(d_filename.getValue(),
                                                               d_exportEveryNbSteps.getValue());

            const auto selectedExporter = d_fileFormat.getValue().getSelectedItem();
            const auto exporter = sofa::defaulttype::matrixExporterMap.find(selectedExporter);
            if (exporter != sofa::defaulttype::matrixExporterMap.end())
            {
                const std::string filename = basename + "." + exporter->first;
                msg_info() << "Writing global system matrix from linear solver '" << l_linearSolver->getName() << "' in " << filename;
                return exporter->second(filename, l_linearSolver->getSystemBaseMatrix(), d_precision.getValue());
            }
        }
        else
        {
            msg_warning() << "Matrix cannot be exported, probably because the linear solver '"
                          << l_linearSolver->getName() << "' does not assemble explicitly the system matrix.";
        }
    }
    return false;
}

}