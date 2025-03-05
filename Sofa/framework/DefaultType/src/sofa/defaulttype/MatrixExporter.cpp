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
#include <sofa/defaulttype/MatrixExporter.h>
#include <sofa/linearalgebra/BaseMatrix.h>

#include <fstream>
#include <iomanip>

namespace sofa::defaulttype
{

std::unordered_map<std::string, MatrixExportFunction> matrixExporterMap
{
    {"txt", writeMatrixTxt},
    {"csv", writeMatrixCsv},
};
sofa::helper::OptionsGroup matrixExporterOptionsGroup{"txt", "csv"};
    
bool writeMatrixTxt(const std::string& filename, sofa::linearalgebra::BaseMatrix* matrix, int precision)
{
    if (matrix)
    {
        std::ofstream file(filename);
        file << std::setprecision(precision) << *matrix;
        file.close();

        return true;
    }
    return false;
}

bool writeMatrixCsv(const std::string& filename, sofa::linearalgebra::BaseMatrix* matrix, int precision)
{
    if (matrix)
    {
        std::ofstream file(filename);
        file << std::setprecision(precision);

        const auto nx = matrix->colSize();
        const auto ny = matrix->rowSize();

        if (nx > 0)
        {
            for (sofa::SignedIndex y = 0; y<ny; ++y)
            {
                for (sofa::SignedIndex x = 0; x < nx - 1; ++x)
                {
                    file << matrix->element(y, x) << ",";
                }
                file << matrix->element(y, nx - 1) << "\n";
            }
        }

        file.close();

        return true;
    }
    return false;
}

}
