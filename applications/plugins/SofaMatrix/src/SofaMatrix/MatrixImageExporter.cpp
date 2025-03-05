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
#include <SofaMatrix/MatrixImageExporter.h>
#include <sofa/defaulttype/MatrixExporter.h>
#include <sofa/linearalgebra/BaseMatrix.h>
#include <sofa/helper/io/STBImage.h>


namespace sofa::defaulttype
{
bool writeMatrixImage(const std::string& filename, sofa::linearalgebra::BaseMatrix* matrix, int precision)
{
    SOFA_UNUSED(precision);
    if (matrix)
    {
        const auto nx = matrix->colSize();
        const auto ny = matrix->rowSize();

        sofa::helper::io::STBImage image;
        image.init(nx, ny, 1, 1, sofa::helper::io::Image::DataType::UNORM8, sofa::helper::io::Image::ChannelFormat::L);

        unsigned char* pixels = image.getPixels();
        for (sofa::SignedIndex y = 0; y < ny; ++y)
        {
            for (sofa::SignedIndex x = 0; x < nx; ++x)
            {
                pixels[y * nx + x] = !static_cast<bool>(matrix->element(nx - y - 1, x)) * std::numeric_limits<unsigned char>::max();
            }
        }
        return image.save(filename);
    }
    return false;
}
} //namespace sofa::defaulttype

namespace sofa::component
{

void initializeMatrixExporterComponents()
{
    static bool first = true;
    if (first)
    {
        constexpr auto addMatrixExporter = [](const std::string& format, sofa::defaulttype::MatrixExportFunction exporter)
        {
            //Add an exporter which writes a matrix as an image
            sofa::defaulttype::matrixExporterMap.insert({format, exporter});

            //The added exporter is made available in the options group
            sofa::defaulttype::matrixExporterOptionsGroup.setNbItems(sofa::defaulttype::matrixExporterOptionsGroup.size() + 1);
            sofa::defaulttype::matrixExporterOptionsGroup.setItemName(sofa::defaulttype::matrixExporterOptionsGroup.size() - 1, format);
        };

        addMatrixExporter("png", sofa::defaulttype::writeMatrixImage);
        addMatrixExporter("jpg", sofa::defaulttype::writeMatrixImage);
        addMatrixExporter("jpeg", sofa::defaulttype::writeMatrixImage);
        addMatrixExporter("bmp", sofa::defaulttype::writeMatrixImage);
        addMatrixExporter("tga", sofa::defaulttype::writeMatrixImage);

        first = false;
    }
}

} //namespace sofa::component
