/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "config.h"

#include <SofaLoader/BaseVTKReader.h>
#include <SofaLoader/BaseVTKReader.inl>


namespace sofa
{

namespace component
{

namespace loader
{

namespace basevtkreader
{


BaseVTKReader::BaseVTKReader():inputPoints (NULL), inputPolygons(NULL), inputCells(NULL),
                               inputCellOffsets(NULL), inputCellTypes(NULL),
                               numberOfPoints(0),numberOfCells(0)
{}

BaseVTKReader::BaseVTKDataIO* BaseVTKReader::newVTKDataIO(const string& typestr)
{
    if  (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8"))
        return new VTKDataIO<char>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_char") || !strcasecmp(typestr.c_str(), "UInt8"))
        return new VTKDataIO<unsigned char>;
    else if (!strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int16"))
        return new VTKDataIO<short>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_short") || !strcasecmp(typestr.c_str(), "UInt16"))
        return new VTKDataIO<unsigned short>;
    else if (!strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
        return new VTKDataIO<int>;
    else if (!strcasecmp(typestr.c_str(), "unsigned_int") || !strcasecmp(typestr.c_str(), "UInt32"))
        return new VTKDataIO<unsigned int>;
    //else if (!strcasecmp(typestr.c_str(), "long") || !strcasecmp(typestr.c_str(), "Int64"))
    //	return new VTKDataIO<long long>;
    //else if (!strcasecmp(typestr.c_str(), "unsigned_long") || !strcasecmp(typestr.c_str(), "UInt64"))
    // 	return new VTKDataIO<unsigned long long>;
    else if (!strcasecmp(typestr.c_str(), "float") || !strcasecmp(typestr.c_str(), "Float32"))
        return new VTKDataIO<float>;
    else if (!strcasecmp(typestr.c_str(), "double") || !strcasecmp(typestr.c_str(), "Float64"))
        return new VTKDataIO<double>;
    else return NULL;
}

BaseVTKReader::BaseVTKDataIO* BaseVTKReader::newVTKDataIO(const string& typestr, int num)
{
    BaseVTKDataIO* result = NULL;

    if (num == 1)
        result = newVTKDataIO(typestr);
    else
    {
        if (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8") ||
                !strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int32") ||
                !strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
        {
            switch (num)
            {
            case 2:
                result = new VTKDataIO<Vec<2, int> >;
                break;
            case 3:
                result = new VTKDataIO<Vec<3, int> >;
                break;
            default:
                return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "unsigned char") || !strcasecmp(typestr.c_str(), "UInt8") ||
                !strcasecmp(typestr.c_str(), "unsigned short") || !strcasecmp(typestr.c_str(), "UInt32") ||
                !strcasecmp(typestr.c_str(), "unsigned int") || !strcasecmp(typestr.c_str(), "UInt32"))
        {
            switch (num)
            {
            case 2:
                result = new VTKDataIO<Vec<2, unsigned int> >;
                break;
            case 3:
                result = new VTKDataIO<Vec<3, unsigned int> >;
                break;
            default:
                return NULL;
            }
        }
        if (!strcasecmp(typestr.c_str(), "float") || !strcasecmp(typestr.c_str(), "Float32"))
        {
            switch (num)
            {
            case 2:
                result = new VTKDataIO<Vec<2, float> >;
                break;
            case 3:
                result = new VTKDataIO<Vec<3, float> >;
                break;
            default:
                return NULL;
            }
        }
        if (!strcasecmp(typestr.c_str(), "double") || !strcasecmp(typestr.c_str(), "Float64"))
        {
            switch (num)
            {
            case 2:
                result = new VTKDataIO<Vec<2, double> >;
                break;
            case 3:
                result = new VTKDataIO<Vec<3, double> >;
                break;
            default:
                return NULL;
            }
        }
    }
    result->nestedDataSize = num;
    return result;
}

bool BaseVTKReader::readVTK(const char* filename)
{
    bool state = readFile(filename);
    return state;
}

} // basevtkreader

} // namespace loader

} // namespace component

} // namespace sofa
