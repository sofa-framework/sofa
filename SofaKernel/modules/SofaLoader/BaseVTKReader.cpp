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
#include "config.h"

#include <SofaLoader/BaseVTKReader.h>
#include <SofaLoader/BaseVTKReader.inl>

#if defined(WIN32) || defined(_XBOX)
#define strcasecmp stricmp
#endif

#include <cstdint>

namespace sofa
{

namespace component
{

namespace loader
{

namespace basevtkreader
{


BaseVTKReader::BaseVTKReader(): inputPoints (NULL), inputNormals (NULL), inputPolygons(NULL), inputCells(NULL),
    inputCellOffsets(NULL), inputCellTypes(NULL),
    numberOfPoints(0), numberOfCells(0)
{}

BaseVTKReader::BaseVTKDataIO* BaseVTKReader::newVTKDataIO(const string& typestr)
{
    if  (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8"))
    {
        return new VTKDataIO<char>;
    }
    else if (!strcasecmp(typestr.c_str(), "unsigned_char") || !strcasecmp(typestr.c_str(), "UInt8"))
    {
        return new VTKDataIO<std::uint8_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int16"))
    {
        return new VTKDataIO<std::int16_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "unsigned_short") || !strcasecmp(typestr.c_str(), "UInt16"))
    {
        return new VTKDataIO<std::uint16_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
    {
        return new VTKDataIO<std::int32_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "unsigned_int") || !strcasecmp(typestr.c_str(), "UInt32"))
    {
        return new VTKDataIO<std::uint32_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "long") || !strcasecmp(typestr.c_str(), "Int64"))
    {
        return new VTKDataIO<std::int64_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "unsigned_long") || !strcasecmp(typestr.c_str(), "UInt64"))
    {
        return new VTKDataIO<std::uint64_t>;
    }
    else if (!strcasecmp(typestr.c_str(), "float") || !strcasecmp(typestr.c_str(), "Float32"))
    {
        return new VTKDataIO<float>;
    }
    else if (!strcasecmp(typestr.c_str(), "double") || !strcasecmp(typestr.c_str(), "Float64"))
    {
        return new VTKDataIO<double>;
    }
    else
    {
        return NULL;
    }
}

BaseVTKReader::BaseVTKDataIO* BaseVTKReader::newVTKDataIO(const string& typestr, int num)
{
    BaseVTKDataIO* result = NULL;

    if (num == 1)
    {
        result = newVTKDataIO(typestr);
    }
    else
    {
        if (!strcasecmp(typestr.c_str(), "char") || !strcasecmp(typestr.c_str(), "Int8"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, char> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, char> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, char> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "unsigned_char") || !strcasecmp(typestr.c_str(), "UInt8"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::uint8_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::uint8_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::uint8_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "short") || !strcasecmp(typestr.c_str(), "Int16"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::int16_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::int16_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::int16_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "unsigned_short") || !strcasecmp(typestr.c_str(), "UInt16"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::uint16_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::uint16_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::uint16_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "int") || !strcasecmp(typestr.c_str(), "Int32"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::int32_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::int32_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::int32_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "unsigned_int") || !strcasecmp(typestr.c_str(), "UInt32"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::uint32_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::uint32_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::uint32_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "long") || !strcasecmp(typestr.c_str(), "Int64"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::int64_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::int64_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::int64_t> >;
                    break;
                default:
                    return NULL;
            }
        }

        if (!strcasecmp(typestr.c_str(), "unsigned_long") || !strcasecmp(typestr.c_str(), "UInt64"))
        {
            switch (num)
            {
                case 2:
                    result = new VTKDataIO<Vec<2, std::uint64_t> >;
                    break;
                case 3:
                    result = new VTKDataIO<Vec<3, std::uint64_t> >;
                    break;
                case 4:
                    result = new VTKDataIO<Vec<4, std::uint64_t> >;
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
                case 4:
                    result = new VTKDataIO<Vec<4, float> >;
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
                case 4:
                    result = new VTKDataIO<Vec<4, double> >;
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
    return readFile(filename);
}

} // basevtkreader

} // namespace loader

} // namespace component

} // namespace sofa
