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

#include <sofa/core/loader/BaseLoader.h>

namespace sofa
{
namespace helper::io
{
    class Image;
}

namespace core::loader
{


class SOFA_CORE_API VoxelLoader : public sofa::core::loader::BaseLoader
{
public:
    SOFA_ABSTRACT_CLASS(VoxelLoader,BaseLoader);

    typedef type::Vec<3, int> Vec3i;
    typedef type::Vec<6, int> Vec6i;
    typedef type::fixed_array<unsigned int,8> Hexahedron;
protected:
    VoxelLoader();
    ~VoxelLoader() override;
public:

    Data< type::vector<sofa::type::Vec3 > > positions; ///< Coordinates of the nodes loaded
    Data< type::vector<Hexahedron > > hexahedra;       ///< Hexahedra loaded


    void addHexahedron(type::vector< Hexahedron >* pHexahedra, const type::fixed_array<unsigned int,8> &p);
    void addHexahedron(type::vector< Hexahedron >* pHexahedra,
                       unsigned int p0, unsigned int p1, unsigned int p2, unsigned int p3,
                       unsigned int p4, unsigned int p5, unsigned int p6, unsigned int p7);

    virtual type::Vec3 getVoxelSize () const = 0;

    virtual type::vector<unsigned int> getHexaIndicesInGrid() const=0;

    virtual int getDataSize() const = 0;

    virtual Vec6i getROI() const = 0;

    virtual unsigned char * getData() = 0;

    virtual unsigned char * getSegmentID() = 0;

    // fill the texture by 'image' only where there is the 'segmentation' of 'd_activeValue' and give the 3D texture sizes
    virtual void createSegmentation3DTexture( unsigned char **textureData, int& width, int& height, int& depth) = 0;
};

} // namespace core::loader

} // namespace sofa
