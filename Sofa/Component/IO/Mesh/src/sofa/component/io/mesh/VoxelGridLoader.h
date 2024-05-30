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
#include <sofa/component/io/mesh/config.h>

#include <sofa/core/loader/VoxelLoader.h>

namespace sofa::component::io::mesh
{

class SOFA_COMPONENT_IO_MESH_API VoxelGridLoader : public sofa::core::loader::VoxelLoader
{
public:
    SOFA_CLASS(VoxelGridLoader,VoxelLoader);

    typedef type::Vec<3, int> Vec3i;
    typedef type::Vec<6, int> Vec6i;
    typedef type::fixed_array<unsigned int,8> Hexahedron;
protected:
    VoxelGridLoader();
    ~VoxelGridLoader() override;

public:
    void init() override;

    void reinit() override;

    virtual void clear();

    bool load() override;
    bool canLoad() override;

    void setVoxelSize ( const type::Vec3 vSize );
    type::Vec3 getVoxelSize () const override;

    void addBackgroundValue ( const int value );
    int getBackgroundValue( const unsigned int idx = 0) const;

    void addActiveDataValue(const int value);
    int getActiveDataValue(const unsigned int idx = 0) const;

    void getResolution ( Vec3i& res ) const;

    int getDataSize() const override;

    unsigned char * getData() override;
    unsigned char * getSegmentID() override;

    type::vector<unsigned int> getHexaIndicesInGrid() const override;

    Vec6i getROI() const override;

    // fill the texture by 'image' only where there is the 'segmentation' of 'd_activeValue' and give the 3D texture sizes
    void createSegmentation3DTexture( unsigned char **textureData, int& width, int& height, int& depth) override;
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< type::Vec3 > voxelSize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< Vec3i > dataResolution;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< Vec6i > roi;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< int > headerSize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< int > segmentationHeaderSize;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< type::vector<unsigned int> > idxInRegularGrid;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< type::vector<int> > backgroundValue;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data< type::vector<int> > activeValue;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_IO_MESH()
    Data<bool> generateHexa;



    Data< type::Vec3 > d_voxelSize; ///< Dimension of one voxel
    Data< Vec3i > d_dataResolution; ///< Resolution of the voxel file
    Data< Vec6i > d_roi; ///< Region of interest (xmin, ymin, zmin, xmax, ymax, zmax)
    Data< int > d_headerSize; ///< Header size in bytes
    Data< int > d_segmentationHeaderSize; ///< Header size in bytes
    Data< type::vector<unsigned int> > d_idxInRegularGrid; ///< indices of the hexa in the grid.

    Data< type::vector<int> > d_backgroundValue; ///< Background values (to be ignored)
    Data< type::vector<int> > d_activeValue; ///< Active data values

    Data<bool> d_generateHexa; ///< Interpret voxel as either hexa or points

private:
    void setResolution ( const Vec3i res );

    bool isActive(const unsigned int idx) const;

    helper::io::Image* loadImage ( const std::string& filename, const Vec3i& res, const int hsize) const;

protected:

    helper::io::Image* image;
    helper::io::Image* segmentation;

    int bpp;
};

} //namespace sofa::component::io::mesh
