/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LOADER_VOXELGRIDLOADER
#define SOFA_COMPONENT_LOADER_VOXELGRIDLOADER
#include "config.h"

#include <sofa/core/loader/VoxelLoader.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/helper/fixed_array.h>

namespace sofa
{
namespace helper { namespace io { class Image; }}

namespace component
{

namespace loader
{


class SOFA_GENERAL_LOADER_API VoxelGridLoader : public sofa::core::loader::VoxelLoader
{
public:
    SOFA_CLASS(VoxelGridLoader,VoxelLoader);

    typedef defaulttype::Vec<3, int> Vec3i;
    typedef defaulttype::Vec<6, int> Vec6i;
    typedef helper::fixed_array<unsigned int,8> Hexahedron;
protected:
    VoxelGridLoader();
    virtual ~VoxelGridLoader();
public:
    virtual void init() override;

    virtual void reinit() override;

    virtual void clear();

    virtual bool load() override;
    virtual bool canLoad() override;

    void setVoxelSize ( const defaulttype::Vector3 vSize );
    defaulttype::Vector3 getVoxelSize () const override;

    void addBackgroundValue ( const int value );
    int getBackgroundValue( const unsigned int idx = 0) const;

    void addActiveDataValue(const int value);
    int getActiveDataValue(const unsigned int idx = 0) const;

    void getResolution ( Vec3i& res ) const;

    int getDataSize() const override;

    unsigned char * getData() override;
    unsigned char * getSegmentID() override;

    helper::vector<unsigned int> getHexaIndicesInGrid() const override;

    Vec6i getROI() const override;

    // fill the texture by 'image' only where there is the 'segmentation' of 'activeValue' and give the 3D texture sizes
    void createSegmentation3DTexture( unsigned char **textureData, int& width, int& height, int& depth) override;

    Data< defaulttype::Vector3 > voxelSize; ///< Dimension of one voxel
    Data< Vec3i > dataResolution; ///< Resolution of the voxel file
    Data< Vec6i > roi; ///< Region of interest (xmin, ymin, zmin, xmax, ymax, zmax)
    Data< int > headerSize; ///< Header size in bytes
    Data< int > segmentationHeaderSize; ///< Header size in bytes
    Data< helper::vector<unsigned int> > idxInRegularGrid; ///< indices of the hexa in the grid.

    Data< helper::vector<int> > backgroundValue; ///< Background values (to be ignored)
    Data< helper::vector<int> > activeValue; ///< Active data values

    Data<bool> generateHexa; ///< Interpret voxel as either hexa or points

private:
    void setResolution ( const Vec3i res );

    bool isActive(const unsigned int idx) const;

    helper::io::Image* loadImage ( const std::string& filename, const Vec3i& res, const int hsize) const;

protected:

    helper::io::Image* image;
    helper::io::Image* segmentation;

    int bpp;
};

}

} // namespace component

} // namespace sofa

#endif
