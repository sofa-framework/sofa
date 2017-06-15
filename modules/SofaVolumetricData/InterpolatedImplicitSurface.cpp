/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include <SofaVolumetricData/InterpolatedImplicitSurface.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace container
{

SOFA_DECL_CLASS(InterpolatedImplicitSurface)

// Register in the Factory
int InterpolatedImplicitSurfaceClass = core::RegisterObject("This class allows to store and query a distancemap loaded from file.")
        .add< InterpolatedImplicitSurface >()
        ;


InterpolatedImplicitSurface::InterpolatedImplicitSurface()
    : ImplicitSurface(),
      distanceMapHeader( initData( &distanceMapHeader, "file", "MHD file for the distance map" ) ),
      maxDomains( initData( &maxDomains, 1, "maxDomains", "Number of domains available for caching" ) ),
      dx( initData( &dx, 0.0, "dx", "x translation" ) ),
      dy( initData( &dy, 0.0, "dy", "y translation" ) ),
      dz( initData( &dz, 0.0, "dz", "z translation" ) )
{
    usedDomains = 0;
    imgData = 0;
}


InterpolatedImplicitSurface::~InterpolatedImplicitSurface()
{
    if (imgData)
    {
        delete[] imgData;
        imgData = 0;
    }
}


void InterpolatedImplicitSurface::init()
{
    domainCache.resize( maxDomains.getValue() );
    bool ok = loadImage( distanceMapHeader.getFullPath().c_str() );
    if (ok) printf( "Successfully loaded distance map.\n" );
}


bool InterpolatedImplicitSurface::loadImage( const char *mhdFile )
{
    imgMin[0]=imgMin[1]=imgMin[2] = 0;
    spacing[0]=spacing[1]=spacing[2] = 1;
    imgSize[0]=imgSize[1]=imgSize[2] = 0;

    char buffer[1024];
    char *value;
    bool dataFileSpecified = false;
    float f0, f1, f2;
    int i0, i1, i2;
    char dataFile[1024];

    // read header file
    std::ifstream header( mhdFile );
    if (!header.is_open()) return false;
    while (!header.eof())
    {
        header.getline( buffer, 1024 );
        if (strncmp( buffer, "ObjectType", 10 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (strncmp( value, "Image", 5 ) != 0)
            {
                printf( "ERROR: Object is no image.\n" );
                return false;
            }
        }
        else if (strncmp( buffer, "NDims", 5 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (*value != '3')
            {
                printf( "ERROR: Wrong number of dimensions.\n" );
                return false;
            }
        }
        else if (strncmp( buffer, "BinaryData ", 11 ) == 0 || strncmp( buffer, "BinaryData=", 11 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (strncmp( value, "True", 4 ) != 0)
            {
                printf( "ERROR: Data is not binary.\n" );
                return false;
            }
        }
        else if (strncmp( buffer, "CompressedData", 14 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (strncmp( value, "False", 5 ) != 0)
            {
                printf( "ERROR: Data is compressed.\n" );
                return false;
            }
        }
        else if (strncmp( buffer, "TransformMatrix", 15 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (strncmp( value, "1 0 0 0 1 0 0 0 1", 17 ) != 0)
            {
                printf( "ERROR: Unsupported transform matrix.\n" );
                return false;
            }
        }
        else if (strncmp( buffer, "Offset", 6 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            sscanf( value, "%f %f %f", &f0, &f1, &f2 );
            imgMin[0]=f0;  imgMin[1]=f1;  imgMin[2]=f2;
            printf( "Image offset = %f %f %f\n", imgMin[0], imgMin[1], imgMin[2] );
        }
        else if (strncmp( buffer, "ElementSpacing", 14 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            sscanf( value, "%f %f %f", &f0, &f1, &f2 );
            spacing[0]=f0;  spacing[1]=f1;  spacing[2]=f2;
            printf( "Image spacing = %f %f %f\n", spacing[0], spacing[1], spacing[2] );
        }
        else if (strncmp( buffer, "DimSize", 7 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            sscanf( value, "%d %d %d", &i0, &i1, &i2 );
            imgSize[0]=i0;  imgSize[1]=i1;  imgSize[2]=i2;
            printf( "Image size = %i %i %i\n", imgSize[0], imgSize[1], imgSize[2] );
        }
        else if (strncmp( buffer, "ElementType", 11 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            if (strncmp( value, "MET_FLOAT", 9) != 0)
            {
                printf( "ERROR: Datatype is not supported.\n" );
                return false;
            }
        }
        /* Don't allow variable names for problems with correct file paths!
        else if (strncmp( buffer, "ElementDataFile", 11 ) == 0) {
          value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
          strcpy( dataFile, value );
          dataFileSpecified = true;
        }*/
    }
    header.close();

    // init remaining variables
    for (int d=0; d<3; d++)
    {
        scale[d] = 1.0/spacing[d];
        imgMax[d] = imgMin[d] + (double)(imgSize[d]-1)*spacing[d];
    }
    deltaOfs[0] = 0;
    deltaOfs[1] = 1;
    deltaOfs[2] = imgSize[0];
    deltaOfs[3] = imgSize[0]+1;
    unsigned int sliceSize = imgSize[0]*imgSize[1];
    deltaOfs[4] = deltaOfs[0] + sliceSize;
    deltaOfs[5] = deltaOfs[1] + sliceSize;
    deltaOfs[6] = deltaOfs[2] + sliceSize;
    deltaOfs[7] = deltaOfs[3] + sliceSize;

    // read data file
    if (!dataFileSpecified)
    {
        // change extension to .raw
        strcpy( dataFile, mhdFile );
        int lenWithoutExt = strlen( mhdFile ) - 3;
        dataFile[lenWithoutExt] = 0;
        strcat( dataFile, "raw" );
    }
    std::ifstream data( dataFile, std::ios_base::binary|std::ios_base::in );
    if (!data.is_open()) return false;
    unsigned int numVoxels = imgSize[0]*imgSize[1]*imgSize[2];
    imgData = new float[numVoxels];
    data.read( (char*)imgData, numVoxels*sizeof(float) );
    if (data.bad()) return false;
    data.close();
    return true;
}


} // namespace container

} // namespace component

} // namespace sofa
