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
#include <fstream>

#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject ;

#include "DiscreteGridField.h"


namespace sofa
{

namespace component
{

namespace geometry
{

namespace _discretegrid_
{
/**
DiscreteGridField::DiscreteGridField()
    : in_filename(initData(&in_filename,"filename","filename"))
    , in_nx(initData(&in_nx,0,"nx","in_nx"))
    , in_ny(initData(&in_ny,0,"ny","in_ny"))
    , in_nz(initData(&in_nz,0,"nz","in_nz"))
    , in_scale(initData(&in_scale,0.0,"scale","in_scale"))
    , in_sampling(initData(&in_sampling,0.0,"sampling","in_sampling"))
{
}



void DiscreteGridField::init()
{
    if(in_nx.getValue()==0 && in_nz.getValue()==0 && in_nz.getValue()==0) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "uninitialized grid";
    }
    else if(in_filename.isSet() == false) {
        m_componentstate = ComponentState::Invalid;
        msg_error() << "unset filename";
    }
    else {
        pmin.set(0,0,-5.0);
        pmax.set(27,27,5.0);
        loadGrid(in_scale.getValue(),in_sampling.getValue(),in_nx.getValue(),in_ny.getValue(),in_nz.getValue(),pmin,pmax);
    }

    m_componentstate = ComponentState::Valid;
}
*/

DiscreteGridField::DiscreteGridField()
    : ScalarField(),
      d_distanceMapHeader( initData( &d_distanceMapHeader, "file", "MHD file for the distance map" ) ),
      d_maxDomains( initData( &d_maxDomains, 1, "maxDomains", "Number of domains available for caching" ) ),
      dx( initData( &dx, 0.0, "dx", "x translation" ) ),
      dy( initData( &dy, 0.0, "dy", "y translation" ) ),
      dz( initData( &dz, 0.0, "dz", "z translation" ) )
{
    m_usedDomains = 0;
    m_imgData = 0;
}


DiscreteGridField::~DiscreteGridField()
{
    if (m_imgData)
    {
        delete[] m_imgData;
        m_imgData = 0;
    }
}


///used to set a name in tests
void DiscreteGridField::setFilename(const std::string& name)
{
    d_distanceMapHeader.setValue(name);
}


void DiscreteGridField::init()
{
    m_domainCache.resize( d_maxDomains.getValue() );
    bool ok = loadGridFromMHD( d_distanceMapHeader.getFullPath().c_str() );
    if (ok) printf( "Successfully loaded distance map.\n" );
}


bool DiscreteGridField::loadGridFromMHD( const char *filename )
{
    m_imgMin[0]=m_imgMin[1]=m_imgMin[2] = 0;
    m_spacing[0]=m_spacing[1]=m_spacing[2] = 1;
    m_imgSize[0]=m_imgSize[1]=m_imgSize[2] = 0;

    char buffer[1024];
    char *value;
    bool dataFileSpecified = false;
    float f0, f1, f2;
    int i0, i1, i2;
    char dataFile[1024];

    // read header file
    std::ifstream header( filename );
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
            m_imgMin[0]=f0;  m_imgMin[1]=f1;  m_imgMin[2]=f2;
            printf( "Image offset = %f %f %f\n", m_imgMin[0], m_imgMin[1], m_imgMin[2] );
        }
        else if (strncmp( buffer, "ElementSpacing", 14 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            sscanf( value, "%f %f %f", &f0, &f1, &f2 );
            m_spacing[0]=f0;  m_spacing[1]=f1;  m_spacing[2]=f2;
            printf( "Image spacing = %f %f %f\n", m_spacing[0], m_spacing[1], m_spacing[2] );
        }
        else if (strncmp( buffer, "DimSize", 7 ) == 0)
        {
            value = strchr( buffer, '=' )+1;  while (*value==' ') value++;
            sscanf( value, "%d %d %d", &i0, &i1, &i2 );
            m_imgSize[0]=i0;  m_imgSize[1]=i1;  m_imgSize[2]=i2;
            printf( "Image size = %i %i %i\n", m_imgSize[0], m_imgSize[1], m_imgSize[2] );
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
        m_scale[d] = 1.0/m_spacing[d];
        m_imgMax[d] = m_imgMin[d] + (double)(m_imgSize[d]-1)*m_spacing[d];
    }
    m_deltaOfs[0] = 0;
    m_deltaOfs[1] = 1;
    m_deltaOfs[2] = m_imgSize[0];
    m_deltaOfs[3] = m_imgSize[0]+1;
    unsigned int sliceSize = m_imgSize[0]*m_imgSize[1];
    m_deltaOfs[4] = m_deltaOfs[0] + sliceSize;
    m_deltaOfs[5] = m_deltaOfs[1] + sliceSize;
    m_deltaOfs[6] = m_deltaOfs[2] + sliceSize;
    m_deltaOfs[7] = m_deltaOfs[3] + sliceSize;

    // read data file
    if (!dataFileSpecified)
    {
        // change extension to .raw
        strcpy( dataFile, filename );
        int lenWithoutExt = strlen( filename ) - 3;
        dataFile[lenWithoutExt] = 0;
        strcat( dataFile, "raw" );
    }
    std::ifstream data( dataFile, std::ios_base::binary|std::ios_base::in );
    if (!data.is_open()) return false;
    unsigned int numVoxels = m_imgSize[0]*m_imgSize[1]*m_imgSize[2];
    m_imgData = new float[numVoxels];
    data.read( (char*)m_imgData, numVoxels*sizeof(float) );
    if (data.bad()) return false;
    data.close();
    return true;
}

void DiscreteGridField::updateCache( DomainCache *cache, Vec3d& pos )
{
    cache->insideImg = true;
    for (int d=0; d<3; d++) if (pos[d]<m_imgMin[d] || pos[d]>=m_imgMax[d])
        {
            cache->insideImg = false;
            break;
        }
    if (cache->insideImg)
    {
        int voxMinPos[3];
        for (int d=0; d<3; d++)
        {
            voxMinPos[d] = (int)(m_scale[d] * (pos[d]-m_imgMin[d]));
            cache->bbMin[d] = m_spacing[d]*(double)voxMinPos[d] + m_imgMin[d];
            cache->bbMax[d] = cache->bbMin[d] + m_spacing[d];
        }
        unsigned int ofs = voxMinPos[0] + m_imgSize[0]*(voxMinPos[1] + m_imgSize[1]*voxMinPos[2]);
        cache->val[0] = m_imgData[ofs];
        for (int i=1; i<8; i++) cache->val[i] = m_imgData[ofs+m_deltaOfs[i]];
    }
    else
    {
        // init bounding box to be as large as possible to prevent unnecessary cache updates while outside image
        const double MIN=-10e6, MAX=10e6;
        int voxMappedPos[3];
        for (int d=0; d<3; d++)
        {
            if (pos[d] < m_imgMin[d])
            {
                cache->bbMin[d] = MIN;
                cache->bbMax[d] = m_imgMin[d];
                voxMappedPos[d] = 0;
            }
            else if (pos[d] >= m_imgMax[d])
            {
                cache->bbMin[d] = m_imgMax[d];
                cache->bbMax[d] = MAX;
                voxMappedPos[d] = m_imgSize[d]-1;
            }
            else
            {
                cache->bbMin[d] = MIN;
                cache->bbMax[d] = MAX;
                voxMappedPos[d] = (int)(m_scale[d] * (pos[d]-m_imgMin[d]));
            }
        }
        unsigned int ofs = voxMappedPos[0] + m_imgSize[0]*(voxMappedPos[1] + m_imgSize[1]*voxMappedPos[2]);
        // if cache lies outside image, the returned distance is not updated anymore, instead this boundary value is returned
        cache->val[0] = m_imgData[ofs] + m_spacing[0]+m_spacing[1]+m_spacing[2];
    }
}


int DiscreteGridField::getNextDomain()
{
    // while we have free domains always return the next one, afterwards always use the last one
    if (m_usedDomains < (int)m_domainCache.size()) m_usedDomains++;
    return m_usedDomains-1;
}


double DiscreteGridField::getValue( Vec3d &transformedPos, int &domain )
{
    // use translation
    Vec3d pos;
    pos[0] = transformedPos[0] - dx.getValue();
    pos[1] = transformedPos[1] - dy.getValue();
    pos[2] = transformedPos[2] - dz.getValue();
    // find cache domain and check if it needs an update
    DomainCache *cache;
    if (domain < 0)
    {
        domain = getNextDomain();
        cache = &(m_domainCache[domain]);
        updateCache( cache, pos );
    }
    else
    {
        cache = &(m_domainCache[domain]);
        for (int d=0; d<3; d++)
        {
            if (pos[d]<cache->bbMin[d] || pos[d]>cache->bbMax[d])
            {
                updateCache( cache, pos );
                break;
            }
        }
    }

    // if cache lies outside image, the returned distance is not updated anymore, instead this boundary value is returned
    if (!cache->insideImg) return cache->val[0];

    // use trilinear interpolation on cached cube
    double weight[3];
    for (int d=0; d<3; d++)
    {
        weight[d] = m_scale[d] * (pos[d]-cache->bbMin[d]);
    }
    double d = weight[0]*weight[1];
    double c = weight[1] - d;
    double b = weight[0] - d;
    double a = (1.0-weight[1]) - b;
    double res = ( cache->val[0]*a + cache->val[1]*b + cache->val[2]*c + cache->val[3]*d ) * (1.0-weight[2])
            + ( cache->val[4]*a + cache->val[5]*b + cache->val[6]*c + cache->val[7]*d ) * weight[2];

    return res;
}


double DiscreteGridField::getValue( Vec3d &transformedPos )
{
    static int domain=-1;
    return getValue( transformedPos, domain );
}

///factory register
int DiscreteGridFieldClass = RegisterObject("A discrete scalar field from a regular grid storing field value with interpolation.")
        .add< DiscreteGridField >() ;

SOFA_DECL_CLASS(DiscreteGridField)

} ///namespace _discretegrid_

} ///namespace geometry

} ///namespace core

} ///namespace sofa
