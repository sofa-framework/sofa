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
#ifndef SOFAIMPLICITFIELD_COMPONENT_DISCRETEGRIDFIELD_H
#define SOFAIMPLICITFIELD_COMPONENT_DISCRETEGRIDFIELD_H
#include <SofaImplicitField/config.h>

#include <sofa/core/objectmodel/DataFileName.h>
#include <SofaImplicitField/components/geometry/ScalarField.h>

namespace sofa
{

namespace component
{

namespace geometry
{

namespace _discretegrid_
{

using sofa::defaulttype::Vec3d;

class  SOFA_SOFAIMPLICITFIELD_API DomainCache
{
public:
    bool insideImg; // shows if the domain lies inside the valid image region or outside
    Vec3d bbMin, bbMax; // bounding box (min and max) of the domain
    double val[8]; // corner values of the domain
};

class SOFA_SOFAIMPLICITFIELD_API DiscreteGridField : public virtual ScalarField
{

public:
    SOFA_CLASS(DiscreteGridField, ScalarField);

public:
    DiscreteGridField();
    ~DiscreteGridField();

    virtual void init() override;

    virtual double getValue( Vec3d &transformedPos );
    virtual double getValue( Vec3d &transformedPos, int &domain ) override;
    virtual int getDomain( Vec3d &pos, int ref_domain ) override { (void)pos; return ref_domain; }

    void setFilename(const std::string& filename) ;
    bool loadGridFromMHD( const char *filename ) ;

    void updateCache( DomainCache *cache, Vec3d& pos );
    int getNextDomain();

    sofa::core::objectmodel::DataFileName d_distanceMapHeader;
    Data< int > d_maxDomains; ///< Number of domains available for caching
    Data< double > dx;    ///< translation of original image
    Data< double > dy;    ///< translation of original image
    Data< double > dz;    ///< translation of original image

    int m_usedDomains;              // number of domains already given out
    unsigned int m_imgSize[3];      // number of voxels
    double m_spacing[3];            // physical distance between two neighboring voxels
    double m_scale[3];              // (1/spacing)
    double m_imgMin[3], m_imgMax[3];  // physical locations of the centers of both corner voxels
    float *m_imgData;               // raw data
    unsigned int m_deltaOfs[8];     // offsets to define 8 corners of cube for interpolation
    std::vector<DomainCache> m_domainCache;
};

} /// namespace _discretegrid_
using _discretegrid_::DiscreteGridField ;

} /// namespace geometry

} /// namespace component

} /// namespace sofa

#endif

