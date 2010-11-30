/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_MATERIAL_GridMaterial_H
#define SOFA_COMPONENT_MATERIAL_GridMaterial_H

#include "initFrame.h"
#include "NewMaterial.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/SVector.h>
#include <sofa/component/container/VoxelGridLoader.h>


namespace sofa
{
namespace component
{
namespace material
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using container::VoxelGridLoader;

template<class TMaterialTypes>
class SOFA_FRAME_API GridMaterial : public Material<TMaterialTypes>
{
public:
    typedef Material<TMaterialTypes> Inherited;
    SOFA_CLASS( SOFA_TEMPLATE(GridMaterial, TMaterialTypes), SOFA_TEMPLATE(Material, TMaterialTypes) );

    typedef typename Inherited::Real Real;        ///< Scalar values.
    typedef typename defaulttype::Vec<3,Real> Vec3;        ///< Material coordinate
    typedef helper::vector<Vec3> VecVec3;        ///< Vector of material coordinates
    typedef typename Inherited::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::El2Str ElStr;            ///< Elaston strain or stress, see DefaultMaterialTypes
    typedef typename Inherited::VecEl2Str VecElStr;      ///< Vector of elaston strain or stress
    typedef typename Inherited::StrStr StrStr;      ///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices

    typedef Vec<3,int> Vec3i;
    typedef sofa::helper::SVector<double> VD;
    typedef sofa::helper::SVector<unsigned int> VUI;
    typedef sofa::helper::SVector<int> VI;
    typedef sofa::helper::SVector<bool> VB;

    GridMaterial();
    virtual ~GridMaterial() {}

    /// Recompute the stress-strain matrix when the parameters are changed.
    virtual void init();

    /// implementation of the abstract function
    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate );

    virtual void computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& strainRate );

    /** Compute uniformly distributed point positions using Lloyd relaxation.
    Parameter points is InOut. The points passed to the method are considered fixed.
    The method creates points.size()-num_points points, moves them to get an as-uniform-as-possible sampling, and appends them to the vector.
    It additionally computes data associated with each point, such as mass.
    The additional data depends on the order of the elaston associated with each point:

    - 0 (traditional Gauss point): mass, volume
    - 1 (first-order elaston): mass, volume, TODO: complete this
    - 2 (second-order elaston): mass, volume, TODO: complete this

    A first-order elaston models the strain as a linear function using the strain at the point and the gradient of the strain at this point, while a second-order elaston models the strain, the strain gradient and the strain Hessian.

      */
    //  Real computeUniformSampling( VecVec3& points, helper::vector<Real>& point_data, unsigned num_points, unsigned order );

//    /// implementation of the abstract function
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange );


    /*   Compute distances   */
    // (biased) Euclidean distance between two voxels
    double getDistance(const unsigned int& index1,const unsigned int& index2,const bool biasDistances);
    // (biased) Geodesical distance between a voxel and all other voxels -> stored in distances
    bool computeGeodesicalDistances ( const Vec3& point, const bool biasDistances, const double distMax =1E100);
    bool computeGeodesicalDistances ( const int& index, const bool biasDistances, const double distMax =1E100);
    // (biased) Geodesical distance between a set of voxels and all other voxels -> id/distances stored in voronoi/distances
    bool computeGeodesicalDistances ( const VecVec3& points, const bool biasDistances, const double distMax =1E100);
    bool computeGeodesicalDistances ( const VI& indices, const bool biasDistances, const double distMax =1E100);

    bool computeUniformSampling ( VecVec3& points, const bool biasDistances, unsigned int num_points, unsigned int max_iterations );


    /*         Utils         */
    inline int getIndex(const Vec3i& icoord);
    inline int getIndex(const Vec3& coord);
    inline bool getiCoord(const Vec3& coord, Vec3i& icoord);
    inline bool getiCoord(const unsigned int& index, Vec3i& icoord);
    inline bool getCoord(const Vec3i& icoord, Vec3& coord) ;
    inline bool getCoord(const unsigned int& index, Vec3& coord) ;
    inline bool get6Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get18Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get26Neighbors ( const int& index, VUI& neighbors ) ;


    static const char* Name();

    std::string getTemplateName() const
    {
        return templateName(this);
    }
    static std::string templateName(const GridMaterial<TMaterialTypes>* = NULL)
    {
        return TMaterialTypes::Name();
    }

protected:
    VoxelGridLoader* voxelGridLoader;

    // Grid parameters (initialized from voxelGridLoader)
    Vec3d voxelSize;
    Vec3d Origin;
    Vec3i Dimension;
    unsigned int nbVoxels;
    const unsigned char *Data;
    const unsigned char *SegmentID;

    // temporary values in grid
    VD Distances;
    VI Voronoi;


};

#ifdef SOFA_FLOAT
template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMateriald"; }
template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterial"; }
#else
template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMaterial"; }
template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterialf"; }
#endif


}

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FEM_BASEMATERIAL_H
