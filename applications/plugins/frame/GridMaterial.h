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
#ifndef SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_H
#define SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_H

#include "initFrame.h"
#include "NewMaterial.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/SVector.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/map.h>
#include <sofa/component/container/VoxelGridLoader.h>
#include "CImg.h"

#define DISTANCE_GEODESIC 0
#define DISTANCE_BIASEDGEODESIC 1
#define DISTANCE_DIFFUSION 2
#define DISTANCE_ANISOTROPICDIFFUSION 3

#define SHOWVOXELS_NONE 0
#define SHOWVOXELS_DATAVALUE 1
#define SHOWVOXELS_STIFFNESS 2
#define SHOWVOXELS_DENSITY 3
#define SHOWVOXELS_VORONOI 4
#define SHOWVOXELS_DISTANCES 5
#define SHOWVOXELS_WEIGHTS 6

namespace sofa
{

namespace component
{

namespace material
{

using namespace cimg_library;
using namespace sofa::defaulttype;
using namespace helper;
using std::map;

template<class TMaterialTypes, typename voxelType>
class SOFA_FRAME_API GridMaterial : public Material<TMaterialTypes>
{
public:
    SOFA_CLASS( SOFA_TEMPLATE2(GridMaterial, TMaterialTypes,voxelType), SOFA_TEMPLATE(Material, TMaterialTypes) );

    typedef Material<TMaterialTypes> Inherited;
    typedef typename Inherited::Real Real;        ///< Scalar values.
    typedef typename Inherited::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::El2Str ElStr;            ///< Elaston strain or stress, see DefaultMaterialTypes
    typedef typename Inherited::VecEl2Str VecElStr;      ///< Vector of elaston strain or stress
    typedef typename Inherited::StrStr StrStr;			///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices

    typedef Vec<3,Real> Vec3;			///< Material coordinate
    typedef vector<Vec3> VecVec3;							///< Vector of material coordinates
    typedef Mat<3,3,Real> Mat33;
    typedef Vec<3,int> Vec3i;							    ///< Vector of grid coordinates
    typedef SVector<double> VD;
    typedef SVector<SVector<double> > VVD;
    typedef SVector<SVector<SVector<double> > > VVVD;
    typedef SVector<unsigned int> VUI;
    typedef SVector<SVector<unsigned int> > VVUI;
    typedef SVector<int> VI;
    typedef SVector<SVector<int> > VVI;
    typedef SVector<bool> VB;
    typedef map<double,double> mapLabelType;

    Data<OptionsGroup> distanceType;  ///< Geodesic, BiasedGeodesic, HeatDiffusion, AnisotropicHeatDiffusion
    Data<OptionsGroup> showVoxels;    ///< None, Grid Values, Voronoi regions, Distances, Weights
    Data<unsigned int> showWeightIndex;    ///

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
    //  Real computeUniformSampling( VecVec3& points, vector<Real>& point_data, unsigned num_points, unsigned order );

//    /// implementation of the abstract function
//    virtual void computeDStress ( VecStr& stressChange, const VecStr& strainChange );


    /*************************/
    /* material properties	  */
    /*************************/

    // return the linearly interpolated value from the label/stiffness pairs
    double getStiffness(const voxelType label);
    // return the linearly interpolated value from the label/density pairs
    double getDensity(const voxelType label);

    /*************************/
    /*   draw	              */
    /*************************/

    void draw();
    void drawCube(double size,bool wireframe);

    /*************************/
    /*   IO	              */
    /*************************/

    bool loadInfos();
    bool saveInfos();
    bool loadImage();
    bool saveImage();
    bool loadWeightRepartion();
    bool saveWeightRepartion();

    /*************************/
    /*   Lumping			  */
    /*************************/

    /// return sum(mu_i.vol_i) in the voronoi region of point
    bool LumpMass(const Vec3& point,double& mass);
    /// return sum(vol_i) in the voronoi region of point
    bool LumpVolume(const Vec3& point,double& vol);
    /// return sum((p_i-p)^(order).vol_i) in the voronoi region of point
    bool LumpMoments(const Vec3& point,const unsigned int order,VD& moments);
    /// return sum(E_i.(p_i-p)^(order).vol_i) in the voronoi region of point
    bool LumpMomentsStiffness(const Vec3& point,const unsigned int order,VD& moments);
    /// fit 1st, 2d or 3d polynomial to the weights in the (dilated by 1 voxel) voronoi region of point.
    bool LumpWeights(const Vec3& point,const bool dilatevoronoi,double& w,Vec3* dw=NULL,Mat33* ddw=NULL);

    /*********************************/
    /*   Compute distances/weights   */
    /*********************************/

    /// compute voxel weights according to 'distanceType' method -> stored in weightsRepartition and repartition
    bool computeWeights(const unsigned int nbrefs,const VecVec3& points);

    /// (biased) Euclidean distance between two voxels
    double getDistance(const unsigned int& index1,const unsigned int& index2);
    /// (biased) Geodesical distance between a voxel and all other voxels -> stored in distances
    bool computeGeodesicalDistances ( const Vec3& point, const double distMax =1E100);
    bool computeGeodesicalDistances ( const int& index, const double distMax =1E100);
    /// (biased) Geodesical distance between a set of voxels and all other voxels -> id/distances stored in voronoi/distances
    bool computeGeodesicalDistances ( const VecVec3& points, const double distMax =1E100);
    bool computeGeodesicalDistances ( const VI& indices, const double distMax =1E100);
    /// (biased) Uniform sampling (with possibly fixed points stored in points) using Lloyd relaxation -> id/distances stored in voronoi/distances
    bool computeUniformSampling ( VecVec3& points, const unsigned int num_points,const unsigned int max_iterations = 100);
    /// linearly decreasing weight with support=factor*distmax_in_voronoi
    bool computeLinearWeightsInVoronoi ( const Vec3& point,const double factor=2.);
    /// Heat diffusion with fixed temperature at points (or regions with same value in grid) -> weights stored in weights
    bool HeatDiffusion( const VecVec3& points, const unsigned int hotpointindex,const bool fixdatavalue=false,const unsigned int max_iterations=1000,const double precision=0.0001);


    /*************************/
    /*         Utils         */
    /*************************/

    inline int getIndex(const Vec3i& icoord);
    inline int getIndex(const Vec3& coord);
    inline bool getiCoord(const Vec3& coord, Vec3i& icoord);
    inline bool getiCoord(const int& index, Vec3i& icoord);
    inline bool getCoord(const Vec3i& icoord, Vec3& coord) ;
    inline bool getCoord(const int& index, Vec3& coord) ;
    inline bool get6Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get18Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get26Neighbors ( const int& index, VUI& neighbors ) ;

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const GridMaterial<TMaterialTypes,voxelType>* = NULL)
    {
        std::string name;
        name.append(TMaterialTypes::Name());
        name.append(CImg<voxelType>::pixel_type());
        return name;
    }

protected:
    // Grid data
    sofa::core::objectmodel::DataFileName imageFile;
    sofa::core::objectmodel::DataFileName infoFile;
    Data< Vec3d > voxelSize;
    Data< Vec3d > origin;
    Data< Vec3i > dimension;

    CImg<voxelType> grid;
    unsigned int nbVoxels;

    // material properties
    Data<mapLabelType> labelToStiffnessPairs;
    Data<mapLabelType> labelToDensityPairs;

    // temporary values in grid
    VD distances;
    VI voronoi;
    VD weights;

    // repartitioned weights
    sofa::core::objectmodel::DataFileName weightFile;
    VVD weightsRepartition;
    VVUI repartition;
    int showedrepartition; // to improve visualization (no need to paste weights on each draw)

    // local functions
    inline void accumulateCovariance(const Vec3& p,const unsigned int order,VVD& Cov);
    inline void getCompleteBasis(const Vec3& p,const unsigned int order,VD& basis);
    inline void getCompleteBasisDeriv(const Vec3& p,const unsigned int order,VVD& basisDeriv);
    inline void getCompleteBasisDeriv2(const Vec3& p,const unsigned int order,VVVD& basisDeriv);
    inline void addWeightinRepartion(const unsigned int index); // add dense weights relative to index, in weight repartion of size nbref if it is large enough
    inline void pasteRepartioninWeight(const unsigned int index); // paste weight relative to index in the dense weight map
    inline void normalizeWeightRepartion();


};

//#ifdef SOFA_FLOAT
//template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMateriald"; }
//template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterial"; }
//#else
//template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMaterial"; }
//template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterialf"; }
//#endif


} // namespace material

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_H
