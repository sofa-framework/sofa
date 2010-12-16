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
#include <sofa/helper/vector.h>
#include <sofa/helper/OptionsGroup.h>
#include <sofa/helper/map.h>
#include <limits>
#include <sofa/core/objectmodel/DataFileName.h>
#include "CImg.h"

#define DISTANCE_GEODESIC 0
#define DISTANCE_DIFFUSION 1
#define DISTANCE_ANISOTROPICDIFFUSION 2

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
using helper::vector;

template<class TMaterialTypes>
class SOFA_FRAME_API GridMaterial : public Material<TMaterialTypes>
{
public:
    SOFA_CLASS( SOFA_TEMPLATE(GridMaterial, TMaterialTypes), SOFA_TEMPLATE(Material, TMaterialTypes) );

    typedef Material<TMaterialTypes> Inherited;
    typedef typename Inherited::Real Real;        ///< Scalar values.
    typedef vector<Real> VecReal;
    typedef typename Inherited::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::El2Str ElStr;            ///< Elaston strain or stress, see DefaultMaterialTypes
    typedef typename Inherited::VecEl2Str VecElStr;      ///< Vector of elaston strain or stress
    typedef typename Inherited::StrStr StrStr;			///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices
    typedef unsigned char voxelType;

    static const unsigned num_material_dimensions = 3;
    static const unsigned num_spatial_dimensions = 3;

    static const unsigned nbRef = 4;

    typedef Vec<num_material_dimensions,Real> Coord;    ///< Material coordinate: parameters of a point in the object (1 for a wire, 2 for a hull, 3 for a volumetric object)
    typedef vector<Coord> VecCoord;
    typedef Vec<num_material_dimensions,Real> Gradient;    ///< gradient of a scalar value in material space
    typedef vector<Gradient> VecGradient;
    typedef Mat<num_material_dimensions,num_material_dimensions,Real> Hessian;    ///< hessian (second derivative) of a scalar value in material space
    typedef vector<Hessian> VecHessian;

    typedef Vec<num_spatial_dimensions,Real> SCoord;     ///< Coordinate of a point in the space the object is moving in
    typedef vector<Coord> VecSCoord;
    typedef Vec<num_spatial_dimensions,Real> SGradient;    ///< gradient of a scalar value in space
    typedef vector<Gradient> VecSGradient;
    typedef Mat<num_spatial_dimensions,num_spatial_dimensions,Real> SHessian;    ///< hessian (second derivative) of a scalar value in space
    typedef vector<Hessian> VecSHessian;

    typedef Vec<num_spatial_dimensions,int> GCoord;			///< Vector of grid coordinates

    typedef Vec<nbRef,Real> VRefReal;
    typedef Vec<nbRef,SCoord> VRefCoord;
    typedef Vec<nbRef,SGradient> VRefGradient;
    typedef Vec<nbRef,SHessian> VRefHessian;
    typedef Vec<nbRef,unsigned> VRef;

    typedef vector<unsigned int> VUI;

    typedef map<Real,Real> mapLabelType; // voxeltype does not work..

    GridMaterial();
    virtual ~GridMaterial() {}

    /// Recompute the stress-strain matrix when the parameters are changed.
    virtual void init();

    /// implementation of the abstract function
    virtual void computeStress  ( VecStr& stress, VecStrStr* stressStrainMatrices, const VecStr& strain, const VecStr& strainRate );

    virtual void computeStress  ( VecElStr& stress, VecStrStr* stressStrainMatrices, const VecElStr& strain, const VecElStr& strainRate );

    typedef defaulttype::DeformationGradient<3,3,1,Real> DeformationGradient331;
    typedef typename DeformationGradient331::SampleIntegVector SampleIntegVector331;
    typedef vector<SampleIntegVector331>  VecSampleIntegVector331;
    typedef typename DeformationGradient331::Strain            Strain331;
    typedef vector<Strain331>  VecStrain331;

    /** \brief Compute stress based on local strain and strain rate at each point.
    */
    virtual void computeStress  ( VecStrain331& stress, const VecStrain331& strain, const VecStrain331& strainRate, const VecSampleIntegVector331& integ ) {}

    /** \brief Compute stress change based on strain change
     */
    virtual void computeStressChange  ( VecStrain331& stressChange, const VecStrain331& strainChange, const VecSampleIntegVector331& integ ) {}


    typedef defaulttype::DeformationGradient<3,3,2,Real> DeformationGradient332;
    typedef typename DeformationGradient332::SampleIntegVector SampleIntegVector332;
    typedef vector<SampleIntegVector332>  VecSampleIntegVector332;
    typedef typename DeformationGradient332::Strain            Strain332;
    typedef vector<Strain332>  VecStrain332;

    /** \brief Compute stress based on local strain and strain rate at each point.
    */
    virtual void computeStress  ( VecStrain332& stress, const VecStrain332& strain, const VecStrain332& strainRate, const VecSampleIntegVector332& integ ) {}

    /** \brief Compute stress change based on strain change
     */
    virtual void computeStressChange  ( VecStrain332& stressChange, const VecStrain332& strainChange, const VecSampleIntegVector332& integ ) {}






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
    Real getStiffness(const voxelType label);
    // return the linearly interpolated value from the label/density pairs
    Real getDensity(const voxelType label);

    /*************************/
    /*   draw	              */
    /*************************/

    void draw();

    /*************************/
    /*   Lumping			  */
    /*************************/

    bool lumpMoments(const SCoord& point, const unsigned int order, vector<SampleIntegVector331>& moments);

    bool lumpMoments(const SCoord& point, const unsigned int order, vector<SampleIntegVector332>& moments);




    /// return mu_i.vol_i , p_i and weights of voxels in the voronoi region of point
    bool getWeightedMasses(const SCoord& point, vector<VRef>& reps, vector<VRefReal>& w, VecSCoord& p,vector<Real>& masses);
    /// return sum(mu_i.vol_i) in the voronoi region of point
    bool lumpMass(const SCoord& point,Real& mass);
    /// return sum(vol_i) in the voronoi region of point
    bool lumpVolume(const SCoord& point,Real& vol);
    /// return sum((p_i-p)^(order).vol_i) in the voronoi region of point
    bool lumpMoments(const SCoord& point,const unsigned int order,vector<Real>& moments);
    /// return sum(E_i.(p_i-p)^(order).vol_i) in the voronoi region of point
    bool lumpMomentsStiffness(const SCoord& point,const unsigned int order,vector<Real>& moments);
    /// return the repartited weights
    bool lumpWeightsRepartition(const SCoord& point,VRef& reps,VRefReal& w,VRefGradient* dw=NULL,VRefHessian* ddw=NULL);
    /// return the interpolated repartited weights
    bool interpolateWeightsRepartition(const SCoord& point,VRef& reps,VRefReal& w);

    /*********************************/
    /*   Compute distances/weights   */
    /*********************************/

    /// compute voxel weights according to 'distanceType' method -> stored in weightsRepartition and repartition
    bool computeWeights(const VecSCoord& points);
    /// (biased) Uniform sampling (with possibly fixed points stored in points) using Lloyd relaxation
    //  -> returns points and store id/distances in voronoi/distances
    bool computeUniformSampling ( VecSCoord& points, const unsigned int num_points,const unsigned int max_iterations = 100);
    /// Regular sampling based on step size
    //  -> returns points and store id/distances in voronoi/distances
    bool computeRegularSampling ( VecSCoord& points, const unsigned int step);

    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const GridMaterial<TMaterialTypes>* = NULL)
    {
        return TMaterialTypes::Name();
    }

protected:


    /*********************************/
    /*         Grid data   		  */
    /*********************************/
    Data< SCoord > voxelSize;
    Data< SCoord > origin;
    Data< GCoord > dimension;

    CImg<voxelType> grid;
    unsigned int nbVoxels;

    // material properties
    Data<mapLabelType> labelToStiffnessPairs;
    Data<mapLabelType> labelToDensityPairs;
    // temporary values in grid
    vector<Real> distances;
    vector<int> voronoi;
    vector<Real> weights;
    // repartitioned weights
    vector<VRefReal> f_weights;
    vector<VRef> f_index;

    int showedrepartition; // to improve visualization (no need to paste weights on each draw)

    /*********************************/
    /*   IO						  */
    /*********************************/
    sofa::core::objectmodel::DataFileName imageFile;
    std::string infoFile;
    sofa::core::objectmodel::DataFileName weightFile;

    bool loadInfos();
    bool saveInfos();
    bool loadImage();
    bool saveImage();
    bool loadWeightRepartion();
    bool saveWeightRepartion();

    /*********************************/
    /*   Compute distances/weights   */
    /*********************************/

    Data<OptionsGroup> distanceType;  ///< Geodesic, HeatDiffusion, AnisotropicHeatDiffusion
    Data<bool> biasDistances;

    /// (biased) Euclidean distance between two voxels
    Real getDistance(const unsigned int& index1,const unsigned int& index2);
    /// (biased) Geodesical distance between a voxel and all other voxels -> stored in distances
    bool computeGeodesicalDistances ( const SCoord& point, const Real distMax =std::numeric_limits<Real>::max());
    bool computeGeodesicalDistances ( const int& index, const Real distMax =std::numeric_limits<Real>::max());
    /// (biased) Geodesical distance between a set of voxels and all other voxels -> id/distances stored in voronoi/distances
    bool computeGeodesicalDistances ( const VecSCoord& points, const Real distMax =std::numeric_limits<Real>::max());
    bool computeGeodesicalDistances ( const vector<int>& indices, const Real distMax =std::numeric_limits<Real>::max());
    /// (biased) Geodesical distance between the border of the voronoi cell containing point and all other voxels -> stored in distances
    bool computeGeodesicalDistancesToVoronoi ( const SCoord& point, const Real distMax =std::numeric_limits<Real>::max());
    bool computeGeodesicalDistancesToVoronoi ( const int& index, const Real distMax =std::numeric_limits<Real>::max());

    /// linearly decreasing weight with support=factor*dist(point,closestVoronoiBorder) -> weight= 1-d/(factor*(d+-disttovoronoi))
    bool computeAnisotropicLinearWeightsInVoronoi ( const SCoord& point,const Real factor=2.);
    /// linearly decreasing weight with support=factor*distmax_in_voronoi -> weight= factor*(1-d/distmax)
    bool computeLinearWeightsInVoronoi ( const SCoord& point,const Real factor=2.);
    /// Heat diffusion with fixed temperature at points (or regions with same value in grid) -> weights stored in weights
    bool HeatDiffusion( const VecSCoord& points, const unsigned int hotpointindex,const bool fixdatavalue=false,const unsigned int max_iterations=2000,const Real precision=1E-10);

    /// fit 1st, 2d or 3d polynomial to the dense weight map in the dilated by 1 voxel voronoi region (usevoronoi=true) or 26 neighbors (usevoronoi=false) of point.
    bool lumpWeights(const SCoord& point,const bool usevoronoi,Real& w,SGradient* dw=NULL,SHessian* ddw=NULL);
    /// interpolate weights (and weight derivatives) in the dense weight map.
    bool interpolateWeights(const SCoord& point,Real& w);

    /*********************************/
    /*         Utils				  */
    /*********************************/

    inline int getIndex(const GCoord& icoord);
    inline int getIndex(const SCoord& coord);
    inline bool getiCoord(const SCoord& coord, GCoord& icoord);
    inline bool getiCoord(const int& index, GCoord& icoord);
    inline bool getCoord(const GCoord& icoord, SCoord& coord) ;
    inline bool getCoord(const int& index, SCoord& coord) ;
    inline bool get6Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get18Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool get26Neighbors ( const int& index, VUI& neighbors ) ;
    inline bool findIndexInRepartition(unsigned int& realIndex, const unsigned int& pointIndex, const unsigned int& frameIndex);

    inline void accumulateCovariance(const SCoord& p,const unsigned int order,vector<vector<Real> >& Cov);
    inline void getCompleteBasis(const SCoord& p,const unsigned int order,vector<Real>& basis);
    inline void getCompleteBasisDeriv(const SCoord& p,const unsigned int order,vector<SGradient>& basisDeriv);
    inline void getCompleteBasisDeriv2(const SCoord& p,const unsigned int order,vector<SHessian>& basisDeriv);
    inline void addWeightinRepartion(const unsigned int index); // add dense weights relative to index, in weight repartion of size nbref if it is large enough
    inline void pasteRepartioninWeight(const unsigned int index); // paste weight relative to index in the dense weight map
    inline void normalizeWeightRepartion();

    /*********************************/
    /*   draw						  */
    /*********************************/

    Data<OptionsGroup> showVoxels;    ///< None, Grid Values, Voronoi regions, Distances, Weights
    Data<unsigned int> showWeightIndex;    ///
    GLuint cubeList;             // storage for the display list
    void genListCube();
    void drawCube(const double& x, const double& y, const double& z);
    Data<GCoord> showPlane;    /// indices of the slices to show (if <0 or >=nbslices, no plane shown in the given direction)

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
