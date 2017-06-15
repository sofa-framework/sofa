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
#include <../extlibs/CImg/CImg.h>

#include <sofa/helper/system/gl.h>

#define DISTANCE_GEODESIC 0
#define DISTANCE_DIFFUSION 1
#define DISTANCE_ANISOTROPICDIFFUSION 2

#define SHOWVOXELS_NONE 0
#define SHOWVOXELS_DATAVALUE 1
#define SHOWVOXELS_STIFFNESS 2
#define SHOWVOXELS_DENSITY 3
#define SHOWVOXELS_BULKMODULUS 4
#define SHOWVOXELS_POISSONRATIO 5
#define SHOWVOXELS_VORONOI 6
#define SHOWVOXELS_VORONOI_FR 7
#define SHOWVOXELS_DISTANCES 8
#define SHOWVOXELS_WEIGHTS 9
#define SHOWVOXELS_LINEARITYERROR 10
#define SHOWVOXELS_FRAMESINDICES 11

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

/** Material distribution defined using a volumetric image.
  */
template<class TMaterialTypes>
class SOFA_FRAME_API GridMaterial : public Material<TMaterialTypes>
{
public:
    SOFA_CLASS( SOFA_TEMPLATE(GridMaterial, TMaterialTypes), SOFA_TEMPLATE(Material, TMaterialTypes) );

    typedef Material<TMaterialTypes> Inherited;


    typedef typename Inherited::Real Real;        ///< Scalar values.
    typedef typename Inherited::MaterialCoord MaterialCoord;
    typedef typename Inherited::VecMaterialCoord VecMaterialCoord;
    typedef typename Inherited::Str Str;            ///< Strain or stress tensor defined as a vector with 6 entries for 3d material coordinates, 3 entries for 2d coordinates, and 1 entry for 1d coordinates.
    typedef typename Inherited::VecStr VecStr;      ///< Vector of strain or stress tensors
    typedef typename Inherited::StrStr StrStr;			///< Stress-strain matrix
    typedef typename Inherited::VecStrStr VecStrStr;      ///< Vector of Stress-strain matrices
    typedef typename Inherited::Strain1 Strain1;
    typedef typename Inherited::VecStrain1 VecStrain1;
    typedef typename Inherited::Strain4 Strain4;
    typedef typename Inherited::VecStrain4 VecStrain4;
    typedef typename Inherited::Strain10 Strain10;
    typedef typename Inherited::VecStrain10 VecStrain10;


    /** @name Dimensions
      Constant for efficiency. These could be template parameters. */
    //@{
    static const unsigned int num_material_dimensions = 3;
    static const unsigned int num_spatial_dimensions = 3;
    //@}

    /** @name Material space
      Types related to material coordinates. */
    //@{
    typedef Vec<num_material_dimensions,Real> Coord;    ///< Material coordinate: parameters of a point in the object (1 for a wire, 2 for a hull, 3 for a volumetric object)
    typedef vector<Coord> VecCoord;
    typedef Vec<num_material_dimensions,Real> Gradient;    ///< gradient of a scalar value in material space
    typedef vector<Gradient> VecGradient;
    typedef Mat<num_material_dimensions,num_material_dimensions,Real> Hessian;    ///< hessian (second derivative) of a scalar value in material space
    typedef vector<Hessian> VecHessian;
    //@}

    /** @name World space
      Types related to spatial coordinates. */
    //@{
    typedef Vec<num_spatial_dimensions,Real> SCoord;     ///< Coordinate of a point in the space the object is moving in
    typedef vector<Coord> VecSCoord;
    typedef Vec<num_spatial_dimensions,Real> SGradient;    ///< gradient of a scalar value in space
    typedef vector<Gradient> VecSGradient;
    typedef Mat<num_spatial_dimensions,num_spatial_dimensions,Real> SHessian;    ///< hessian (second derivative) of a scalar value in space
    typedef vector<Hessian> VecSHessian;
    //@}


    /** @name Image types
      Types related to the volumetric image used to represent the material. */
    //@{
    typedef unsigned char voxelType;
    typedef Vec<num_spatial_dimensions,int> GCoord;			///< Vector of grid coordinates
    //@}


    /** @name Sample types
      Types related to the sampling points. */
    //@{
    static const unsigned int nbRef = 4; ///< Number of frames influencing each point.
    typedef Vec<nbRef,Real> VRefReal;
    typedef Vec<nbRef,SCoord> VRefCoord;
    typedef Vec<nbRef,SGradient> VRefGradient;
    typedef Vec<nbRef,SHessian> VRefHessian;
    typedef Vec<nbRef,unsigned int> VRef;
    //@}


    GridMaterial();
    virtual ~GridMaterial();

    virtual void init();
    /// Recompute the stress-strain matrix when the parameters are changed.
    virtual void reinit();


    /** @name Compute Stress
      Compute stress or stress change at each point based on local strain and strain rate. */
    //@{
    virtual void computeStress  ( VecStrain1& stress, VecStrStr* stressStrainMatrices, const VecStrain1& strain, const VecStrain1& strainRate, const VecMaterialCoord& point );
    virtual void computeStress  ( VecStrain4& stress, VecStrStr* stressStrainMatrices, const VecStrain4& strain, const VecStrain4& strainRate, const VecMaterialCoord& point );
    virtual void computeStress  ( VecStrain10& stress, VecStrStr* stressStrainMatrices, const VecStrain10& strain, const VecStrain10& strainRate, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain1& stressChange, const VecStrain1& strainChange, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain4& stressChange, const VecStrain4& strainChange, const VecMaterialCoord& point );
    virtual void computeStressChange  ( VecStrain10& stressChange, const VecStrain10& strainChange, const VecMaterialCoord& point );
    //@}

    /// get the StressStrain matrices at the given points, assuming null strain or linear material
    virtual void getStressStrainMatrix( StrStr& matrix, const MaterialCoord& point ) const;

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

    /*************************/
    /*   draw	              */
    /*************************/

    void draw(const core::visual::VisualParams* vparams);


    /*************************/
    /* material properties	  */
    /*************************/

    void updateSampleMaterialProperties();
    virtual Real getBulkModulus(const unsigned int sampleindex) const;


    /*************************/
    /*   Lumping			  */
    /*************************/

    /// return mu_i.vol_i , p_i and weights of voxels in the voronoi region labeled 'sampleindex'
    bool getWeightedMasses(const unsigned int sampleindex, vector<VRef>& reps, vector<VRefReal>& w, VecSCoord& p,vector<Real>& masses);
    /// return sum(mu_i.vol_i) in the voronoi region labeled 'sampleindex'
    bool lumpMass(const unsigned int sampleindex,Real& mass);
    /// return sum(vol_i) in the voronoi region labeled 'sampleindex'
    bool lumpVolume(const unsigned int sampleindex,Real& vol);
    /// return sum(Stiffness_i.(p_i-p)^(order).vol_i) in the voronoi region labeled 'sampleindex' centered on point
    virtual bool computeVolumeIntegrationFactors(const unsigned int sampleindex,const SCoord& point,const unsigned int order,vector<Real>& moments);
    /// return the weights of the voronoi region labeled 'sampleindex'
    bool lumpWeightsRepartition(const unsigned int sampleindex,const SCoord& point,VRef& reps,VRefReal& w,VRefGradient* dw=NULL,VRefHessian* ddw=NULL);
    /// return the interpolated weights
    bool interpolateWeightsRepartition(const SCoord& point,VRef& reps,VRefReal& w,const int restrictToLabel=-1);

    /*********************************/
    /*   Compute distances/weights   */
    /*********************************/

    /// compute voxel weights related to 'points' according to 'distanceType' method -> stored in weightsRepartition and repartition
    bool computeWeights(const VecSCoord& points);
    // insert a point in each rigid part where stifness>=STIFFNESS_RIGID
    bool isRigid(const voxelType label) const;
    bool rigidPartsSampling ( VecSCoord& points);
    /// (biased) Uniform sampling (with possibly fixed points stored in points) using Lloyd relaxation
    //  -> returns points and store id/distances in voronoi/distances
    Data<unsigned int> maxLloydIterations;
    bool computeUniformSampling ( VecSCoord& points, const unsigned int num_points);
    /// Regular sampling based on step size
    //  -> returns points and store id/distances in voronoi/distances
    bool computeRegularSampling ( VecSCoord& points, const unsigned int step);
    /// Identify regions where weights are linear up to the tolerance
    //  -> returns points and store id/distances in voronoi/distances
    bool computeLinearRegionsSampling ( VecSCoord& points, const unsigned int num_points);
    /// (biased) Geodesical distance between a set of voxels and all other voxels -> id/distances stored in voronoi/distances
    bool computeGeodesicalDistances ( const VecSCoord& points, const Real distMax =std::numeric_limits<Real>::max());


    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }

    static std::string templateName(const GridMaterial<TMaterialTypes>* = NULL)
    {
        return TMaterialTypes::Name();
    }

protected:

    typedef Vec<3,unsigned int> Vec3i;
    typedef vector<unsigned int> VUI;
    typedef map<Real,Real> mapLabelType; // voxeltype does not work..

    /*********************************/
    /*           Grid data           */
    /*********************************/
public:
    Data< SCoord > voxelSize;
    Data< SCoord > origin;
    Data< GCoord > dimension;

    CImg<voxelType> grid;
    unsigned int gridOffset; // Use to enlarge weight interpolation
    unsigned int nbVoxels;

    CImg<voxelType>& getGrid() {return grid;}

protected:
    // material properties
    Data<mapLabelType> labelToStiffnessPairs;
    Data<mapLabelType> labelToDensityPairs;
    Data<mapLabelType> labelToBulkModulusPairs;
    Data<mapLabelType> labelToPoissonRatioPairs;

    // temporary grid data
    //vector<vector<Real> > all_distances; // will soon replace distances (contains all the distances)
    vector<Real> distances;
    vector<int> voronoi;
    vector<int> voronoi_frames;
    vector<Real> weights;
    vector<Real> linearityError;

    // voxel data
    vector<VRefReal> v_weights;
    vector<VRef> v_index;
public:
    inline void getWeights( VRefReal& weights, const unsigned int& index) { weights = v_weights[index];};
    inline void getIndices( VRef& indices, const unsigned int& index) { indices = v_index[index];};
    inline Real getDistance( const unsigned int& index) {return distances[index];};
    inline double getVolumeForVoronoi( const unsigned int index) const { unsigned int nbV = 0; for (unsigned int i = 0; i < nbVoxels; ++i) if (voronoi[i] == (int)index) nbV++; return nbV * voxelSize.getValue()[0] * voxelSize.getValue()[1] * voxelSize.getValue()[2];};

protected:
    int showedrepartition; // to improve visualization (no need to paste weights on each draw)
    int showederror; // to improve visualization (no need to recompute error on each draw)

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

    /*************************/
    /* material properties	  */
    /*************************/

    // store sample material properties to speed up access during simulation
    Data<vector<Real> > bulkModulus;
    Data<vector<Real> > stiffness;
    Data<vector<Real> > density;
    Data<vector<Real> > poissonRatio;

public:
    // return the linearly interpolated value from the label/stiffness pairs
    Real getStiffness(const voxelType label) const;
    // return the linearly interpolated value from the label/density pairs
    Real getDensity(const voxelType label) const;
    // return the linearly interpolated value from the label/bulkModulus pairs
    Real getBulkModulus(const voxelType label) const;
    // return the linearly interpolated value from the label/PoissonRatio pairs
    Real getPoissonRatio(const voxelType label) const;

    /*********************************/
    /*   Compute distances/weights   */
    /*********************************/

protected:
    Data<OptionsGroup> distanceType;  ///< Geodesic, HeatDiffusion, AnisotropicHeatDiffusion
    Data<bool> biasDistances;
    Data<Real> distanceBiasFactor; Real biasFactor;
    Data<bool> useDijkstra;

    /// diffuse the weights outside the objects to avoid interpolation problems
    void offsetWeightsOutsideObject(unsigned int offestdist=2);

    /// (biased) Euclidean distance between two voxels
    Real getDistance(const unsigned int& index1,const unsigned int& index2,const int fromLabel=-1);
    /// (biased) Geodesical distance between a voxel and all other voxels -> stored in distances
    bool computeGeodesicalDistances ( const SCoord& point, const Real distMax =std::numeric_limits<Real>::max(),const vector<Real>* distanceScaleFactors=NULL);
    bool computeGeodesicalDistances ( const int& index, const Real distMax =std::numeric_limits<Real>::max(),const vector<Real>* distanceScaleFactors=NULL);
    /// (biased) Geodesical distance between a set of voxels and all other voxels -> id/distances stored in voronoi/distances
    bool computeGeodesicalDistances ( const vector<int>& indices, const Real distMax =std::numeric_limits<Real>::max());


    // compute voronoi of voronoi "nbVoronoiSubdivisions" times
    // input : a region initialized at 1 in "voronoi"
    //       : distances used to compute the voronoi
    //       : nbVoronoiSubdivisions data
    // returns  : linearly decreasing weights from 0 to 1
    //			: the distance from the region center in "distances"
    bool computeVoronoiRecursive ( const unsigned int fromLabel );

    // compute distance from voronoi
    // input : a region initialized at 1 in "voronoi"
    // return : the distance from the border
    bool computeGeodesicalDistancesToVoronoi ( const unsigned int fromLabel,const Real distMax =std::numeric_limits<Real>::max(), VUI* ancestors =NULL);


    // subdivide a voronoi region in two subregions using lloyd relaxation and euclidean distances
    bool SubdivideVoronoiRegion( const unsigned int voronoiindex, const unsigned int newvoronoiindex, const unsigned int max_iterations =100);

    // scale the distance according to frame-to-voronoi border paths
    Data<bool> useDistanceScaleFactor;
    Data<Real> weightSupport;  ///< support of the weight function (2=interpolating, >2 = approximating)
    Data<unsigned int> nbVoronoiSubdivisions;  ///< number of subdvisions of the voronoi during weight computation (1 by default)

    /// linearly decreasing weight with support=factor*dist(point,closestVoronoiBorder) -> weight= 1-d/(factor*(d+-disttovoronoi))
    bool computeAnisotropicLinearWeightsInVoronoi ( const SCoord& point);
    /// linearly decreasing weight with support=factor*distmax_in_voronoi -> weight= factor*(1-d/distmax)
    bool computeLinearWeightsInVoronoi ( const SCoord& point);
    /// Heat diffusion with fixed temperature at points (or regions with same value in grid) -> weights stored in weights
    bool HeatDiffusion( const VecSCoord& points, const unsigned int hotpointindex,const bool fixdatavalue=false,const unsigned int max_iterations=2000,const Real precision=1E-10);

    /// fit 1st, 2d or 3d polynomial to the dense weight map in the region defined by indices
    bool lumpWeights(const VUI& indices,const SCoord& point,Real& w,SGradient* dw=NULL,SHessian* ddw=NULL,Real* err=NULL);

    // compute and store in the map "linearity error" the approximation error of each voxel weight according the fit in its voronoi region
    bool updateLinearityError(const bool allweights);

    /*********************************/
    /*         Utils				  */
    /*********************************/

public:
    Data<bool> verbose;

    inline int getIndex(const GCoord& icoord) const;
    inline int getIndex(const SCoord& coord) const;
    inline bool getiCoord(const SCoord& coord, GCoord& icoord) const;
    inline bool getiCoord(const int& index, GCoord& icoord) const;
    inline bool getCoord(const GCoord& icoord, SCoord& coord) const;
    inline bool getCoord(const int& index, SCoord& coord) const;
    inline bool get6Neighbors ( const int& index, VUI& neighbors ) const;
    inline bool get18Neighbors ( const int& index, VUI& neighbors ) const;
    inline bool get26Neighbors ( const int& index, VUI& neighbors ) const;
protected:
    inline Real findWeightInRepartition(const unsigned int& pointIndex, const unsigned int& frameIndex);
    inline bool areRepsSimilar(const unsigned int i1,const unsigned int i2);

    inline void accumulateCovariance(const SCoord& p,const unsigned int order,vector<vector<Real> >& Cov);
    inline void getCompleteBasis(const SCoord& p,const unsigned int order,vector<Real>& basis) const;
    inline void getCompleteBasisDeriv(const SCoord& p,const unsigned int order,vector<SGradient>& basisDeriv) const;
    inline void getCompleteBasisDeriv2(const SCoord& p,const unsigned int order,vector<SHessian>& basisDeriv) const;
    inline void addWeightinRepartion(const unsigned int index); // add dense weights relative to index, in weight repartion of size nbref if it is large enough
    inline bool pasteRepartioninWeight(const unsigned int index); // paste weight relative to index in the dense weight map
    inline void normalizeWeightRepartion();

    /*********************************/
    /*         Adaptivity                 */
    /*********************************/

public:
    Data<bool> voxelsHaveChanged;

    void removeVoxels( const vector<unsigned int>& voxelsToRemove);
    void localyUpdateWeights( const vector<unsigned int>& removedVoxels);

    /*********************************/
    /*   draw                        */
    /*********************************/

protected:
    Data<OptionsGroup> showVoxels;    ///< None, Grid Values, Voronoi regions, Distances, Weights
    Data<unsigned int> showWeightIndex;    ///
    GLuint cubeList; GLuint wcubeList;            // storage for the display list
    Data<GCoord> showPlane;    /// indices of the slices to show (if <0 or >=nbslices, no plane shown in the given direction)
    bool showWireframe;
    float maxValues[12];
    Data<float> show3DValuesHeight;
    Data<bool> vboSupported;
    GLuint vboValuesId1; // ID of VBO for 3DValues vertex arrays (to store vertex coords and normals)
    GLuint vboValuesId2; // ID of VBO for 3DValues index array

public:
    float getLabel (const int&x, const int& y, const int& z) const;
    void getColor( float* color, const float& label) const;

protected:
    void genListCube ();
    void drawCube (const double& x, const double& y, const double& z) const;
    GLuint createVBO (const void* data, int dataSize, GLenum target, GLenum usage);
    void deleteVBO (const GLuint vboId);
    void initVBO ();
    void initPlaneGeometry (GLfloat* valuesVertices, GLfloat* valuesNormals, GLfloat* valuesColors, GLushort* valuesIndices, const int& axis, const int nbVerticesOffset, const int nbIndicesOffset);
    void updateValuesVBO () const;
    void displayValuesVBO () const;
    void drawPlaneBBox (const int& axis) const;
    void updateMaxValues ();
    void displayValues() const;
    void displayPlane( const int& axis) const;
};

//#ifdef SOFA_FLOAT
//template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMateriald"; }
//template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterial"; }
//#else
//template<> inline const char* GridMaterial<Material3d>::Name() { return "GridMaterial"; }
//template<> inline const char* GridMaterial<Material3f>::Name() { return "GridMaterialf"; }
//#endif


#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_FRAME_API GridMaterial<Material3d>;
#endif //SOFA_FLOAT
#ifndef SOFA_DOUBLE
extern template class SOFA_FRAME_API GridMaterial<Material3f>;
#endif //SOFA_DOUBLE
#endif //defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_CPP)


} // namespace material

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_H
