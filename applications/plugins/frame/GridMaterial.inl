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
#ifndef SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_INL
#define SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_INL

#include "GridMaterial.h"
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <queue>
#include <string>
//#include <omp.h>

namespace sofa
{

namespace component
{

namespace material
{

template<class MaterialTypes>
GridMaterial< MaterialTypes>::GridMaterial()
    : Inherited()
    , distanceType ( initData ( &distanceType,"distanceType","Distance measure." ) )
    , imageFile( initData(&imageFile,"imageFile","Image file."))
    , weightFile( initData(&weightFile,"weightFile","Voxel weight file."))
    , voxelSize ( initData ( &voxelSize, Vec3d ( 0,0,0 ), "voxelSize", "Voxel size." ) )
    , origin ( initData ( &origin, Vec3d ( 0,0,0 ), "origin", "Grid origin." ) )
    , dimension ( initData ( &dimension, Vec3i ( 0,0,0 ), "dimension", "Grid dimensions." ) )
    , labelToStiffnessPairs ( initData ( &labelToStiffnessPairs, "labelToStiffnessPairs","Correspondances between grid value and stiffness." ) )
    , labelToDensityPairs ( initData ( &labelToDensityPairs, "labelToDensityPairs","Correspondances between grid value and material density." ) )
    , showVoxels ( initData ( &showVoxels, "showVoxelData","Show voxel data." ) )
    , showWeightIndex ( initData ( &showWeightIndex, ( unsigned ) 0, "showWeightIndex","Weight index." ) )
{
    helper::OptionsGroup distanceTypeOptions(4,"Geodesic", "BiasedGeodesic", "HeatDiffusion", "AnisotropicHeatDiffusion");
    distanceTypeOptions.setSelectedItem(DISTANCE_GEODESIC);
    distanceType.setValue(distanceTypeOptions);

    helper::OptionsGroup showVoxelsOptions(7,"None", "Data", "Stiffness", "Density", "Voronoi regions", "Distances", "Weights");
    showVoxelsOptions.setSelectedItem(SHOWVOXELS_NONE);
    showVoxels.setValue(showVoxelsOptions);
}

template<class MaterialTypes>
void GridMaterial< MaterialTypes>::init()
{
    if (imageFile.isSet())
    {
        infoFile=imageFile.getFullPath(); infoFile.replace(infoFile.find_last_of('.')+1,infoFile.size(),"nfo");
        bool writeinfos=false;	if(!loadInfos()) writeinfos=true;
        loadImage();
        if (writeinfos) saveInfos();
    }
    if (weightFile.isSet()) loadWeightRepartion();
    showedrepartition=-1;


//TEST
    /*
    if(weightsRepartition.size()!=nbVoxels)
      {
      VecVec3 points;
      computeUniformSampling(points,6,100);
      computeWeights(6,points);
      }
      */
////


    Inherited::init();
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStress  ( VecStr& /*stress*/, VecStrStr* /*stressStrainMatrices*/, const VecStr& /*strain*/, const VecStr& )
{

//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0] = stressDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][1] = stressOffDiagonal * strain[i][0] + stressDiagonal * strain[i][1] + stressOffDiagonal * strain[i][2];
//                    stress[i][2] = stressOffDiagonal * strain[i][0] + stressOffDiagonal * strain[i][1] + stressDiagonal * strain[i][2];
//                    stress[i][3] = shear * strain[i][3];
//                    stress[i][4] = shear * strain[i][4];
//                    stress[i][5] = shear * strain[i][5];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStress  ( VecElStr& /*stress*/, VecStrStr* /*stressStrainMatrices*/, const VecElStr& /*strain*/, const VecElStr& )
{
//                Real f = youngModulus.getValue()/((1 + poissonRatio.getValue())*(1 - 2 * poissonRatio.getValue()));
//                stressDiagonal = f * (1 - poissonRatio.getValue());
//                stressOffDiagonal = poissonRatio.getValue() * f;
//                shear = f * (1 - 2 * poissonRatio.getValue()) /2;
//
//
//                for(unsigned e=0; e<10; e++)
//                for(unsigned i=0; i<stress.size(); i++)
//                {
//                    stress[i][0][e] = stressDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][1][e] = stressOffDiagonal * strain[i][0][e] + stressDiagonal * strain[i][1][e] + stressOffDiagonal * strain[i][2][e];
//                    stress[i][2][e] = stressOffDiagonal * strain[i][0][e] + stressOffDiagonal * strain[i][1][e] + stressDiagonal * strain[i][2][e];
//                    stress[i][3][e] = shear * strain[i][3][e];
//                    stress[i][4][e] = shear * strain[i][4][e];
//                    stress[i][5][e] = shear * strain[i][5][e];
//                }
//                if( stressStrainMatrices != NULL ){
//                    VecStrStr&  m = *stressStrainMatrices;
//                    m.resize( stress.size() );
//                    m[0].fill(0);
//                    m[0][0][0] = m[0][1][1] = m[0][2][2] = stressDiagonal;
//                    m[0][0][1] = m[0][0][2] = m[0][1][0] = m[0][1][2] = m[0][2][0] = m[0][2][1] = stressOffDiagonal;
//                    m[0][3][3] = m[0][4][4] = m[0][5][5] = shear;
//                    for( unsigned i=1; i<m.size(); i++ ){
//                        m[i] = m[0];
//                    }
//                }
}

template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial<MaterialTypes>::getStiffness(const voxelType label)
{
    const mapLabelType& pairs = labelToStiffnessPairs.getValue();
    if (pairs.size()==0) return (Real)label; // no map defined -> return label

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if (pit->first>label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(Real)((pit->first-label)/(pit->first-mit->first));
                return alpha*vlow+(1.-alpha)*vup;
            }
        }
        mit=pit;
    }
    return mit->second;
}

template < class MaterialTypes>
typename GridMaterial<MaterialTypes>::Real GridMaterial<MaterialTypes>::getDensity(const voxelType label)
{
    const mapLabelType& pairs = labelToDensityPairs.getValue();
    if (pairs.size()==0) return (Real)label; // no map defined -> return label

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if (pit->first>label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(Real)((pit->first-label)/(pit->first-mit->first));
                return alpha*vlow+(1.-alpha)*vup;
            }
        }
        mit=pit;
    }
    return mit->second;
}

/*************************/
/*   IO               */
/*************************/

template < class MaterialTypes>
bool GridMaterial<MaterialTypes>::loadInfos()
{
    if (!infoFile.size()) return false;
    std::ifstream fileStream (infoFile.c_str(), std::ifstream::in);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFile << sendl;
        return false;
    }
    std::string str;
    fileStream >> str;	char vtype[32]; fileStream.getline(vtype,32); // voxeltype not used yet
    fileStream >> str; Vec3i& dim = *this->dimension.beginEdit();       fileStream >> dim;      this->dimension.endEdit();
    fileStream >> str; Vec3d& origin = *this->origin.beginEdit();       fileStream >> origin;   this->origin.endEdit();
    fileStream >> str; Vec3d& voxelsize = *this->voxelSize.beginEdit(); fileStream >> voxelsize; this->voxelSize.endEdit();
    fileStream.close();
    std::cout << "Loaded info file "<< infoFile << std::endl;
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::saveInfos()
{
    if (!infoFile.size()) return false;
    std::ofstream fileStream (infoFile.c_str(), std::ofstream::out);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFile << sendl;
        return false;
    }
    std::cout << "Writing info file " << infoFile << std::endl;
    fileStream << "voxelType: " << CImg<voxelType>::pixel_type() << endl;
    fileStream << "dimensions: " << dimension.getValue() << endl;
    fileStream << "origin: " << origin.getValue() << endl;
    fileStream << "voxelSize: " << voxelSize.getValue() << endl;
    fileStream.close();
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::loadImage()
{
    if (!imageFile.isSet()) return false;
    grid.load_raw(imageFile.getFullPath().c_str(),dimension.getValue()[0],dimension.getValue()[1],dimension.getValue()[2]);
    if (grid.size()==0)
    {
        serr << "Can not open " << imageFile << sendl;
        return false;
    }
    std::cout << "Loaded image "<< imageFile <<" of voxel type " << grid.pixel_type() << std::endl;
    this->nbVoxels = dimension.getValue()[0]*dimension.getValue()[1]*dimension.getValue()[2];
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::saveImage()
{
    if (!imageFile.isSet()) return false;
    if (nbVoxels==0) return false;
    grid.save_raw(imageFile.getFullPath().c_str());
    return true;
}




template < class MaterialTypes>
bool GridMaterial<MaterialTypes>::loadWeightRepartion()
{
    if (!weightFile.isSet()) return false;
    if (nbVoxels==0) return false;

    std::ifstream fileStream (weightFile.getFullPath().c_str(), std::ifstream::in);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << weightFile << sendl;
        return false;
    }
    unsigned int nbrefs,nbvox;
    fileStream >> nbvox;
    fileStream >> nbrefs;
    if (nbVoxels!=nbvox)
    {
        serr << "Invalid grid size in " << weightFile << sendl;
        return false;
    }

    this->repartition.resize(nbVoxels);
    for (unsigned int i=0; i<nbVoxels; i++) this->repartition[i].resize(nbrefs);
    this->weightsRepartition.resize(nbVoxels);
    for (unsigned int i=0; i<nbVoxels; i++)
    {
        this->weightsRepartition[i].resize(nbrefs);
        this->weightsRepartition[i].fill(0);
    }
    for (unsigned int i=0; i<nbVoxels; i++)
        for (unsigned int j=0; j<nbrefs; j++)
        {
            fileStream >> repartition[i][j] ;
            fileStream >> weightsRepartition[i][j];
        }
    fileStream.close();
    std::cout << "Loaded weight file "<< weightFile << std::endl;
    showedrepartition=-1;
    return true;
}



template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::saveWeightRepartion()
{
    if (!weightFile.isSet()) return false;
    if (nbVoxels==0) return false;
    if (weightsRepartition.size()!=nbVoxels) return false;
    if (repartition.size()!=nbVoxels) return false;

    std::ofstream fileStream (weightFile.getFullPath().c_str(), std::ofstream::out);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << weightFile << sendl;
        return false;
    }
    std::cout << "Writing grid weights repartion file " << weightFile << std::endl;
    unsigned int nbrefs=repartition[0].size();
    fileStream << nbVoxels << " " << nbrefs << std::endl;
    for (unsigned int i=0; i<nbVoxels; i++)
    {
        for (unsigned int j=0; j<nbrefs; j++)
        {
            fileStream << repartition[i][j] << " " << weightsRepartition[i][j] << " ";
        }
        fileStream << std::endl;
    }
    fileStream.close();
    return true;
}



/*************************/
/*   Lumping        */
/*************************/

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::LumpMass(const Vec3& point,Real& mass)
{
    if (!nbVoxels)
    {
        mass=0;
        return false;
    }
    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if (index==-1)
    {
        mass=0;
        return false;
    }
    if (voronoi.size()!=nbVoxels)
    {
        mass=voxelvolume*getDensity(grid.data()[index]);    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    if (voronoi[index]==-1)
    {
        mass=voxelvolume*getDensity(grid.data()[index]);    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    mass=0;
    for (unsigned int i=0; i<nbVoxels; i++) if (voronoi[i]==voronoi[index]) mass+=voxelvolume*getDensity(grid.data()[i]);
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::LumpVolume(const Vec3& point,Real& vol)
{
    if (!nbVoxels)
    {
        vol=0;
        return false;
    }
    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if (index==-1)
    {
        vol=0;
        return false;
    }
    if (voronoi.size()!=nbVoxels)
    {
        vol=voxelvolume;    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    if (voronoi[index]==-1)
    {
        vol=voxelvolume;    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    vol=0;
    for (unsigned int i=0; i<nbVoxels; i++) if (voronoi[i]==voronoi[index]) vol+=voxelvolume;
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::LumpMoments(const Vec3& point,const unsigned int order,VD& moments)
{
    if (!nbVoxels)
    {
        moments.clear();
        return false;
    }
    unsigned int dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    unsigned int i,j;
    moments.resize(dim);
    for (i=0; i<dim; i++) moments[i]=0;

    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if (index==-1)
    {
        moments.clear();
        return false;
    }
    if (voronoi.size()!=nbVoxels)
    {
        moments[0]=voxelvolume;    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    if (voronoi[index]==-1)
    {
        moments[0]=voxelvolume;    // no voronoi -> 1 point = 1 voxel
        return false;
    }

    Vec3 P,G;
    getCoord(index,P);
    VD momentPG;
    for (i=0; i<nbVoxels; i++)
        if (voronoi[i]==voronoi[index])
        {
            getCoord(i,G);
            getCompleteBasis(G-P,order,momentPG);
            for (j=0; j<dim; j++) moments[j]+=momentPG[j]*voxelvolume;
        }
    return true;  // 1 point = voxels in its voronoi region
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::LumpMomentsStiffness(const Vec3& point,const unsigned int order,VD& moments)
{
    if (!nbVoxels)
    {
        moments.clear();
        return false;
    }
    LumpMoments(point,order,moments);
    unsigned int dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    int index=getIndex(point);
    if (index==-1)
    {
        moments.clear();
        return false;
    }
    if (voronoi.size()!=nbVoxels)
    {
        moments[0]*=getStiffness(grid.data()[index]);    // no voronoi -> 1 point = 1 voxel
        return false;
    }
    if (voronoi[index]==-1)
    {
        moments[0]*=getStiffness(grid.data()[index]);    // no voronoi -> 1 point = 1 voxel
        return false;
    }

    unsigned int i,j;
    for (i=0; i<nbVoxels; i++)
        if (voronoi[i]==voronoi[index])
            for (j=0; j<dim; j++) moments[j]*=getStiffness(grid.data()[i]);

    return true;
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::LumpWeights(const Vec3& point,const bool dilatevoronoi,Real& w,Vec3* dw,Mat33* ddw)
{
    if (!nbVoxels) return false;

    int index=getIndex(point);
    if (index==-1 || weights.size()!=nbVoxels) return false; // point not in grid or no weight omputed

    unsigned int i,j,k;

    bool fitononevoxel=false;
    if (voronoi.size()!=nbVoxels) fitononevoxel=true;
    if (voronoi[index]==-1) fitononevoxel=true;

// get point indices for fitting
    VUI neighbors;
    if (fitononevoxel)
    {
        get26Neighbors(index, neighbors);
        neighbors.push_back((unsigned int)index);
    }
    else
    {
        for (i=0; i<nbVoxels; i++) if (voronoi[i]==voronoi[index]) neighbors.push_back((unsigned int)i);
        if (dilatevoronoi)
            for (i=0; i<nbVoxels; i++)
                if (grid.data()[i])
                {
                    VUI tmp;
                    get6Neighbors(index, tmp);
                    bool insert=false;
                    for (j=0; j<tmp.size(); j++) if (voronoi[tmp[j]]==voronoi[index]) insert=true;
                    if (insert) neighbors.push_back((unsigned int)i);
                }
        if (neighbors.size()<26) // not enough samples -> go back to a local neighbor fit
        {
            get26Neighbors(index, neighbors);
            neighbors.push_back((unsigned int)index);
        }
    }

// least squares fit
    unsigned int order=0;
    if (ddw && dw) order=2;
    else if (dw) order=1;
    unsigned int dim=(order+1)*(order+2)*(order+3)/6;
    VVD Cov((int)dim,VD((int)dim,(Real)0)); // accumulate dp^(order)dp^(order)T
    Vec3 pi;
    for (j=0; j<neighbors.size(); j++)
    {
        getCoord(neighbors[j],pi);
        accumulateCovariance(pi-point,order,Cov);
    }

    // invert covariance matrix
    VVD invCov((int)dim,VD((int)dim,(Real)0));
    if (order==0)
    {
        invCov[0][0]=1./Cov[0][0];    // simple average
    }
    else if (order==1)
    {
        Mat<4,4,Real> tmp,invtmp;
        for (i=0; i<4; i++) for (j=0; j<4; j++) tmp[i][j]=Cov[i][j];
        if (!invtmp.invert(tmp))
        {
            invCov[0][0]=1./Cov[0][0];    // coplanar points->not invertible->go back to simple average
        }
        else
        {
            for (i=0; i<4; i++) for (j=0; j<4; j++) invCov[i][j]=invtmp[i][j];
        }
    }
    else if (order==2)
    {
        Mat<10,10,Real> tmp,invtmp;
        for (i=0; i<10; i++) for (j=0; j<10; j++) tmp[i][j]=Cov[i][j];
        if (!invtmp.invert(tmp))
        {
            invCov[0][0]=1./Cov[0][0];    // coplanar points->not invertible->go back to simple average
        }
        else
        {
            for (i=0; i<10; i++) for (j=0; j<10; j++) invCov[i][j]=invtmp[i][j];
        }
    }

    // compute the weights and its derivatives
    VD basis,wdp((int)dim,(Real)0);
    for (j=0; j<neighbors.size(); j++)
    {
        getCoord(neighbors[j],pi);
        getCompleteBasis(pi-point,order,basis);
        for (i=0; i<dim; i++) wdp[i]+=basis[i]*weights[neighbors[j]];
    }
    VD W((int)dim,(Real)0);
    for (i=0; i<dim; i++) for (j=0; i<dim; j++) W[i]+=invCov[i][j]*wdp[j];

    w=W[0];
    if (order==0) return true;

    VVD basisderiv;
    getCompleteBasisDeriv(Vec3(0,0,0),order,basisderiv);
    for (i=0; i<3; i++)
    {
        (*dw)[i]=0;
        for (j=0; j<dim; j++) (*dw)[i]+=W[j]*basisderiv[j][i];
    }
    if (order==1) return true;

    VVVD basisderiv2;
    getCompleteBasisDeriv2(Vec3(0,0,0),order,basisderiv2);
    for (i=0; i<3; i++) for (k=0; k<3; k++)
        {
            (*ddw)[i][k]=0;    // dddw[i][k]=dw[i]/p_k
            for (j=0; j<dim; j++) (*ddw)[i][k]+=W[j]*basisderiv2[j][i][k];
        }
    return true;
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::accumulateCovariance(const Vec3& p,const unsigned int order,VVD& Cov)
{
    VD basis;
    getCompleteBasis(p,order,basis);
    unsigned int dim=(order+1)*(order+2)*(order+3)/6;
    for (unsigned int i=0; i<dim; i++) for (unsigned int j=0; j<dim; j++) Cov[i][j]+=basis[i]*basis[j];
}


/*********************************/
/*   Compute weights/distances   */
/*********************************/


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeWeights(const unsigned int nbrefs,const VecVec3& points)
{
// TO DO: add params as data? : GEODESIC factor, DIFFUSION fixdatavalue, DIFFUSION max_iterations, DIFFUSION precision
    if (!nbVoxels) return false;
    unsigned int i,dtype=this->distanceType.getValue().getSelectedId(),nbp=points.size();

// init
    this->repartition.resize(nbVoxels);
    for (i=0; i<nbVoxels; i++) this->repartition[i].resize(nbrefs);
    this->weightsRepartition.resize(nbVoxels);
    for (i=0; i<nbVoxels; i++)
    {
        this->weightsRepartition[i].resize(nbrefs);
        this->weightsRepartition[i].fill(0);
    }

    if (dtype==DISTANCE_GEODESIC || dtype==DISTANCE_BIASEDGEODESIC)
    {
        computeGeodesicalDistances (points); // voronoi
        for (i=0; i<nbp; i++)
        {
            computeLinearWeightsInVoronoi(points[i]);
            addWeightinRepartion(i);
        }
    }
    else if (dtype==DISTANCE_DIFFUSION || dtype==DISTANCE_ANISOTROPICDIFFUSION)
    {
        for (i=0; i<nbp; i++)
        {
            HeatDiffusion(points,i);
            addWeightinRepartion(i);
        }
    }

    normalizeWeightRepartion();

    std::cout<<"Grid weight computation completed"<<std::endl;
    if (weightFile.isSet()) saveWeightRepartion();
    showedrepartition=-1;
    return true;
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::addWeightinRepartion(const unsigned int index)
{
    if (!nbVoxels) return;
    unsigned int j,k,nbrefs=repartition[0].size();

    for (unsigned int i=0; i<nbVoxels; i++)
        if (grid.data()[i])
        {
            j=0;
            while (j!=nbrefs && weightsRepartition[i][j]>weights[i]) j++;
            if (j!=nbrefs) // insert weight and index in the ordered list
            {
                for (k=nbrefs-1; k>j; k--)
                {
                    weightsRepartition[i][k]=weightsRepartition[i][k-1];
                    repartition[i][k]=repartition[i][k-1];
                }
                weightsRepartition[i][j]=weights[i];
                repartition[i][j]=index;
            }
        }
    showedrepartition=-1;
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::pasteRepartioninWeight(const unsigned int index)
{
    if (!nbVoxels) return;
    if (repartition.size()!=nbVoxels) return;
    if (weightsRepartition.size()!=nbVoxels) return;
    weights.resize(this->nbVoxels);

    unsigned int i,j,nbrefs=repartition[0].size();
    for (i=0; i<nbVoxels; i++)
        if (grid.data()[i])
        {
            j=0;
            while (j!=nbrefs && repartition[i][j]!=index) j++;
            if (j!=nbrefs) weights[i]=weightsRepartition[i][j];
            else weights[i]=0;
        }
        else weights[i]=0;
    showedrepartition=-1;
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::normalizeWeightRepartion()
{
    if (!nbVoxels) return;
    unsigned int j,nbrefs=repartition[0].size();
    for (unsigned int i=0; i<nbVoxels; i++)
        if (grid.data()[i])
        {
            Real W=0;
            for (j=0; j<nbrefs; j++) W+=weightsRepartition[i][j];
            if (W!=0) for (j=0; j<nbrefs; j++) weightsRepartition[i][j]/=W;
        }
    showedrepartition=-1;
}


template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial< MaterialTypes>::getDistance(const unsigned int& index1,const unsigned int& index2)
{
    if (!nbVoxels) return -1;
    Vec3 coord1;
    if (!getCoord(index1,coord1)) return -1; // point1 not in grid
    Vec3 coord2;
    if (!getCoord(index2,coord2)) return -1; // point2 not in grid

    if (this->distanceType.getValue().getSelectedId()==DISTANCE_BIASEDGEODESIC) // bias distances according to stiffness
    {
        Real meanstiff=(getStiffness(grid.data()[index1])+getStiffness(grid.data()[index2]))/2.;
        return ((Real)(coord2-coord1).norm()/meanstiff);
    }
    else return (Real)(coord2-coord1).norm();
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const Vec3& point, const Real distMax )
{
    if (!nbVoxels) return false;
    int index=getIndex(point);
    return computeGeodesicalDistances (index, distMax );
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const int& index, const Real distMax )
{
    if (!nbVoxels) return false;
    unsigned int i,index1,index2;
    distances.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++) distances[i]=distMax;

    if (index<0 || index>=(int)nbVoxels) return false; // voxel out of grid
    if (!grid.data()[index]) return false;  // voxel out of object

    VUI neighbors;
    Real d;
    std::queue<int> fifo;
    distances[index]=0;
    fifo.push(index);
    while (!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for (i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if (grid.data()[index2]) // test if voxel is not void
            {
                d=distances[index1]+getDistance(index1,index2);
                if (distances[index2]>d)
                {
                    distances[index2]=d;
                    fifo.push(index2);
                }
            }
        }
        fifo.pop();
    }
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const VecVec3& points, const Real distMax )
{
    if (!nbVoxels) return false;
    VI indices;
    for (unsigned int i=0; i<points.size(); i++) indices.push_back(getIndex(points[i]));
    return computeGeodesicalDistances ( indices, distMax );
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const VI& indices, const Real distMax )
{
    if (!nbVoxels) return false;
    unsigned int i,nbi=indices.size(),index1,index2;
    distances.resize(this->nbVoxels);
    voronoi.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)
    {
        distances[i]=distMax;
        voronoi[i]=-1;
    }

    VUI neighbors;
    Real d;

    std::queue<unsigned int> fifo;
    for (i=0; i<nbi; i++) if (indices[i]>=0 && indices[i]<(int)nbVoxels) if (grid.data()[indices[i]]!=0)
            {
                distances[indices[i]]=0;
                voronoi[indices[i]]=i;
                fifo.push(indices[i]);
            }
    if (fifo.empty()) return false; // all input voxels out of grid
    while (!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for (i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if (grid.data()[index2]) // test if voxel is not void
            {
                d=distances[index1]+getDistance(index1,index2);
                if (distances[index2]>d)
                {
                    distances[index2]=d;
                    voronoi[index2]=voronoi[index1];
                    fifo.push(index2);
                }
            }
        }
        fifo.pop();
    }
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeUniformSampling ( VecVec3& points, const unsigned int num_points, const unsigned int max_iterations )
{
    if (!nbVoxels) return false;
    unsigned int i,k,initial_num_points=points.size();
    VI indices((int)num_points,-1);
    for (i=0; i<initial_num_points; i++) indices[i]=getIndex(points[i]);
    points.resize(num_points);

// initialization: farthest point sampling (see [adams08])
    Real dmax;
    int indexmax;
    if (initial_num_points==0)
    {
        indices[0]=0;    // take the first not empty voxel as a random point
        while (grid.data()[indices[0]]==0)
        {
            indices[0]++;
            if (indices[0]==(int)nbVoxels) return false;
        }
    }
    for (i=initial_num_points; i<num_points; i++)
    {
        if (i==0) i=1; // a random point has been inserted
        // get farthest point from all inserted points
        computeGeodesicalDistances(indices);
        dmax=-1;
        indexmax=-1;
        for (k=0; k<nbVoxels; k++)  if (grid.data()[k])
            {
                if (distances[k]>dmax && voronoi[k]!=-1)
                {
                    dmax=distances[k];
                    indexmax=k;
                }
            }
        if (indexmax==-1)
        {
            return false;    // unable to add point
        }
        indices[i]=indexmax;
    }
// Lloyd relaxation
    Vec3 pos,u,pos_point,pos_voxel;
    unsigned int count,nbiterations=0;
    bool ok=false,ok2;
    Real d,dmin;
    int indexmin;

    while (!ok && nbiterations<max_iterations)
    {
        ok2=true;
        computeGeodesicalDistances(indices); // Voronoi
        VB flag((int)nbVoxels,false);
        for (i=initial_num_points; i<num_points; i++) // move to centroid of Voronoi cells
        {
            // estimate centroid given the measured distances = p + 1/N sum d(p,pi)*(pi-p)/|pi-p|
            getCoord(indices[i],pos_point);
            pos.fill(0);
            count=0;
            for (k=0; k<nbVoxels; k++)
                if (voronoi[k]==(int)i)
                {
                    getCoord(k,pos_voxel);
                    u=pos_voxel-pos_point;
                    u.normalize();
                    pos+=u*(Real)distances[k];
                    count++;
                }
            pos/=(Real)count;
            pos+=pos_point;
            // get closest unoccupied point in object
            dmin=std::numeric_limits<Real>::max();
            indexmin=-1;
            for (k=0; k<nbVoxels; k++) if (!flag[k]) if (grid.data()[k]!=0)
                    {
                        getCoord(k,pos_voxel);
                        d=(pos-pos_voxel).norm2();
                        if (d<dmin)
                        {
                            dmin=d;
                            indexmin=k;
                        }
                    }
            flag[indexmin]=true;
            if (indices[i]!=indexmin)
            {
                ok2=false;
                indices[i]=indexmin;
            }
        }
        ok=ok2;
        nbiterations++;
    }

// get points from indices
    for (i=initial_num_points; i<num_points; i++)
    {
        getCoord(indices[i],points[i]) ;
    }

    if (nbiterations==max_iterations)
    {
        serr<<"Lloyd relaxation has not converged in "<<nbiterations<<" iterations"<<sendl;
        return false;
    }
    else std::cout<<"Lloyd relaxation completed in "<<nbiterations<<" iterations"<<std::endl;
    return true;
}



template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeLinearWeightsInVoronoi ( const Vec3& point, const Real factor)
{
    unsigned int i;
    weights.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)  weights[i]=0;
    if (!this->nbVoxels) return false;
    int index=getIndex(point);
    if (voronoi.size()!=nbVoxels) return false;
    if (voronoi[index]==-1) return false;
    Real dmax=0;
    for (i=0; i<nbVoxels; i++) if (grid.data()[i])  if (voronoi[i]==voronoi[index]) if (distances[i]>dmax) dmax=distances[i];
    if (dmax==0) return false;
    dmax*=factor;
    VD backupdistance;
    backupdistance.swap(distances);
    computeGeodesicalDistances(point,dmax);
    for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (distances[i]<dmax) weights[i]=1.-distances[i]/dmax;
    backupdistance.swap(distances);
    showedrepartition=-1;
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::HeatDiffusion( const VecVec3& points, const unsigned int hotpointindex,const bool fixdatavalue,const unsigned int max_iterations,const Real precision  )
{
    if (!this->nbVoxels) return false;
    unsigned int i,j,k,num_points=points.size();
    int index;
    weights.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)  weights[i]=0;

    VB isfixed((int)nbVoxels,false);
    VB update((int)nbVoxels,false);
    VUI neighbors;

// intialisation: fix weight of points or regions
    for (i=0; i<num_points; i++)
    {
        index=getIndex(points[i]);
        if (index!=-1)
        {
            isfixed[index]=true;
            if (i==hotpointindex) weights[index]=1;
            else weights[index]=0;
            get6Neighbors(index, neighbors);
            for (j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
            if (fixdatavalue) // fix regions
                for (k=0; k<this->nbVoxels; k++)
                    if (grid.data()[k]==grid.data()[index])
                    {
                        isfixed[k]=true;
                        weights[k]=weights[index];
                        get6Neighbors(k, neighbors);
                        for (j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
                    }
        }
    }

// diffuse
    unsigned int nbiterations=0;
    bool ok=false,ok2;
    Real maxchange=0.;
    while (!ok && nbiterations<max_iterations)
    {
        ok2=true;
        maxchange=0;
//  #pragma omp parallel for private(j,neighbors)
        for (i=0; i<this->nbVoxels; i++)
            if (grid.data()[i])
                if (update[i])
                {
                    if (isfixed[i]) update[i]=false;
                    else
                    {
                        Real val=0,W=0;

                        if (this->distanceType.getValue().getSelectedId()==DISTANCE_ANISOTROPICDIFFUSION)
                        {
                            Real dv2;
                            bool okp,okm;
                            int ip,im;
                            Vec3i icoord;
                            getiCoord(i,icoord);
                            neighbors.clear();
                            for (j=0; j<3 ; j++)
                            {
                                icoord[j]+=1;
                                ip=getIndex(icoord);
                                icoord[j]-=2;
                                im=getIndex(icoord);
                                icoord[j]+=1;
                                okp=false;
                                if (ip!=-1) if (grid.data()[ip])
                                    {
                                        okp=true;
                                        neighbors.push_back(ip);
                                    }
                                okm=false;
                                if (im!=-1) if (grid.data()[im])
                                    {
                                        okm=true;
                                        neighbors.push_back(im);
                                    }
                                if (okp && okm)
                                {
                                    dv2=(weights[ip]-weights[im])*(weights[ip]-weights[im])/4.;
                                    val+=(weights[ip]+weights[im])*dv2;
                                    W+=2*dv2;
                                }
                                else if (!okp && okm)
                                {
                                    dv2=(weights[i]-weights[im])*(weights[i]-weights[im]);
                                    val+=(weights[i]+weights[im])*dv2;
                                    W+=2*dv2;
                                }
                                else if (okp && !okm)
                                {
                                    dv2=(weights[ip]-weights[i])*(weights[ip]-weights[i]);
                                    val+=(weights[ip]+weights[i])*dv2;
                                    W+=2*dv2;
                                }
                            }
                        }
                        else
                        {
                            get6Neighbors(i, neighbors);
                            for (j=0; j<neighbors.size(); j++)
                            {
                                val+=weights[neighbors[j]];
                                W+=1.;
                            }
                            for (j=neighbors.size(); j<6; j++)
                            {
                                val+=weights[i];    // dissipative border
                                W+=1.;
                            }
                        }
                        if (W!=0) val=val/W; // normalize value

                        if (fabs(val-weights[i])<precision) update[i]=false;
                        else
                        {
                            if (fabs(val-weights[i])>maxchange) maxchange=fabs(val-weights[i]);
                            weights[i]=val;
                            ok2=false;
                            for (j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
                        }
                    }
                }
        ok=ok2;
        nbiterations++;
    }

    if (nbiterations==max_iterations)
    {
        serr<<"Heat diffusion has not converged in "<<nbiterations<<" iterations (precision="<<maxchange<<")"<<sendl;
        return false;
    }
    else std::cout<<"Heat diffusion completed in "<<nbiterations<<" iterations"<<std::endl;
    showedrepartition=-1;
    return true;
}





/*************************/
/*         Draw          */
/*************************/

template<class MaterialTypes>
void GridMaterial< MaterialTypes>::draw()
{
    if (!nbVoxels) return;

    unsigned int showvox=this->showVoxels.getValue().getSelectedId();

    if ( showvox!=SHOWVOXELS_NONE)
    {
        //glDisable ( GL_LIGHTING );
        //glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;

        unsigned int i;
        Real s=(voxelSize.getValue()[0]+voxelSize.getValue()[1]+voxelSize.getValue()[2])/3.;
        float defaultcolor[4]= {0.8,0.8,0.8,0.3},color[4];
        bool wireframe=this->getContext()->getShowWireFrame();

        float label,labelmax=-1;

        if (showvox==SHOWVOXELS_DATAVALUE)
        {
            for (i=0; i<nbVoxels; i++) if (grid.data()[i]>labelmax) labelmax=(float)grid.data()[i];
        }
        else if (showvox==SHOWVOXELS_STIFFNESS)
        {
            labelmax=20000;
        }
        else if (showvox==SHOWVOXELS_DENSITY)
        {
            labelmax=10000;
        }
        else if (voronoi.size()==nbVoxels && showvox==SHOWVOXELS_VORONOI)
        {
            for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (voronoi[i]>labelmax) labelmax=(float)voronoi[i];
        }
        else if (distances.size()==nbVoxels && showvox==SHOWVOXELS_DISTANCES)
        {
            for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (distances[i]>labelmax) labelmax=(float)distances[i];
        }
        else if (showvox==SHOWVOXELS_WEIGHTS)
        {
            labelmax=1;
            if (weightsRepartition.size()==nbVoxels && repartition.size()==nbVoxels) // paste weightsRepartition into weights
            {
                if (showedrepartition!=(int)showWeightIndex.getValue()) pasteRepartioninWeight(showWeightIndex.getValue());
                showedrepartition=(int)showWeightIndex.getValue();
            }
        }


        cimg_forXYZ(grid,x,y,z)
        {
            if (grid(x,y,z)!=0)
            {
                VUI neighbors;
                get6Neighbors(getIndex(Vec3i(x,y,z)), neighbors);
                if (neighbors.size()!=6) // disable internal voxels
                {
                    label=-1;
                    if (labelmax!=-1)
                    {
                        if (showvox==SHOWVOXELS_DATAVALUE) label=(float)grid(x,y,z);
                        else if (showvox==SHOWVOXELS_STIFFNESS) label=(float)getStiffness(grid(x,y,z));
                        else if (showvox==SHOWVOXELS_DENSITY) label=(float)getDensity(grid(x,y,z));
                        else if (voronoi.size()==nbVoxels && showvox==SHOWVOXELS_VORONOI)  label=(float)voronoi[getIndex(Vec3i(x,y,z))];
                        else if (distances.size()==nbVoxels && showvox==SHOWVOXELS_DISTANCES)  label=(float)distances[getIndex(Vec3i(x,y,z))];
                        else if (weights.size()==nbVoxels && showvox==SHOWVOXELS_WEIGHTS)  label=(float)weights[getIndex(Vec3i(x,y,z))];
                    }

                    if (label==-1)
                    {
                        glColor4fv(defaultcolor);
                        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,defaultcolor);
                    }
                    else
                    {
                        if (label>labelmax) label=labelmax;
                        helper::gl::Color::setHSVA(240.*label/labelmax,1.,.8,defaultcolor[3]);
                        glGetFloatv(GL_CURRENT_COLOR, color);
                        glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
                    }
                    Vec3 coord;
                    getCoord(Vec3i(x,y,z),coord);
                    glTranslated ((double)coord[0],(double)coord[1],(double)coord[2]);
                    drawCube(s,wireframe);
                    glTranslated (-(double)coord[0],-(double)coord[1],-(double)coord[2]);
                    //GlText::draw ( (int), coord, showTextScaleFactor.getValue() );
                }
            }
        }
    }
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::drawCube(Real size,bool wireframe)
{
    double ss2=(double)size/2.;
    if (!wireframe) glBegin(GL_QUADS);
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(1,0,0);
    glVertex3d(ss2,-ss2,-ss2);
    glNormal3f(1,0,0);
    glVertex3d(ss2,-ss2,ss2);
    glNormal3f(1,0,0);
    glVertex3d(ss2,ss2,ss2);
    glNormal3f(1,0,0);
    glVertex3d(ss2,ss2,-ss2);
    if (wireframe) glEnd ();
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(-1,0,0);
    glVertex3d(-ss2,-ss2,-ss2);
    glNormal3f(-1,0,0);
    glVertex3d(-ss2,-ss2,ss2);
    glNormal3f(-1,0,0);
    glVertex3d(-ss2,ss2,ss2);
    glNormal3f(-1,0,0);
    glVertex3d(-ss2,ss2,-ss2);
    if (wireframe) glEnd ();
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(0,1,0);
    glVertex3d(-ss2,ss2,-ss2);
    glNormal3f(0,1,0);
    glVertex3d(ss2,ss2,-ss2);
    glNormal3f(0,1,0);
    glVertex3d(ss2,ss2,ss2);
    glNormal3f(0,1,0);
    glVertex3d(-ss2,ss2,ss2);
    if (wireframe) glEnd ();
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(0,-1,0);
    glVertex3d(-ss2,-ss2,-ss2);
    glNormal3f(0,-1,0);
    glVertex3d(ss2,-ss2,-ss2);
    glNormal3f(0,-1,0);
    glVertex3d(ss2,-ss2,ss2);
    glNormal3f(0,-1,0);
    glVertex3d(-ss2,-ss2,ss2);
    if (wireframe) glEnd ();
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(0,0,1);
    glVertex3d(-ss2,-ss2,ss2);
    glNormal3f(0,0,1);
    glVertex3d(-ss2,ss2,ss2);
    glNormal3f(0,0,1);
    glVertex3d(ss2,ss2,ss2);
    glNormal3f(0,0,1);
    glVertex3d(ss2,-ss2,ss2);
    if (wireframe) glEnd ();
    if (wireframe) glBegin(GL_LINE_LOOP);
    glNormal3f(0,0,-1);
    glVertex3d(-ss2,-ss2,-ss2);
    glNormal3f(0,0,-1);
    glVertex3d(-ss2,ss2,-ss2);
    glNormal3f(0,0,-1);
    glVertex3d(ss2,ss2,-ss2);
    glNormal3f(0,0,-1);
    glVertex3d(ss2,-ss2,-ss2);
    if (wireframe) glEnd ();
    if (!wireframe) glEnd ();
}


/*************************/
/*         Utils         */
/*************************/
template < class MaterialTypes>
int GridMaterial< MaterialTypes>::getIndex(const Vec3i& icoord)
{
    if (!nbVoxels) return -1;
    for (int i=0; i<3; i++) if (icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return -1; // invalid icoord (out of grid)
    return icoord[0]+dimension.getValue()[0]*icoord[1]+dimension.getValue()[0]*dimension.getValue()[1]*icoord[2];
}

template < class MaterialTypes>
int GridMaterial< MaterialTypes>::getIndex(const Vec3& coord)
{
    if (!nbVoxels) return -1;
    Vec3i icoord;
    if (!getiCoord(coord,icoord)) return -1;
    return getIndex(icoord);
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getiCoord(const Vec3& coord, Vec3i& icoord)
{
    if (!nbVoxels) return false;
    Real val;
    for (unsigned int i=0; i<3; i++)
    {
        val=(coord[i]-(Real)origin.getValue()[i])/(Real)voxelSize.getValue()[i];
        val=((val-floor(val))<0.5)?floor(val):ceil(val); //round
        if (val<0 || val>=dimension.getValue()[i]) return false;
        icoord[i]=(int)val;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getiCoord(const int& index, Vec3i& icoord)
{
    if (!nbVoxels) return false;
    if (index<0 || index>=(int)nbVoxels) return false; // invalid index
    icoord[2]=index/(dimension.getValue()[0]*dimension.getValue()[1]);
    icoord[1]=(index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1])/dimension.getValue()[0];
    icoord[0]=index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1]-icoord[1]*dimension.getValue()[0];
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getCoord(const Vec3i& icoord, Vec3& coord)
{
    if (!nbVoxels) return false;
    for (unsigned int i=0; i<3; i++) if (icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return false; // invalid icoord (out of grid)
    coord=this->origin.getValue();
    for (unsigned int i=0; i<3; i++) coord[i]+=(Real)this->voxelSize.getValue()[i]*(Real)icoord[i];
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getCoord(const int& index, Vec3& coord)
{
    if (!nbVoxels) return false;
    Vec3i icoord;
    if (!getiCoord(index,icoord)) return false;
    else return getCoord(icoord,coord);
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get6Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]+=1;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get18Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]+=1;
    }

    for (unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]-=2;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get26Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
        icoord[j]+=1;
    }
    for (unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]-=2;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    icoord[0]+=1;
    icoord[1]+=1;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]-=2;
    icoord[1]+=1;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) if (grid.data()[i]) neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]+=1;
    return true;
}



template < class MaterialTypes>
void GridMaterial< MaterialTypes>::getCompleteBasis(const Vec3& p,const unsigned int order,VD& basis)
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'
    basis.resize(dim);
    for (j=0; j<dim; j++) basis[j]=0;

    Vec3 p2;
    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    Vec3 p3;
    for (j=0; j<3; j++) p3[j]=p2[j]*p[j];

    count=0;
// order 0
    basis[count]=1;
    count++;
    if (count==dim) return;
// order 1
    for (j=0; j<3; j++)
    {
        basis[count]=p[j];
        count++;
    }
    if (count==dim) return;
// order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basis[count]=p[j]*p[k];
            count++;
        }
    if (count==dim) return;
// order 3
    basis[count]=p[0]*p[1]*p[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
        {
            basis[count]=p2[j]*p[k];
            count++;
        }
    if (count==dim) return;
// order 4
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basis[count]=p2[j]*p2[k];
            count++;
        }
    basis[count]=p2[0]*p[1]*p[2];
    count++;
    basis[count]=p[0]*p2[1]*p[2];
    count++;
    basis[count]=p[0]*p[1]*p2[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
            {
                basis[count]=p3[j]*p[k];
                count++;
            }
    if (count==dim) return;

    return; // order>4 not implemented...
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::getCompleteBasisDeriv(const Vec3& p,const unsigned int order,VVD& basisDeriv)
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    basisDeriv.resize(dim);
    for (j=0; j<dim; j++)
    {
        basisDeriv[j].resize(3);
        basisDeriv[j].fill(0);
    }

    Vec3 p2;
    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    Vec3 p3;
    for (j=0; j<3; j++) p3[j]=p2[j]*p[j];

    count=0;
// order 0
    count++;
    if (count==dim) return;
// order 1
    for (j=0; j<3; j++)
    {
        basisDeriv[count][j]=1;
        count++;
    }
    if (count==dim) return;
// order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basisDeriv[count][k]+=p[j];
            basisDeriv[count][j]+=p[k];
            count++;
        }
    if (count==dim) return;
// order 3
    basisDeriv[count][0]=p[1]*p[2];
    basisDeriv[count][1]=p[0]*p[2];
    basisDeriv[count][2]=p[0]*p[1];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
        {
            basisDeriv[count][k]+=p2[j];
            basisDeriv[count][j]+=2*p[j]*p[k];
            count++;
        }
    if (count==dim) return;
// order 4
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basisDeriv[count][k]=2*p2[j]*p[k];
            basisDeriv[count][j]=2*p[j]*p2[k];
            count++;
        }
    basisDeriv[count][0]=2*p[0]*p[1]*p[2];
    basisDeriv[count][1]=p2[0]*p[2];
    basisDeriv[count][2]=p2[0]*p[1];
    count++;
    basisDeriv[count][0]=p2[1]*p[2];
    basisDeriv[count][1]=2*p[0]*p[1]*p[2];
    basisDeriv[count][2]=p[0]*p2[1];
    count++;
    basisDeriv[count][0]=p[1]*p2[2];
    basisDeriv[count][1]=p[0]*p2[2];
    basisDeriv[count][2]=2*p[0]*p[1]*p[2];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
            {
                basisDeriv[count][k]=p3[j];
                basisDeriv[count][j]=3*p2[j]*p[k];
                count++;
            }
    if (count==dim) return;

    return; // order>4 not implemented...
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::getCompleteBasisDeriv2(const Vec3& p,const unsigned int order,VVVD& basisDeriv)
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    basisDeriv.resize(dim);
    for (k=0; k<dim; k++)
    {
        basisDeriv[k].resize(dim);
        for (j=0; j<dim; j++)
        {
            basisDeriv[k][j].resize(3);
            basisDeriv[k][j].fill(0);
        }
    }

    Vec3 p2;
    for (j=0; j<3; j++) p2[j]=p[j]*p[j];

    count=0;
// order 0
    count++;
    if (count==dim) return;
// order 1
    count+=3;
    if (count==dim) return;
// order 2
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basisDeriv[count][k][j]+=1;
            basisDeriv[count][j][k]+=1;
            count++;
        }
    if (count==dim) return;
// order 3
    basisDeriv[count][0][1]=p[2];
    basisDeriv[count][0][2]=p[1];
    basisDeriv[count][1][0]=p[2];
    basisDeriv[count][1][2]=p[0];
    basisDeriv[count][2][0]=p[1];
    basisDeriv[count][2][1]=p[0];
    count++;
    for (j=0; j<3; j++) for (k=0; k<3; k++)
        {
            basisDeriv[count][k][j]+=2*p[j];
            basisDeriv[count][j][k]+=2*p[j];
            count++;
        }
    if (count==dim) return;
// order 4
    for (j=0; j<3; j++) for (k=j; k<3; k++)
        {
            basisDeriv[count][k][j]=4*p[j]*p[k];
            basisDeriv[count][k][k]=2*p2[j];
            basisDeriv[count][j][j]=2*p2[k];
            basisDeriv[count][j][k]=4*p[j]*p[k];
            count++;
        }
    basisDeriv[count][0][0]=2*p[1]*p[2];
    basisDeriv[count][0][1]=2*p[0]*p[2];
    basisDeriv[count][0][2]=2*p[0]*p[1];
    basisDeriv[count][1][0]=2*p[0]*p[2];
    basisDeriv[count][1][2]=p2[0];
    basisDeriv[count][2][0]=2*p[0]*p[1];
    basisDeriv[count][2][1]=p2[0];
    count++;
    basisDeriv[count][0][1]=2*p[1]*p[2];
    basisDeriv[count][0][2]=p2[1];
    basisDeriv[count][1][0]=2*p[1]*p[2];
    basisDeriv[count][1][1]=2*p[0]*p[2];
    basisDeriv[count][1][2]=2*p[0]*p[1];
    basisDeriv[count][2][0]=p2[1];
    basisDeriv[count][2][1]=2*p[0]*p[1];
    count++;
    basisDeriv[count][0][1]=p2[2];
    basisDeriv[count][0][2]=2*p[1]*p[2];
    basisDeriv[count][1][0]=p2[2];
    basisDeriv[count][1][2]=2*p[0]*p[2];
    basisDeriv[count][2][0]=2*p[1]*p[2];
    basisDeriv[count][2][1]=2*p[0]*p[2];
    basisDeriv[count][2][2]=2*p[0]*p[1];
    count++;

    for (j=0; j<3; j++) for (k=0; k<3; k++) if (j!=k)
            {
                basisDeriv[count][k][j]=3*p2[j];
                basisDeriv[count][j][j]=6*p[j]*p[k];
                basisDeriv[count][j][k]=3*p2[j];
                count++;
            }
    if (count==dim) return;

    return; // order>4 not implemented...
}

} // namespace material

} // namespace component

} // namespace sofa

#endif


