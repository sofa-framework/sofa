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


template<class MaterialTypes, typename voxelType>
GridMaterial< MaterialTypes,voxelType>::GridMaterial()
    : Inherited()
    , showVoxelData ( initData ( &showVoxelData, false, "showVoxelData","Show voxel data" ) )
    , showVoronoi ( initData ( &showVoronoi, false, "showVoronoi","Show voronoi" ) )
    , showDistances ( initData ( &showDistances, false, "showDistances","Show distances" ) )
    , showWeights ( initData ( &showWeights, false, "showWeights","Show weights" ) )
    , imageFile( initData(&imageFile,"imageFile","Image file"))
    , infoFile( initData(&infoFile,"infoFile","Info file"))
    , voxelSize ( initData ( &voxelSize, Vec3d ( 0,0,0 ), "voxelSize", "Voxel size" ) )
    , origin ( initData ( &origin, Vec3d ( 0,0,0 ), "origin", "Grid origin" ) )
    , dimension ( initData ( &dimension, Vec3i ( 0,0,0 ), "dimension", "Grid dimensions" ) )
{
}

template<class MaterialTypes, typename voxelType>
void GridMaterial< MaterialTypes,voxelType>::init()
{

    /*  vector<VoxelGridLoader*> vg;
      sofa::core::objectmodel::BaseContext* context=  this->getContext();
      context->get<VoxelGridLoader>( &vg, core::objectmodel::BaseContext::Local);
      assert(vg.size()>0);
      this->voxelGridLoader = vg[0];
    if ( !this->voxelGridLoader )
    {
    	serr << "VoxelGrid component not found" << sendl;
    	this->nbVoxels = 0;
    	this->data = NULL;
    	this->segmentID = NULL;
    }
    else
    {
    	this->voxelGridLoader->getVoxelSize(this->voxelSize);
    	this->voxelGridLoader->getResolution(dimension.getValue());
    	// this->voxelGridLoader->getorigin(this->origin); // TO DO : update voxelGridLoader to add offsets (temporary solution: use a data Origin)
    	this->nbVoxels = dimension.getValue()[0]*dimension.getValue()[1]*dimension.getValue()[2];
    	this->data = this->voxelGridLoader->getData();
    	this->segmentID = this->voxelGridLoader->getSegmentID();
    }
    */

    bool writeinfos=false; if(!loadInfos()) writeinfos=true;
    loadImage();
    this->nbVoxels = dimension.getValue()[0]*dimension.getValue()[1]*dimension.getValue()[2];
    if(writeinfos) saveInfos();

    VecVec3 points;
    //points.push_back(Vec3(-0.2,0.9,0.1));	points.push_back(Vec3(-0.7,1.3,0.2 ));
    //points.push_back(Vec3(0.732348,0,0.316815));points.push_back(Vec3(-0.534989,-0.661314,-0.479452));points.push_back(Vec3(-0.534955,0.661343,-0.479452));points.push_back(Vec3(0.257823,-0.46005,-0.527559));points.push_back(Vec3(0.257847,0.460036,-0.527559));points.push_back(Vec3(-0.865567,-0.017041,-0.206655));points.push_back(Vec3(-0.15,0,0.2));
    computeUniformSampling(points,true,10,100); //TEST
    computeLinearWeightsInVoronoi ( points[0], true,2.);
    //  HeatDiffusion( points, 0,false,2000);
    //  computeGeodesicalDistances(points,false);

    Inherited::init();
}

// WARNING : The strain is defined as exx, eyy, ezz, 2eyz, 2ezx, 2exy
template<class MaterialTypes, typename voxelType>
void GridMaterial< MaterialTypes,voxelType>::computeStress  ( VecStr& /*stress*/, VecStrStr* /*stressStrainMatrices*/, const VecStr& /*strain*/, const VecStr& )
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
template<class MaterialTypes, typename voxelType>
void GridMaterial< MaterialTypes,voxelType>::computeStress  ( VecElStr& /*stress*/, VecStrStr* /*stressStrainMatrices*/, const VecElStr& /*strain*/, const VecElStr& )
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

template < class MaterialTypes, typename voxelType >
double GridMaterial<MaterialTypes,voxelType>::getStiffness(const voxelType label)
{
// TO DO: add transfer function based on data
    return (double)label;
}

template < class MaterialTypes, typename voxelType >
double GridMaterial<MaterialTypes,voxelType>::getDensity(const voxelType label)
{
// TO DO: add transfer function based on data
    return (double)label;
}

/*************************/
/*   IO   			      */
/*************************/

template < class MaterialTypes, typename voxelType >
bool GridMaterial<MaterialTypes,voxelType>::loadInfos()
{
    std::ifstream fileStream (infoFile.getFullPath().c_str(), std::ifstream::in);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFile << sendl;
        return false;
    }
    Vec3i& dim = *this->dimension.beginEdit();		fileStream >> dim;			this->dimension.endEdit();
    Vec3d& origin = *this->origin.beginEdit();		fileStream >> origin;		this->origin.endEdit();
    Vec3d& voxelsize = *this->voxelSize.beginEdit();  fileStream >> voxelsize;	this->voxelSize.endEdit();
    fileStream.close();
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::saveInfos()
{
    std::ofstream fileStream (infoFile.getFullPath().c_str(), std::ofstream::out);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << infoFile << sendl;
        return false;
    }
    std::cout << "Writing info file " << infoFile << std::endl;
    fileStream << dimension.getValue() << " " << origin.getValue() << " " << voxelSize.getValue() << std::endl;
    fileStream.close();
    return true;
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::loadImage()
{
    grid.load_raw(imageFile.getFullPath().c_str(),dimension.getValue()[0],dimension.getValue()[1],dimension.getValue()[2]);
    if(grid.size()==0)
    {
        serr << "Can not open " << imageFile << sendl;
        return false;
    }
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::saveImage()
{
    grid.save_raw(imageFile.getFullPath().c_str());
    return true;
}


/*************************/
/*   Lumping			  */
/*************************/

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::LumpMass(const Vec3& point,double& mass)
{
    if(!nbVoxels) {mass=0; return false;}
    double voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if(voronoi.size()!=nbVoxels) {mass=voxelvolume*getDensity(grid.data()[index]); return false;} // no voronoi -> 1 point = 1 voxel
    if(voronoi[index]==-1) {mass=voxelvolume*getDensity(grid.data()[index]); return false;} // no voronoi -> 1 point = 1 voxel
    mass=0; for(unsigned int i=0; i<nbVoxels; i++) if(voronoi[i]==voronoi[index]) mass+=voxelvolume*getDensity(grid.data()[i]);
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::LumpVolume(const Vec3& point,double& vol)
{
    if(!nbVoxels) {vol=0; return false;}
    double voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if(voronoi.size()!=nbVoxels) {vol=voxelvolume; return false;} // no voronoi -> 1 point = 1 voxel
    if(voronoi[index]==-1) {vol=voxelvolume; return false;} // no voronoi -> 1 point = 1 voxel
    vol=0; for(unsigned int i=0; i<nbVoxels; i++) if(voronoi[i]==voronoi[index]) vol+=voxelvolume;
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::LumpMoments(const Vec3& point,const unsigned int order,VD& moments)
{
    if(!nbVoxels) {moments.clear(); return false;}
    unsigned int dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    unsigned int i,j;
    moments.resize(dim); for(i=0; i<dim; i++) moments[i]=0;

    double voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    int index=getIndex(point);
    if(voronoi.size()!=nbVoxels) {moments[0]=voxelvolume; return false;} // no voronoi -> 1 point = 1 voxel
    if(voronoi[index]==-1) {moments[0]=voxelvolume; return false;} // no voronoi -> 1 point = 1 voxel

    Vec3 P,G; getCoord(index,P);
    VD momentPG;
    for(i=0; i<nbVoxels; i++)
        if(voronoi[i]==voronoi[index])
        {
            getCoord(i,G);
            getCompleteBasis(G-P,order,momentPG);
            for(j=0; j<dim; j++) moments[j]+=momentPG[j]*voxelvolume;
        }
    return true;  // 1 point = voxels in its voronoi region
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::LumpMomentsStiffness(const Vec3& point,const unsigned int order,VD& moments)
{
    if(!nbVoxels) {moments.clear(); return false;}
    LumpMoments(point,order,moments);
    unsigned int dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    int index=getIndex(point);
    if(voronoi.size()!=nbVoxels) {moments[0]*=getStiffness(grid.data()[index]); return false;} // no voronoi -> 1 point = 1 voxel
    if(voronoi[index]==-1) {moments[0]*=getStiffness(grid.data()[index]); return false;} // no voronoi -> 1 point = 1 voxel

    unsigned int i,j;
    for(i=0; i<nbVoxels; i++)
        if(voronoi[i]==voronoi[index])
            for(j=0; j<dim; j++) moments[j]*=getStiffness(grid.data()[i]);

    return true;
}


template < class MaterialTypes, typename voxelType >
void GridMaterial< MaterialTypes,voxelType >::getCompleteBasis(const Vec3& p,const unsigned int order,VD& basis)
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'
    basis.resize(dim); for(j=0; j<dim; j++) basis[j]=0;

    Vec3 p2; for(j=0; j<3; j++) p2[j]=p[j]*p[j];
    Vec3 p3; for(j=0; j<3; j++) p3[j]=p2[j]*p[j];

    count=0;
// order 0
    basis[count]=1; count++;
    if(count==dim) return;
// order 1
    for(j=0; j<3; j++) {basis[count]=p[j]; count++;}
    if(count==dim) return;
// order 2
    for(j=0; j<3; j++) for(k=j; k<3; k++) {basis[count]=p[j]*p[k];   count++;}
    if(count==dim) return;
// order 3
    basis[count]=p[0]*p[1]*p[2]; count++;
    for(j=0; j<3; j++) for(k=0; k<3; k++) {basis[count]=p2[j]*p[k];  count++;}
    if(count==dim) return;
// order 4
    for(j=0; j<3; j++) for(k=j; k<3; k++) {basis[count]=p2[j]*p2[k]; count++;}
    basis[count]=p2[0]*p[1]*p[2]; count++;
    basis[count]=p[0]*p2[1]*p[2]; count++;
    basis[count]=p[0]*p[1]*p2[2]; count++;
    for(j=0; j<3; j++) for(k=0; k<3; k++) if(j!=k) {basis[count]=p3[j]*p[k];  count++;}
    if(count==dim) return;

    return; // order>4 not implemented...
}



/*************************/
/*   Compute distances   */
/*************************/

template < class MaterialTypes, typename voxelType >
double GridMaterial< MaterialTypes,voxelType >::getDistance(const unsigned int& index1,const unsigned int& index2,const bool biasDistances)
{
    if(!nbVoxels) return -1;
    Vec3 coord1; if(!getCoord(index1,coord1)) return -1; // point1 not in grid
    Vec3 coord2; if(!getCoord(index2,coord2)) return -1; // point2 not in grid
    if(biasDistances)
    {
        double meanstiff=(getStiffness(grid.data()[index1])+getStiffness(grid.data()[index2]))/2.;
        return (double)(coord2-coord1).norm()/meanstiff;
    }
    else return (double)(coord2-coord1).norm();
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeGeodesicalDistances ( const Vec3& point, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    int index=getIndex(point);
    return computeGeodesicalDistances (index, biasDistances, distMax );
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeGeodesicalDistances ( const int& index, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    unsigned int i,index1,index2;
    distances.resize(this->nbVoxels);
    for(i=0; i<this->nbVoxels; i++) distances[i]=distMax;

    if(index<0 || index>=(int)nbVoxels) return false; // voxel out of grid
    if(!grid.data()[index]) return false;	// voxel out of object

    VUI neighbors;
    double d;
    std::queue<int> fifo;
    distances[index]=0; fifo.push(index);
    while(!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for(i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if(grid.data()[index2]) // test if voxel is not void
            {
                d=distances[index1]+getDistance(index1,index2,biasDistances);
                if(distances[index2]>d)
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


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeGeodesicalDistances ( const VecVec3& points, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    VI indices;
    for(unsigned int i=0; i<points.size(); i++) indices.push_back(getIndex(points[i]));
    return computeGeodesicalDistances ( indices, biasDistances, distMax );
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeGeodesicalDistances ( const VI& indices, const bool biasDistances, const double distMax )
{
    if(!nbVoxels) return false;
    unsigned int i,nbi=indices.size(),index1,index2;
    distances.resize(this->nbVoxels); voronoi.resize(this->nbVoxels);
    for(i=0; i<this->nbVoxels; i++) {distances[i]=distMax; voronoi[i]=-1;}

    VUI neighbors;
    double d;

    std::queue<unsigned int> fifo;
    for(i=0; i<nbi; i++) if(indices[i]>=0 && indices[i]<(int)nbVoxels) if(grid.data()[indices[i]]!=0) {distances[indices[i]]=0; voronoi[indices[i]]=i; fifo.push(indices[i]);}
    if(fifo.empty()) return false; // all input voxels out of grid
    while(!fifo.empty())
    {
        index1=fifo.front();
        get26Neighbors(index1, neighbors);
        for(i=0; i<neighbors.size(); i++)
        {
            index2=neighbors[i];
            if(grid.data()[index2]) // test if voxel is not void
            {
                d=distances[index1]+getDistance(index1,index2,biasDistances);
                if(distances[index2]>d)
                {
                    distances[index2]=d; voronoi[index2]=voronoi[index1];
                    fifo.push(index2);
                }
            }
        }
        fifo.pop();
    }
    return true;
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeUniformSampling ( VecVec3& points, const bool biasDistances,const unsigned int num_points, const unsigned int max_iterations )
{
    if(!nbVoxels) return false;
    unsigned int i,k,initial_num_points=points.size();
    VI indices((int)num_points,-1);
    for(i=0; i<initial_num_points; i++) indices[i]=getIndex(points[i]);
    points.resize(num_points);

// initialization: farthest point sampling (see [adams08])
    double dmax; int indexmax;
    if(initial_num_points==0) {indices[0]=0; while(grid.data()[indices[0]]==0) {indices[0]++; if(indices[0]==(int)nbVoxels) return false;} } // take the first not empty voxel as a random point
    for(i=initial_num_points; i<num_points; i++)
    {
        if(i==0) i=1; // a random point has been inserted
        // get farthest point from all inserted points
        computeGeodesicalDistances(indices,biasDistances);
        dmax=-1; indexmax=-1; for(k=0; k<nbVoxels; k++) {if(distances[k]>dmax && voronoi[k]!=-1) {dmax=distances[k]; indexmax=k;}}
        if(indexmax==-1) {return false; }// unable to add point
        indices[i]=indexmax;
    }
// Lloyd relaxation
    Vec3 pos,u,pos_point,pos_voxel;
    unsigned int count,nbiterations=0;
    bool ok=false,ok2;
    double d,dmin; int indexmin;

    while(!ok && nbiterations<max_iterations)
    {
        ok2=true;
        computeGeodesicalDistances(indices,biasDistances); // Voronoi
        VB flag((int)nbVoxels,false);
        for(i=initial_num_points; i<num_points; i++) 	// move to centroid of Voronoi cells
        {
            // estimate centroid given the measured distances = p + 1/N sum d(p,pi)*(pi-p)/|pi-p|
            getCoord(indices[i],pos_point);
            pos.fill(0); count=0;
            for(k=0; k<nbVoxels; k++)
                if(voronoi[k]==(int)i)
                {
                    getCoord(k,pos_voxel);
                    u=pos_voxel-pos_point; u.normalize();
                    pos+=u*(Real)distances[k];
                    count++;
                }
            pos/=(Real)count; 		pos+=pos_point;
            // get closest unoccupied point in object
            dmin=1E100; indexmin=-1;
            for(k=0; k<nbVoxels; k++) if(!flag[k]) if(grid.data()[k]!=0) {getCoord(k,pos_voxel); d=(pos-pos_voxel).norm2(); if(d<dmin) {dmin=d; indexmin=k;}}
            flag[indexmin]=true;
            if(indices[i]!=indexmin) {ok2=false; indices[i]=indexmin;}
        }
        ok=ok2; nbiterations++;
    }

// get points from indices
    for(i=initial_num_points; i<num_points; i++)
    {
        getCoord(indices[i],points[i]) ;
    }
    std::cout<<"Lloyd completed in "<<nbiterations<<" iterations"<<std::endl;

    if(nbiterations==max_iterations) return false; // reached max iterations
    return true;
}



template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::computeLinearWeightsInVoronoi ( const Vec3& point, const bool biasDistances, const double factor)
{
    unsigned int i;
    weights.resize(this->nbVoxels);  for(i=0; i<this->nbVoxels; i++)  weights[i]=0;
    if(!this->nbVoxels) return false;
    int index=getIndex(point);
    if(voronoi.size()!=nbVoxels) return false;
    if(voronoi[index]==-1) return false;
    double dmax=0; for(i=0; i<nbVoxels; i++) if(voronoi[i]==voronoi[index]) if(distances[i]>dmax) dmax=distances[i];
    if(dmax==0) return false;
    VD backupdistance; backupdistance.swap(distances);
    computeGeodesicalDistances(point,true,biasDistances);
    for(i=0; i<nbVoxels; i++) if(distances[i]<dmax) weights[i]=1.-distances[i]/dmax;
    backupdistance.swap(distances);
    return true;  // 1 point = voxels its his voronoi region
}


template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::HeatDiffusion( const VecVec3& points, const unsigned int hotpointindex,const bool fixdatavalue,const unsigned int max_iterations,const double precision  )
{
    if(!this->nbVoxels) return false;
    unsigned int i,j,k,num_points=points.size();
    int index;
    weights.resize(this->nbVoxels);  for(i=0; i<this->nbVoxels; i++)  weights[i]=0.5;

    VB isfixed((int)nbVoxels,false);
    VB update((int)nbVoxels,false);
    VUI neighbors;

// intialisation: fix weight of points or regions
    for(i=0; i<num_points; i++)
    {
        index=getIndex(points[i]);
        if(index!=-1)
        {
            isfixed[index]=true;
            if(i==hotpointindex) weights[index]=1; else weights[index]=0;
            get6Neighbors(index, neighbors); for(j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
            if(fixdatavalue) // fix regions
                for(k=0; k<this->nbVoxels; k++)
                    if(grid.data()[k]==grid.data()[index])
                    {
                        isfixed[k]=true; weights[k]=weights[index];
                        get6Neighbors(k, neighbors); for(j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
                    }
        }
    }

// diffuse
    unsigned int nbiterations=0;
    bool ok=false,ok2;

    while(!ok && nbiterations<max_iterations)
    {
        ok2=true;
//	#pragma omp parallel for private(j,neighbors)
        for(i=0; i<this->nbVoxels; i++)
            if(update[i])
            {
                if(isfixed[i]) update[i]=false;
                else
                {
                    double val=0,W=0;
                    //if(aniso) {W+=Update_AnisoDiffw(i,vol); val+=W*n->v;}

                    get6Neighbors(i, neighbors);
                    for(j=0; j<neighbors.size(); j++) {val+=weights[neighbors[j]]; W+=1.; } if(W!=0) val=val/W; // average neighboor values

                    if(fabs(val-weights[i])<precision) update[i]=false;
                    else
                    {
                        weights[i]=val; ok2=false;
                        for(j=0; j<neighbors.size(); j++) update[neighbors[j]]=true;
                    }
                }
            }
        ok=ok2; nbiterations++;
    }
    std::cout<<"Heat diffusion completed in "<<nbiterations<<" iterations"<<std::endl;

//for(i=0;i<this->nbVoxels;i++)  serr<<weights[i]<<sendl;

    if(nbiterations==max_iterations) return false; // reached max iterations
    return true;
}




/*************************/
/*         Draw          */
/*************************/

template<class MaterialTypes, typename voxelType>
void GridMaterial< MaterialTypes,voxelType >::draw()
{
    if ( this->showVoxelData.getValue() || this->showVoronoi.getValue() ||this->showDistances.getValue() ||this->showWeights.getValue() )
    {
        //glDisable ( GL_LIGHTING );
        //glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;

        unsigned int i;
        double s=(voxelSize.getValue()[0]+voxelSize.getValue()[1]+voxelSize.getValue()[2])/3.;
        float defaultcolor[4]= {0.8,0.8,0.8,0.3},color[4];
        bool wireframe=this->getContext()->getShowWireFrame();

        float label,labelmax=-1;

        if(this->showVoxelData.getValue()) {for(i=0; i<nbVoxels; i++) if(grid.data()[i]>labelmax) labelmax=(float)grid.data()[i];}
        else if(voronoi.size()==nbVoxels && this->showVoronoi.getValue()) {for(i=0; i<nbVoxels; i++) if(grid.data()[i]!=0) if(voronoi[i]>labelmax) labelmax=(float)voronoi[i];}
        else if(distances.size()==nbVoxels && this->showDistances.getValue()) {for(i=0; i<nbVoxels; i++) if(grid.data()[i]!=0) if(distances[i]>labelmax) labelmax=(float)distances[i];}
        else if(weights.size()==nbVoxels && this->showWeights.getValue()) labelmax=1;

        cimg_forXYZ(grid,x,y,z)
        {
            if(grid(x,y,z)!=0)
            {
                label=-1;
                if(this->showVoxelData.getValue()) label=(float)grid(x,y,z);
                else if(voronoi.size()==nbVoxels && this->showVoronoi.getValue())  label=(float)voronoi[getIndex(Vec3i(x,y,z))];
                else if(distances.size()==nbVoxels && this->showDistances.getValue())  label=(float)distances[getIndex(Vec3i(x,y,z))];
                else if(weights.size()==nbVoxels && this->showWeights.getValue())  label=(float)weights[getIndex(Vec3i(x,y,z))];

                if(label==-1) {glColor4fv(defaultcolor); glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,defaultcolor);}
                else
                {
                    sofa::helper::gl::Color::setHSVA(240.*label/labelmax,1.,.8,defaultcolor[3]);
                    glGetFloatv(GL_CURRENT_COLOR, color); glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
                }
                Vec3 coord; getCoord(Vec3i(x,y,z),coord);
                glTranslated (coord[0],coord[1],coord[2]);
                drawCube(s,wireframe);
                glTranslated (-coord[0],-coord[1],-coord[2]);
                //GlText::draw ( (int), coord, showTextScaleFactor.getValue() );
            }
        }
    }
}

template < class MaterialTypes, typename voxelType >
void GridMaterial< MaterialTypes,voxelType >::drawCube(double size,bool wireframe)
{
    double ss2=size/2.;
    if(!wireframe) glBegin(GL_QUADS);
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(1,0,0);   glVertex3d(ss2,-ss2,-ss2);         glNormal3f(1,0,0);   glVertex3d(ss2,-ss2,ss2);         glNormal3f(1,0,0);   glVertex3d(ss2,ss2,ss2);         glNormal3f(1,0,0);   glVertex3d(ss2,ss2,-ss2); if(wireframe) glEnd ();
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(-1,0,0);  glVertex3d(-ss2,-ss2,-ss2);        glNormal3f(-1,0,0);  glVertex3d(-ss2,-ss2,ss2);        glNormal3f(-1,0,0);  glVertex3d(-ss2,ss2,ss2);        glNormal3f(-1,0,0);  glVertex3d(-ss2,ss2,-ss2); if(wireframe) glEnd ();
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(0,1,0);   glVertex3d(-ss2,ss2,-ss2);         glNormal3f(0,1,0);   glVertex3d(ss2,ss2,-ss2);         glNormal3f(0,1,0);   glVertex3d(ss2,ss2,ss2);         glNormal3f(0,1,0);   glVertex3d(-ss2,ss2,ss2); if(wireframe) glEnd ();
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(0,-1,0);  glVertex3d(-ss2,-ss2,-ss2);        glNormal3f(0,-1,0);  glVertex3d(ss2,-ss2,-ss2);        glNormal3f(0,-1,0);  glVertex3d(ss2,-ss2,ss2);        glNormal3f(0,-1,0);  glVertex3d(-ss2,-ss2,ss2); if(wireframe) glEnd ();
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(0,0,1);   glVertex3d(-ss2,-ss2,ss2);         glNormal3f(0,0,1);   glVertex3d(-ss2,ss2,ss2);         glNormal3f(0,0,1);   glVertex3d(ss2,ss2,ss2);         glNormal3f(0,0,1);   glVertex3d(ss2,-ss2,ss2); if(wireframe) glEnd ();
    if(wireframe) glBegin(GL_LINE_LOOP);  glNormal3f(0,0,-1);  glVertex3d(-ss2,-ss2,-ss2);        glNormal3f(0,0,-1);  glVertex3d(-ss2,ss2,-ss2);        glNormal3f(0,0,-1);  glVertex3d(ss2,ss2,-ss2);        glNormal3f(0,0,-1);  glVertex3d(ss2,-ss2,-ss2); if(wireframe) glEnd ();
    if(!wireframe) glEnd ();
}


/*************************/
/*         Utils         */
/*************************/
template < class MaterialTypes, typename voxelType >
int GridMaterial< MaterialTypes,voxelType >::getIndex(const Vec3i& icoord)
{
    if(!nbVoxels) return -1;
    for(int i=0; i<3; i++) if(icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return -1; // invalid icoord (out of grid)
    return icoord[0]+dimension.getValue()[0]*icoord[1]+dimension.getValue()[0]*dimension.getValue()[1]*icoord[2];
}

template < class MaterialTypes, typename voxelType >
int GridMaterial< MaterialTypes,voxelType >::getIndex(const Vec3& coord)
{
    if(!nbVoxels) return -1;
    Vec3i icoord;
    if(!getiCoord(coord,icoord)) return -1;
    return getIndex(icoord);
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::getiCoord(const Vec3& coord, Vec3i& icoord)
{
    if(!nbVoxels) return false;
    Real val;
    for(unsigned int i=0; i<3; i++)
    {
        val=(coord[i]-(Real)origin.getValue()[i])/(Real)voxelSize.getValue()[i];
        val=((val-floor(val))<0.5)?floor(val):ceil(val); //round
        if(val<0 || val>=dimension.getValue()[i]) return false;
        icoord[i]=(int)val;
    }
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::getiCoord(const int& index, Vec3i& icoord)
{
    if(!nbVoxels) return false;
    if(index<0 || index>=(int)nbVoxels) return false;  // invalid index
    icoord[2]=index/(dimension.getValue()[0]*dimension.getValue()[1]);
    icoord[1]=(index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1])/dimension.getValue()[0];
    icoord[0]=index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1]-icoord[1]*dimension.getValue()[0];
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::getCoord(const Vec3i& icoord, Vec3& coord)
{
    if(!nbVoxels) return false;
    for(unsigned int i=0; i<3; i++) if(icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return false; // invalid icoord (out of grid)
    coord=this->origin.getValue();
    for(unsigned int i=0; i<3; i++) coord[i]+=(Real)this->voxelSize.getValue()[i]*(Real)icoord[i];
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::getCoord(const int& index, Vec3& coord)
{
    if(!nbVoxels) return false;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    else return getCoord(icoord,coord);
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::get6Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::get18Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }

    for(unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]-=2;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    return true;
}

template < class MaterialTypes, typename voxelType >
bool GridMaterial< MaterialTypes,voxelType >::get26Neighbors ( const int& index, VUI& neighbors )
{
    neighbors.clear();
    if(!nbVoxels) return false;
    int i;
    Vec3i icoord;
    if(!getiCoord(index,icoord)) return false;
    for(unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
        icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
    }
    for(unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]-=2;
        for(unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]-=1;
            icoord[j]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    icoord[0]+=1;
    icoord[1]+=1;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]-=2;
    icoord[1]+=1;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]-=1;
    icoord[2]-=1; i=getIndex(icoord); if(i!=-1) neighbors.push_back(i); icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]+=1;
    return true;
}


}

} // namespace component

} // namespace sofa

#endif


