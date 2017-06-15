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
#ifndef SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_INL
#define SOFA_COMPONENT_MATERIAL_GRIDMATERIAL_INL

#include <sofa/core/visual/VisualParams.h>
#include "GridMaterial.h"
#include <sofa/helper/gl/Color.h>
#include <sofa/helper/gl/glText.inl>
#include <queue>
#include <list>
#include <string>
//#include <omp.h>
#include <sofa/simulation/Visitor.h>

namespace sofa
{

namespace component
{

namespace material
{
using helper::WriteAccessor;

template<class MaterialTypes>
GridMaterial< MaterialTypes>::GridMaterial()
    : Inherited()
    , maxLloydIterations ( initData ( &maxLloydIterations, (unsigned int)100, "maxLloydIterations", "Maximum iteration number for Lloyd algorithm." ) )
    , voxelSize ( initData ( &voxelSize, SCoord ( 0,0,0 ), "voxelSize", "Voxel size." ) )
    , origin ( initData ( &origin, SCoord ( 0,0,0 ), "origin", "Grid origin." ) )
    , dimension ( initData ( &dimension, GCoord ( 0,0,0 ), "dimension", "Grid dimensions." ) )
    , gridOffset (5)
    , nbVoxels (0)
    , labelToStiffnessPairs ( initData ( &labelToStiffnessPairs, "labelToStiffnessPairs","Correspondances between grid value and material stiffness." ) )
    , labelToDensityPairs ( initData ( &labelToDensityPairs, "labelToDensityPairs","Correspondances between grid value and material density." ) )
    , labelToBulkModulusPairs ( initData ( &labelToBulkModulusPairs, "labelToBulkModulusPairs","Correspondances between grid value and material bulk modulus." ) )
    , labelToPoissonRatioPairs ( initData ( &labelToPoissonRatioPairs, "labelToPoissonRatioPairs","Correspondances between grid value and material Poisson Ratio." ) )
    , imageFile( initData(&imageFile,"imageFile","Image file."))
    , weightFile( initData(&weightFile,"weightFile","Voxel weight file."))
    , bulkModulus ( initData ( &bulkModulus, "bulkModulus","Sample bulk Modulus." ) )
    , stiffness ( initData ( &stiffness, "stiffness","Sample stiffness." ) )
    , density ( initData ( &density, "density","Sample density." ) )
    , poissonRatio ( initData ( &poissonRatio, "poissonRatio","Sample poisson Ratio." ) )
    , distanceType ( initData ( &distanceType,"distanceType","Distance measure." ) )
    , biasDistances ( initData ( &biasDistances,true, "biasDistances","Bias distances according to stiffness." ) )
    , distanceBiasFactor ( initData ( &distanceBiasFactor,(Real)1., "distanceBiasFactor","Exponent factor of the stiffness during frame sampling (0=uniform sampling, 1=compliance distance-based sampling)." ) )
    , useDijkstra ( initData ( &useDijkstra, true, "useDijkstra","Use Dijkstra's algorithm to compute the distance fields." ) )
    , useDistanceScaleFactor ( initData ( &useDistanceScaleFactor,false, "useDistanceScaleFactor","useDistanceScaleFactor." ) )
    , weightSupport ( initData ( &weightSupport,(Real)2., "weightSupport","Support of the weight function (2=interpolating, >2 = approximating)." ) )
    , nbVoronoiSubdivisions ( initData ( &nbVoronoiSubdivisions,(unsigned int)0, "nbVoronoiSubdivisions","number of subdvisions of the voronoi during weight computation (1 by default)." ) )
    , verbose ( initData ( &verbose, false, "verbose","Set the verbose mode" ) )
    , voxelsHaveChanged ( initData ( &voxelsHaveChanged, false, "voxelsHaveChanged","Voxels Have Changed." ) )
    , showVoxels ( initData ( &showVoxels, "showVoxelData","Show voxel data." ) )
    , showWeightIndex ( initData ( &showWeightIndex, ( unsigned int ) 0, "showWeightIndex","Weight index." ) )
    , showPlane ( initData ( &showPlane, GCoord ( -1,-1,-1 ), "showPlane","Indices of slices to be shown." ) )
    , show3DValuesHeight ( initData ( &show3DValuesHeight, (float)0.0, "show3DValuesHeight","When show plane is activated, values are displayed in 3D." ) )
    , vboSupported ( initData ( &vboSupported, true, "vboSupported","Allow to disply 2.5D functions." ) )
    , vboValuesId1(0)
    , vboValuesId2(0)
{
    voxelsHaveChanged.setDisplayed (false);

    helper::OptionsGroup distanceTypeOptions(3,"Geodesic", "HeatDiffusion", "AnisotropicHeatDiffusion");
    distanceTypeOptions.setSelectedItem(DISTANCE_GEODESIC);
    distanceType.setValue(distanceTypeOptions);

    helper::OptionsGroup showVoxelsOptions(12,"None", "Data", "Stiffness", "Density", "Bulk modulus", "Poisson ratio", "Voronoi Samples", "Voronoi Frames", "Distances", "Weights","Linearity Error", "Frames Indices");
    showVoxelsOptions.setSelectedItem(SHOWVOXELS_NONE);
    showVoxels.setValue(showVoxelsOptions);
}


template<class MaterialTypes>
GridMaterial< MaterialTypes >::~GridMaterial()
{
    deleteVBO(vboValuesId2);
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
    showederror=-1;
    showWireframe=false;
    biasFactor=(Real)1.;

    //TEST
    /*
    if(v_weights.size()!=nbVoxels)
    {

    VecSCoord points;	  points.push_back(SCoord(0.8,0,0.3));	  points.push_back(SCoord(-0.534989,-0.661314,-0.58));	  points.push_back(SCoord(-0.534955,0.661343,-0.58));	  points.push_back(SCoord(0.257823,-0.46005,-0.63));	  points.push_back(SCoord(0.257847,0.460036,-0.63));	  points.push_back(SCoord(-0.15,0,0.2 ));
    computeUniformSampling(points,6);
    computeWeights(points);

    VecSCoord samples;
    computeUniformSampling(samples,50);

    //	  Vec<nbRef,unsigned int> reps; vector<Real> w; VecSGradient dw; VecSHessian ddw;
    // for(unsigned int j=0;j<samples.size();j++) lumpWeightsRepartition(samples[j],reps,w,&dw,&ddw);
    }*/
    ////

    initVBO();
    genListCube();

    Inherited::init();
}


template<class MaterialTypes>
void GridMaterial< MaterialTypes>::reinit()
{
    updateSampleMaterialProperties();
    updateMaxValues();
}

template<class MaterialTypes>
void GridMaterial<MaterialTypes>::getStressStrainMatrix( StrStr& materialMatrix, const MaterialCoord& point ) const
{

    GCoord gcoord;
    getiCoord( point, gcoord);
    int voxelIndex = getIndex(gcoord);
    Real young = this->getStiffness(grid.data()[voxelIndex]);
    Real poisson = this->getPoissonRatio(grid.data()[voxelIndex]);
//                            cerr<<"GridMaterial<MaterialTypes>::getStressStrainMatrix, point = "<<point <<", gcoord = "<< gcoord << ", young = " << young << endl;


    materialMatrix[0][0] = materialMatrix[1][1] = materialMatrix[2][2] = 1;
    materialMatrix[0][1] = materialMatrix[0][2] = materialMatrix[1][0] =
            materialMatrix[1][2] = materialMatrix[2][0] = materialMatrix[2][1] = poisson/(1-poisson);
    materialMatrix[0][3] = materialMatrix[0][4] = materialMatrix[0][5] = 0;
    materialMatrix[1][3] = materialMatrix[1][4] = materialMatrix[1][5] = 0;
    materialMatrix[2][3] = materialMatrix[2][4] = materialMatrix[2][5] = 0;
    materialMatrix[3][0] = materialMatrix[3][1] = materialMatrix[3][2] =
            materialMatrix[3][4] = materialMatrix[3][5] = 0;
    materialMatrix[4][0] = materialMatrix[4][1] = materialMatrix[4][2] =
            materialMatrix[4][3] = materialMatrix[4][5] = 0;
    materialMatrix[5][0] = materialMatrix[5][1] = materialMatrix[5][2] =
            materialMatrix[5][3] = materialMatrix[5][4] = 0;
    materialMatrix[3][3] = materialMatrix[4][4] = materialMatrix[5][5] =
            (1-2*poisson)/(2*(1-poisson));
    materialMatrix *= (young*(1-poisson))/((1+poisson)*(1-2*poisson));
}


// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStress  ( VecStrain1& stresses, VecStrStr* stressStrainMatrices, const VecStrain1& strains, const VecStrain1& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio,youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        Str& stress = stresses[i][0];
        const Str& strain = strains[i][0];

        stress = this->hookeStress( strain, stressDiagonal, stressOffDiagonal, shear );

        if( stressStrainMatrices != NULL )
        {
            this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
        }
    }
}


// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStress  ( VecStrain4& stresses, VecStrStr* stressStrainMatrices, const VecStrain4& strains, const VecStrain4& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio,youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        for(unsigned int j=0; j<4; j++ )
        {
            Str& stress = stresses[i][j];
            const Str& strain = strains[i][j];

            stress = this->hookeStress( strain, stressDiagonal, stressOffDiagonal, shear );

            if( stressStrainMatrices != NULL )
            {
                this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
            }
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStress  ( VecStrain10& stresses, VecStrStr* stressStrainMatrices, const VecStrain10& strains, const VecStrain10& /*strainRates*/, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio, youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        for(unsigned int j=0; j<10; j++ )
        {
            Str& stress = stresses[i][j];
            const Str& strain = strains[i][j];

            stress = this->hookeStress( strain, stressDiagonal, stressOffDiagonal, shear );

            if( stressStrainMatrices != NULL )
            {
                this->fillHookeMatrix( (*stressStrainMatrices)[i], stressDiagonal, stressOffDiagonal,  shear );
            }
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStressChange  ( VecStrain1& stresses, const VecStrain1& strains, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio,youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        for(unsigned int j=0; j<1; j++ )
        {

            stresses[i][j] = this->hookeStress( strains[i][j], stressDiagonal, stressOffDiagonal, shear );
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStressChange  ( VecStrain4& stresses, const VecStrain4& strains, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio,youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        for(unsigned int j=0; j<4; j++ )
        {

            stresses[i][j] = this->hookeStress( strains[i][j], stressDiagonal, stressOffDiagonal, shear );
        }
    }
}

// WARNING : The strain is defined as exx, eyy, ezz, exy, eyz, ezx
template<class MaterialTypes>
void GridMaterial< MaterialTypes>::computeStressChange  ( VecStrain10& stresses, const VecStrain10& strains, const VecMaterialCoord& /*point*/  )
{
    Real stressDiagonal, stressOffDiagonal, shear, poissRatio,youngModulus;

    for( unsigned int i=0; i<stresses.size(); i++ )
    {
        if(poissonRatio.getValue().size()>i) poissRatio= poissonRatio.getValue()[i]; else poissRatio=0;
        if(stiffness.getValue().size()>i) youngModulus= stiffness.getValue()[i]; else youngModulus=1;

        Real f = youngModulus/((1 + poissRatio)*(1 - 2 * poissRatio));
        stressDiagonal = f * (1 - poissRatio);
        stressOffDiagonal = poissRatio * f;
        shear = f * (1 - 2 * poissRatio);

        for(unsigned int j=0; j<10; j++ )
        {

            stresses[i][j] = this->hookeStress( strains[i][j], stressDiagonal, stressOffDiagonal, shear );
        }
    }
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::updateSampleMaterialProperties()
{
    if (!nbVoxels) return ;
    if (voronoi.size()!=nbVoxels) return ;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Update_Sample_Material_Properties");
#endif
    clock_t timer = clock();

    unsigned int nbsamples=0;
    for(unsigned int i=0; i<nbVoxels; i++)
        if(voronoi[i]!=-1) if((unsigned int)voronoi[i]>nbsamples) nbsamples=(unsigned int)voronoi[i];
    nbsamples++; // indices in voronoi start from 0

    WriteAccessor<Data<vector<Real> > >  m_bulkModulus  ( bulkModulus );	m_bulkModulus.resize(nbsamples);
    WriteAccessor<Data<vector<Real> > >  m_stiffness  ( stiffness );		m_stiffness.resize(nbsamples);
    WriteAccessor<Data<vector<Real> > >  m_density  ( density );			m_density.resize(nbsamples);
    WriteAccessor<Data<vector<Real> > >  m_poissonRatio  ( poissonRatio );	m_poissonRatio.resize(nbsamples);

    for(unsigned int sampleindex=0; sampleindex<nbsamples; sampleindex++)
    {
        unsigned int count=0;
        m_bulkModulus[sampleindex]=0;
        m_stiffness[sampleindex]=0;
        m_density[sampleindex]=0;
        m_poissonRatio[sampleindex]=0;

        for(unsigned int i=0; i<nbVoxels; i++)
            if((unsigned int)voronoi[i]==sampleindex)
            {
                m_bulkModulus[sampleindex]+=getBulkModulus(grid.data()[i]);
                m_stiffness[sampleindex]+=getStiffness(grid.data()[i]);;
                m_density[sampleindex]+=getDensity(grid.data()[i]);;
                m_poissonRatio[sampleindex]+=getPoissonRatio(grid.data()[i]);
                count++;
            }
        if(count!=0)
        {
            m_bulkModulus[sampleindex]/=(Real)count;
            m_stiffness[sampleindex]/=(Real)count;
            m_density[sampleindex]/=(Real)count;
            m_poissonRatio[sampleindex]/=(Real)count;
        }
    }

    if (verbose.getValue())
        std::cout << "INITTIME updateSampleMaterialProperties: " << (clock() - timer) / (float)CLOCKS_PER_SEC << " sec." << std::endl;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Sample_Material_Properties");
#endif

}


template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial<MaterialTypes>::getBulkModulus(const unsigned int sampleindex) const
{
    if (bulkModulus.getValue().size()>sampleindex) return bulkModulus.getValue()[sampleindex]; else return 0;
}



template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial<MaterialTypes>::getStiffness(const voxelType label) const
{
    if(label==0) return (Real)0;

    const mapLabelType& pairs = labelToStiffnessPairs.getValue();
    if (pairs.size()==0) return (Real)1; // no map defined -> return 1

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if ((Real)pit->first>(Real)label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(((Real)pit->first-(Real)label)/((Real)pit->first-(Real)mit->first));
                return alpha*vlow+(1.-alpha)*vup;
            }
        }
        mit=pit;
    }
    return mit->second;
}

template < class MaterialTypes>
typename GridMaterial<MaterialTypes>::Real GridMaterial<MaterialTypes>::getDensity(const voxelType label) const
{
    if(label==0) return (Real)0;

    const mapLabelType& pairs = labelToDensityPairs.getValue();
    if (pairs.size()==0) return (Real)1; // no map defined -> return 1

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if ((Real)pit->first>(Real)label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(((Real)pit->first-(Real)label)/((Real)pit->first-(Real)mit->first));
                return alpha*vlow+(1.-alpha)*vup;
            }
        }
        mit=pit;
    }
    return mit->second;
}

template < class MaterialTypes>
typename GridMaterial<MaterialTypes>::Real GridMaterial<MaterialTypes>::getBulkModulus(const voxelType label) const
{
    if(label==0) return (Real)0;

    const mapLabelType& pairs = labelToBulkModulusPairs.getValue();
    if (pairs.size()==0) return (Real)0; // no map defined -> return 0

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if ((Real)pit->first>(Real)label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(((Real)pit->first-(Real)label)/((Real)pit->first-(Real)mit->first));
                return alpha*vlow+(1.-alpha)*vup;
            }
        }
        mit=pit;
    }
    return mit->second;
}

template < class MaterialTypes>
typename GridMaterial<MaterialTypes>::Real GridMaterial<MaterialTypes>::getPoissonRatio(const voxelType label) const
{
    if(label==0) return (Real)0;

    const mapLabelType& pairs = labelToPoissonRatioPairs.getValue();
    if (pairs.size()==0) return (Real)0; // no map defined -> return 0

    typename mapLabelType::const_iterator mit;
    for (typename mapLabelType::const_iterator pit=pairs.begin(); pit!=pairs.end(); pit++)
    {
        if ((Real)pit->first>(Real)label)
        {
            if (pit==pairs.begin()) return pit->second;
            else
            {
                Real vlow=mit->second,vup=pit->second;
                Real alpha=(((Real)pit->first-(Real)label)/((Real)pit->first-(Real)mit->first));
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
    if (sofa::helper::system::DataRepository.findFile(infoFile)) // If the file is existing
    {
        infoFile=sofa::helper::system::DataRepository.getFile(infoFile);
        std::ifstream fileStream (infoFile.c_str(), std::ifstream::in);
        if (!fileStream.is_open())
        {
            serr << "Can not open " << infoFile << sendl;
            return false;
        }
        std::string str;
        fileStream >> str;	char vtype[32]; fileStream.getline(vtype,32); // voxeltype not used yet
        fileStream >> str; GCoord& dim = *this->dimension.beginEdit();       fileStream >> dim;      this->dimension.endEdit();
        fileStream >> str; SCoord& origin = *this->origin.beginEdit();       fileStream >> origin;   this->origin.endEdit();
        fileStream >> str; SCoord& voxelsize = *this->voxelSize.beginEdit(); fileStream >> voxelsize; this->voxelSize.endEdit();
        fileStream.close();
        std::cout << "GridMaterial, Loaded info file "<< infoFile <<", dim = "<< dimension.getValue() <<", origin = " << this->origin.getValue() <<", voxelSize = " << this->voxelSize.getValue() << std::endl;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::saveInfos()
{
    if (!infoFile.size()) return false;
    std::ofstream fileStream (infoFile.c_str(), std::ofstream::out);
    if (!fileStream.is_open())
    {
        serr << "GridMaterial, Can not open " << infoFile << sendl;
        return false;
    }
    std::cout << "GridMaterial, Writing info file " << infoFile << std::endl;
    fileStream << "voxelType: " << CImg<voxelType>::pixel_type() << std::endl;
    fileStream << "dimensions: " << dimension.getValue() << std::endl;
    fileStream << "origin: " << origin.getValue() << std::endl;
    fileStream << "voxelSize: " << voxelSize.getValue() << std::endl;
    fileStream.close();
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::loadImage()
{
    if (!imageFile.isSet()) return false;

    std::string fName (imageFile.getValue());
    if( strcmp( fName.substr(fName.find_last_of('.')+1).c_str(), "raw") == 0 ||
        strcmp( fName.substr(fName.find_last_of('.')+1).c_str(), "RAW") == 0)
    {
        grid.load_raw(imageFile.getFullPath().c_str(),dimension.getValue()[0],dimension.getValue()[1],dimension.getValue()[2]);
    }
    else
    {
        grid.load(imageFile.getFullPath().c_str());

        // Convert image to black and white to avoid access problems
        cimg_forXY(grid,x,y)
        {
            grid(x,y)=(grid(x,y,0)+grid(x,y,1)+grid(x,y,2))/3.0;
        }

        // Extrud Z dimension of the image to the wanted size.
        if (dimension.getValue()[2] > 1)
            grid.resize(grid.width(), grid.height(), dimension.getValue()[2], grid.spectrum(), 0, 1); // no interpolation, extrude values
    }

    // offset by one voxel to prevent from interpolation outside the grid
    gridOffset = 5;
    GCoord off; off.fill(2*gridOffset);
    dimension.setValue(dimension.getValue()+off);
    origin.setValue(origin.getValue()-voxelSize.getValue()*(Real)gridOffset);

//                cerr<<"GridMaterial< MaterialTypes>::loadImage, values before " << (int)grid(0,0,0) << ", " << (int)grid(1,0,0) << endl;
//                typeof(grid) grid2(grid);
//                cerr<<"GridMaterial< MaterialTypes>::loadImage, values before " << (int)grid2(0,0,0) << ", " << (int)grid2(1,0,0) << endl;
    grid.resize(dimension.getValue()[0],dimension.getValue()[1],dimension.getValue()[2],1,0,0,0.5,0.5,0.5,0.5);
//                for( unsigned i = gridOffset; i<dimension.getValue()[0]-gridOffset; i++ )
//                    for( unsigned j = gridOffset; j<dimension.getValue()[1]-gridOffset; j++ )
//                        for( unsigned k = gridOffset; k<dimension.getValue()[2]-gridOffset; k++ )
//                            grid(i,j,k) = grid2(i-gridOffset,j-gridOffset,k-gridOffset);
//                cerr<<"GridMaterial< MaterialTypes>::loadImage, values after " << (int)grid(5,5,5) << ", " << (int)grid(6,5,5) << endl;
//                cerr<<"GridMaterial< MaterialTypes>::loadImage, values after " << (int)grid(0,0,0) << ", " << (int)grid(1,0,0) << endl;
//                cerr<<"GridMaterial< MaterialTypes>::loadImage, values after " << (int)grid2(0,0,0) << ", " << (int)grid2(1,0,0) << endl;

    if (grid.size()==0)
    {
        serr << "Can not open " << imageFile << sendl;
        return false;
    }
    this->nbVoxels = dimension.getValue()[0]*dimension.getValue()[1]*dimension.getValue()[2];

    unsigned int count=0;
    for(unsigned int i=0; i<this->nbVoxels; i++) if(grid.data()[i]) count++;

    updateMaxValues();
    std::cout << "GridMaterial, Loaded image "<< imageFile <<" of voxel type " << grid.pixel_type() << "( "<<count<<" non empty voxels)"<< std::endl;
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
    if (nbRef!=nbrefs)
    {
        serr << "Invalid nbRef in " << weightFile << sendl;
        return false;
    }

    this->v_index.resize(nbVoxels);
    this->v_weights.resize(nbVoxels);

    for (unsigned int i=0; i<nbVoxels; i++)
        for (unsigned int j=0; j<nbRef; j++)
        {
            fileStream >> v_index[i][j] ;
            fileStream >> v_weights[i][j];
        }
    fileStream.close();
    std::cout << "GridMaterial, Loaded weight file "<< weightFile << std::endl;
    showedrepartition=-1;
    return true;
}



template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::saveWeightRepartion()
{
    if (!weightFile.isSet()) return false;
    if (nbVoxels==0) return false;
    if (v_weights.size()!=nbVoxels) return false;
    if (v_index.size()!=nbVoxels) return false;

    std::ofstream fileStream (weightFile.getFullPath().c_str(), std::ofstream::out);
    if (!fileStream.is_open())
    {
        serr << "Can not open " << weightFile << sendl;
        return false;
    }
    std::cout << "Writing grid weights repartion file " << weightFile << std::endl;

    fileStream << nbVoxels << " " << nbRef << std::endl;
    for (unsigned int i=0; i<nbVoxels; i++)
    {
        for (unsigned int j=0; j<nbRef; j++)
        {
            fileStream << v_index[i][j] << " " << v_weights[i][j] << " ";
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
bool GridMaterial< MaterialTypes>::getWeightedMasses(const unsigned int sampleindex, vector<VRef>& reps, vector<VRefReal>& w, VecSCoord& p,vector<Real>& masses)
{
    p.clear();
    masses.clear();
    reps.clear();
    w.clear();

    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels || v_weights.size()!=nbVoxels || v_index.size()!=nbVoxels) return false; // weights not computed

    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];

    unsigned int i;
    for(i=0; i<nbVoxels; i++) if(voronoi[i]==(int)sampleindex)
        {
            p.push_back(SCoord());  getCoord(i,p.back());
            w.push_back(v_weights[i]);
            reps.push_back(v_index[i]);
            masses.push_back(voxelvolume*getDensity(grid.data()[i]));
        }
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::lumpMass(const unsigned int sampleindex,Real& mass)
{
    mass=0;
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;
    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    for (unsigned int i=0; i<nbVoxels; i++) if (voronoi[i]==(int)sampleindex) mass+=voxelvolume*getDensity(grid.data()[i]);
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::lumpVolume(const unsigned int sampleindex,Real& vol)
{
    vol=0;
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;
    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];
    for (unsigned int i=0; i<nbVoxels; i++) if (voronoi[i]==(int)sampleindex) vol+=voxelvolume;
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeVolumeIntegrationFactors(const unsigned int sampleindex,const SCoord& point,const unsigned int order,vector<Real>& moments)
{
    unsigned int i,j,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'
    moments.resize(dim);	for (i=0; i<dim; i++) moments[i]=0;

    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;
    Real voxelvolume=voxelSize.getValue()[0]*voxelSize.getValue()[1]*voxelSize.getValue()[2];

    SCoord G;
    vector<Real> momentPG;
    for (i=0; i<nbVoxels; i++)
        if (voronoi[i]==(int)sampleindex)
        {
            getCoord(i,G);
            getCompleteBasis(G-point,order,momentPG);
            for (j=0; j<dim; j++) moments[j]+=momentPG[j]*voxelvolume;
            //for (j=0;j<dim;j++) moments[j]+=momentPG[j]*voxelvolume*getStiffness(grid.data()[i]); // for linear materials stiffness can be integrated into the precomputed factors -> need to update compute stress in this case // not use here to be able to modify stiffness through the gui
        }
    return true;
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::lumpWeightsRepartition(const unsigned int sampleindex,const SCoord& point,VRef& reps,VRefReal& w,VRefGradient* dw,VRefHessian* /*ddw*/)
{
    unsigned int i,j,k;
    for (i=0; i<nbRef; i++) {reps[i]=0; w[i]=0;}

    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels || v_weights.size()!=nbVoxels || v_index.size()!=nbVoxels) return false; // weights not computed

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Lump_Weights_Repartition");
#endif

    // get the nbrefs most relevant weights in the voronoi region
    unsigned int maxFrameIndex=0;
    for (i=0; i<nbVoxels; i++) if(voronoi[i]==(int)sampleindex) for (j=0; j<nbRef; j++) { if(v_weights[i][j]!=0) if(v_index[i][j]>maxFrameIndex) maxFrameIndex=v_index[i][j]; }
    vector<Real> W((int)(maxFrameIndex+1),0);
    for (i=0; i<nbVoxels; i++) if(voronoi[i]==(int)sampleindex) for (j=0; j<nbRef; j++) if(v_weights[i][j]!=0) W[v_index[i][j]]+=v_weights[i][j];

    for (i=0; i<maxFrameIndex+1; i++)
    {
        j=0; while (j!=nbRef && w[j]>W[i]) j++;
        if(j!=nbRef)
        {
            for (k=nbRef-1; k>j; k--) {w[k]=w[k-1]; reps[k]=reps[k-1];}
            w[j]=W[i];
            reps[j]=i;
        }
    }

    // get point indices in voronoi
    VUI neighbors;   for (i=0; i<nbVoxels; i++) if (voronoi[i]==(int)sampleindex) neighbors.push_back((unsigned int)i);
    bool dilatevoronoi=true;
    if (dilatevoronoi) // take points outside grid for non-singular fitting
    {
        unsigned int nbneighb=neighbors.size();
        for (i=0; i<nbneighb; i++)
        {
            VUI tmp;
            get6Neighbors(neighbors[i], tmp);
            for (j=0; j<tmp.size(); j++)
                if (grid.data()[tmp[j]]==0)
                {
                    k=nbneighb; while(k<neighbors.size() && neighbors[k]!=tmp[j]) k++;
                    if (k==neighbors.size()) neighbors.push_back(tmp[j]);
                }
        }
    }

    // lump the weights
    for (i=0; i<nbRef; i++)
        if(w[i]!=0)
        {
            pasteRepartioninWeight(reps[i]);
            if(!dw) lumpWeights(neighbors,point,w[i]);
            else /*if(!ddw)*/ lumpWeights(neighbors,point,w[i],&(*dw)[i]);
            //else lumpWeights(neighbors,point,w[i],&(*dw)[i],&(*ddw)[i]);  // desctivated for speed... (weights are supposed to be linear and not quadratic)
        }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Lump_Weights_Repartition");
#endif
    return true;
}




template<class real>
inline real determ(vector<vector<real> >& m, int S)
{
    if (S < 2) {return 0;}
    else if (S == 2) { return m[0][0] * m[1][1] - m[1][0] * m[0][1]; }
    else
    {
        int i,j,j1,j2;
        real det = 0;
        vector<vector<real> > m1(S-1,vector<real>(S-1));
        for (j1=0; j1<S; j1++)
        {
            for (i=1; i<S; i++)
            {
                j2 = 0;
                for (j=0; j<S; j++)
                {
                    if (j == j1) continue;
                    m1[i-1][j2] = m[i][j];
                    j2++;
                }
            }
            det += pow(-1.0,1.0+j1+1.0) * m[0][j1] * determ(m1,S-1);
        }
        return(det);
    }
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::lumpWeights(const VUI& indices,const SCoord& point,Real& w,SGradient* dw,SHessian* ddw,Real* err)
{
    if (!nbVoxels) return false;
    if (weights.size()!=nbVoxels) return false; // point not in grid or no weight computed

    //sout<<"fit on "<<indices.size()<<" voxels"<<sendl;

    unsigned int i,j,k;
    Real MIN_DET=1E-20;

    // least squares fit
    unsigned int order=0;
    if (ddw && dw) order=2;
    else if (dw) order=1;
    unsigned int dim=(order+1)*(order+2)*(order+3)/6;
    vector<vector<Real> > Cov((int)dim,vector<Real>((int)dim,(Real)0)); // accumulate dp^(order)dp^(order)T
    SCoord pi;
    for (j=0; j<indices.size(); j++)
    {
        getCoord(indices[j],pi);
        //sout<<"dp["<<j<<"]="<<pi-point<<",w="<<weights[indices[j]]<<sendl;
        accumulateCovariance(pi-point,order,Cov);
    }

    //sout<<"Cov="<<Cov<<sendl;

    // invert covariance matrix
    vector<vector<Real> > invCov((int)dim,vector<Real>((int)dim,(Real)0));
    if (order==0) invCov[0][0]=1./Cov[0][0];    // simple average
    else if (order==1)
    {
        Mat<4,4,Real> tmp,invtmp;
        for (i=0; i<4; i++) for (j=0; j<4; j++) tmp[i][j]=Cov[i][j];
        //sout<<"det="<<determ(Cov,dim)<<sendl;
        if (determ(Cov,dim)<MIN_DET) invCov[0][0]=1./Cov[0][0];    // coplanar points->not invertible->go back to simple average
        else	{ invtmp.invert(tmp); for (i=0; i<4; i++) for (j=0; j<4; j++) invCov[i][j]=invtmp[i][j];}
    }
    else if (order==2)
    {
        Mat<10,10,Real> tmp,invtmp;
        for (i=0; i<10; i++) for (j=0; j<10; j++) tmp[i][j]=Cov[i][j];
        if (determ(Cov,dim)<MIN_DET) // try order 1
        {
            ddw->fill(0);
            return lumpWeights(indices,point,w,dw,NULL,err);
        }
        else { invtmp.invert(tmp); for (i=0; i<10; i++) for (j=0; j<10; j++) invCov[i][j]=invtmp[i][j];}
    }

    // compute weights and its derivatives
    vector<Real> basis,wdp((int)dim,(Real)0);
    for (j=0; j<indices.size(); j++)
    {
        getCoord(indices[j],pi);
        getCompleteBasis(pi-point,order,basis);
        for (i=0; i<dim; i++) wdp[i]+=basis[i]*weights[indices[j]];
    }

    vector<Real> W((int)dim,(Real)0);
    for (i=0; i<dim; i++) for (j=0; j<dim; j++) W[i]+=invCov[i][j]*wdp[j];


    //
    //sout<<"wdp="; for (i=0;i<dim;i++) sout<<wdp[i]<<","; sout<<sendl;
    //vector<Real> sdp((int)dim,(Real)0);
    //for (j=0;j<indices.size();j++) { getCoord(indices[j],pi); getCompleteBasis(pi-point,order,basis); for (i=0;i<dim;i++) sdp[i]+=basis[i]; }
    //sout<<"sdp="; for (i=0;i<dim;i++) sout<<sdp[i]<<","; sout<<sendl;
    //sout<<"W="; for (i=0;i<dim;i++) sout<<W[i]<<","; sout<<sendl;
    //sout<<"invCov="; for (i=0;i<dim;i++) for (j=0;j<dim;j++) sout<<invCov[i][j]<<","; sout<<sendl;
    if(err)
    {
        for (j=0; j<indices.size(); j++)
        {
            getCoord(indices[j],pi);
            getCompleteBasis(pi-point,order,basis);
            Real er=0; for (k=0; k<dim; k++) er+=basis[k]*W[k];
            er=weights[indices[j]]-er; if(er<0) er*=-1;
            *err+=er;
        }
    }

    w=W[0];
    if (order==0) return true;

    vector<SGradient> basisderiv;
    getCompleteBasisDeriv(SCoord(0,0,0),order,basisderiv);
    for (i=0; i<3; i++)
    {
        (*dw)[i]=0;
        for (j=0; j<dim; j++) (*dw)[i]+=W[j]*basisderiv[j][i];
    }
    if (order==1) return true;

    vector<SHessian> basisderiv2;
    getCompleteBasisDeriv2(SCoord(0,0,0),order,basisderiv2);
    for (i=0; i<3; i++) for (k=0; k<3; k++)
        {
            (*ddw)[i][k]=0;
            for (j=0; j<dim; j++) (*ddw)[i][k]+=W[j]*basisderiv2[j][i][k];
        }
    return true;
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::accumulateCovariance(const SCoord& p,const unsigned int order,vector<vector<Real> >& Cov)
{
    vector<Real> basis;
    getCompleteBasis(p,order,basis);
    unsigned int dim=(order+1)*(order+2)*(order+3)/6;
    for (unsigned int i=0; i<dim; i++) for (unsigned int j=0; j<dim; j++) Cov[i][j]+=basis[i]*basis[j];
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::interpolateWeightsRepartition(const SCoord& point,VRef& reps,VRefReal& w,const int restrictToLabel)
{
    for (unsigned int i=0; i<nbRef; i++) {reps[i]=0; w[i]=0;}
    if (!nbVoxels) return false;

    int index=getIndex(point);
    if (index==-1) std::cout<<"problem with point:"<<point<<" (out of the grid)"<<std::endl;

    if (index==-1) return false; // point not in grid
    if (v_weights.size()!=nbVoxels || v_index.size()!=nbVoxels) return false; // weights not computed

    if(restrictToLabel!=-1)
    {
        SCoord coord;
        Real d,dmin=std::numeric_limits<Real>::max();
        index=-1;
        for (unsigned int i=0; i<this->nbVoxels; i++) if(grid.data()[i]==restrictToLabel)
            {
                getCoord(i,coord);
                d=(point-coord).norm2();
                if(d<dmin) {dmin=d; index=i;}
            }
        if(index==-1) return interpolateWeightsRepartition(point,reps,w);
        for (unsigned int i=0; i<nbRef; i++) { reps[i]=v_index[index][i]; w[i]=v_weights[index][i];} // take the weight of the closest voxel (ok for rigid parts). To do: interpolate linearly in the voxels of the label
        return true;
    }

    VUI ptlist; get26Neighbors(index,ptlist);
    SGradient dw;

    unsigned int i,j;
    bool ok=false;
    for (i=0; i<nbRef; i++)
        if(v_weights[index][i]!=0)
        {
            reps[i]=v_index[index][i];
            for (j=0; j<ptlist.size(); j++) weights[ptlist[j]]=findWeightInRepartition(ptlist[j],reps[i]);
            lumpWeights(ptlist,point,w[i],&dw,NULL);
            if(w[i]>0) ok=true;
        }
        else reps[i]=w[i]=0;

    if (!ok)
    {
        std::cout<<"problem with point:"<<point<<" data:"<<(int)grid.data()[index]<<" (out of data)"<<std::endl;
        // take the weight of the closest voxel.
        SCoord coord;
        Real d,dmin=std::numeric_limits<Real>::max();
        index=-1;
        for (unsigned int i=0; i<this->nbVoxels; i++) if(v_weights[i][0]!=0)
            {
                getCoord(i,coord);
                d=(point-coord).norm2();
                if(d<dmin) {dmin=d; index=i;}
            }
        if(index==-1) return false;
        for (unsigned int i=0; i<nbRef; i++) { reps[i]=v_index[index][i]; w[i]=v_weights[index][i];}
        std::cout<<"resolved using the closest voxel:"<<index<<std::endl;
        return true;
    }

    return true;
}




/*********************************/
/*   Compute weights/distances   */
/*********************************/


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeWeights(const VecSCoord& points)
{
    // TO DO: add params as data? : GEODESIC factor, DIFFUSION fixdatavalue, DIFFUSION max_iterations, DIFFUSION precision
    if (!nbVoxels) return false;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Weights");
#endif
    clock_t timer = clock();

    unsigned int i,dtype=this->distanceType.getValue().getSelectedId(),nbp=points.size();

    // init
    this->v_index.resize(nbVoxels);
    for (i=0; i<nbVoxels; i++)
        this->v_index[i].fill((unsigned int)-1);
    this->v_weights.resize(nbVoxels);
    for (i=0; i<nbVoxels; i++)
        this->v_weights[i].fill(0);


    if (dtype==DISTANCE_GEODESIC)
    {
        computeGeodesicalDistances (points); // compute voronoi and distances inside voronoi

        for (i=0; i<nbp; i++)
        {
            computeAnisotropicLinearWeightsInVoronoi(points[i]);
            offsetWeightsOutsideObject();
            addWeightinRepartion(i);
        }
    }
    else if (dtype==DISTANCE_DIFFUSION || dtype==DISTANCE_ANISOTROPICDIFFUSION)
    {
        for (i=0; i<nbp; i++)
        {
            HeatDiffusion(points,i,false,this->maxLloydIterations.getValue());
            offsetWeightsOutsideObject();
            addWeightinRepartion(i);
        }
    }


    normalizeWeightRepartion();

    std::cout<<"Grid weight computation completed"<<std::endl;
    if (weightFile.isSet()) saveWeightRepartion();
    showedrepartition=-1;

    updateMaxValues();

    if (verbose.getValue())
        std::cout << "INITTIME computeWeights: " << (clock() - timer) / (float)CLOCKS_PER_SEC << " sec." << std::endl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Weights");
#endif
    return true;
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::addWeightinRepartion(const unsigned int index)
{
    if (!nbVoxels) return;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Add_Weight_in_Repartion");
#endif
    unsigned int j,k;

    for (unsigned int i=0; i<nbVoxels; i++)
        //if (grid.data()[i])
        if(weights[i])
        {
            j=0;
            while (j!=nbRef && v_weights[i][j]>weights[i]) j++;
            if (j!=nbRef) // insert weight and index in the ordered list
            {
                for (k=nbRef-1; k>j; k--)
                {
                    v_weights[i][k]=v_weights[i][k-1];
                    v_index[i][k]=v_index[i][k-1];
                }
                v_weights[i][j]=weights[i];
                v_index[i][j]=index;
            }
        }
    showedrepartition=-1;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Add_Weight_in_Repartion");
#endif
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::pasteRepartioninWeight(const unsigned int index)
{
    if (!nbVoxels || v_index.size()!=nbVoxels || v_weights.size()!=nbVoxels) return false;
    weights.resize(this->nbVoxels);
    unsigned int i;
    bool allzero=true;
    for (i=0; i<nbVoxels; i++) {weights[i]=findWeightInRepartition(i,index); if(weights[i]!=0) allzero=false;}
    showedrepartition=-1;
    if(allzero) return false; else return true;
}

template < class MaterialTypes>
void GridMaterial< MaterialTypes>::normalizeWeightRepartion()
{
    if (!nbVoxels) return;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Normalize_Weight_Repartion");
#endif
    unsigned int j;
    for (unsigned int i=0; i<nbVoxels; i++)
        //if (grid.data()[i])
        if(v_weights[i][0])
        {
            Real W=0;
            for (j=0; j<nbRef && v_weights[i][j]!=0; j++) W+=v_weights[i][j];
            if (W!=0) for (j=0; j<nbRef  && v_weights[i][j]!=0; j++) v_weights[i][j]/=W;
        }
    showedrepartition=-1;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Normalize_Weight_Repartion");
#endif
}



template < class MaterialTypes>
void GridMaterial< MaterialTypes>::removeVoxels( const vector<unsigned int>& voxelsToRemove)
{
    // Delete Voxels
    GCoord gCoord;
    for (vector<unsigned int>::const_iterator it = voxelsToRemove.begin(); it != voxelsToRemove.end(); ++it)
    {
        getiCoord( *it, gCoord);
        grid(gCoord[0], gCoord[1], gCoord[2]) = 0;
    }

    //localyUpdateWeights (voxelsToRemove);

    voxelsHaveChanged.setValue(true);
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::localyUpdateWeights( const vector<unsigned int>& /*removedVoxels*/)
{
    /*
    set<unsigned int> voxelsToUpdate;
    set<unsigned int> border;
    VUI neighbors;
    for (unsigned int index = 0; index < v_weights.size(); ++index) // For each frame, update the weights
    {
        // Find 'voxelsToUpdate' and 'border'
        //set<unsigned int> parsedVoxels;
        std::queue<unsigned int > front;
        for (vector<unsigned int>::const_iterator it = removedVoxels.begin(); it != removedVoxels.end(); ++it)
            front.push( *it);
        while( !front.empty())
        {
            unsigned int voxelID = front.front(); // Get the last voxel on the stack.
            front.pop();          // Remove it from the stack.

            //if ( parsedVoxels.find( voxelID ) != parsedVoxels.end() )   continue; // if this hexa is ever parsed, continue
            //parsedVoxels.insert (voxelID);

            // Propagate to the neighbors
            get26Neighbors ( voxelID, neighbors );
            for ( VUI::const_iterator it = neighbors.begin(); it != neighbors.end(); it++ )
            {
                if (all_distances[*it][index] > all_distances[voxelID][index])
                {
                    front.push (*it);
                    voxelsToUpdate.insert (*it);
                    border.erase( *it);
                }
                else //if (parsedVoxels.find( *it ) == parsedVoxels.end())
                {
                    border.insert( *it);
                }
            }
        }

        // Set new distances on 'voxelsToUpdate' set
        typedef std::pair<Real,unsigned> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue
        q.insert( DistanceToPoint(0.,index) );
        distances[index]=0;
        int fromLabel=(int)grid.data()[index];

        while( !q.empty() ){
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;

            get26Neighbors(v, neighbors);
            for (unsigned int i=0;i<neighbors.size();i++){
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                {
                    Real cost = getDistance(v,v2,fromLabel);
                    //if(distanceScaleFactors) cost*=(*distanceScaleFactors)[v2];

                    if(distances[v2] > distances[v] + cost)
                    {
                        if(distances[v2] != distMax) {
                            q.erase(q.find(DistanceToPoint(distances[v2],v2)));
                        }
                        distances[v2] = distances[v] + cost;
                        q.insert( DistanceToPoint(distances[v2],v2) );
                    }
                }
            }
        }

        // Check unset voxels to insert a new frame
        // TODO
    }*/
}


template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial< MaterialTypes>::getDistance(const unsigned int& index1,const unsigned int& index2,const int fromLabel)
{
    if (!nbVoxels) return -1;
    SCoord coord1;
    if (!getCoord(index1,coord1)) return -1; // point1 not in grid
    SCoord coord2;
    if (!getCoord(index2,coord2)) return -1; // point2 not in grid

    if (this->biasDistances.getValue() && biasFactor!=(Real)0.) // bias distances according to stiffness
    {
        if(isRigid(fromLabel) && isRigid(grid.data()[index2])) // does not allow communication accross rigid parts
        {
            if(fromLabel!=grid.data()[index2]) return (Real)1E10; else  return (Real)1E-10;
        }
        Real meanstiff=(getStiffness(grid.data()[index1])+getStiffness(grid.data()[index2]))/2.;
        if(biasFactor!=(Real)1.) meanstiff=(Real)pow(meanstiff,biasFactor);

        return ((Real)(coord2-coord1).norm()/meanstiff);
    }
    else return (Real)(coord2-coord1).norm();
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const SCoord& point, const Real distMax ,const vector<Real>* distanceScaleFactors)
{
    if (!nbVoxels) return false;
    int index=getIndex(point);
    return computeGeodesicalDistances (index, distMax ,distanceScaleFactors);
}



template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const int& index, const Real distMax ,const vector<Real>* distanceScaleFactors)
{
    if (!nbVoxels) return false;
    unsigned int i,index2;
    distances.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++) distances[i]=distMax;

    if (index<0 || index>=(int)nbVoxels) return false; // voxel out of grid
    int fromLabel=(int)grid.data()[index];
    if (!fromLabel) return false;  // voxel out of object

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Geodesical_Distances");
#endif
    clock_t timer = clock();

    VUI neighbors;

    if( useDijkstra.getValue()==true )
    {

        /// dijkstra algorithm
        typedef std::pair<Real,unsigned> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue
        q.insert( DistanceToPoint(0.,index) );
        distances[index]=0;

        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;

            get26Neighbors(v, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                {
                    Real cost = getDistance(v,v2,fromLabel);
                    if(distanceScaleFactors)	cost*=(*distanceScaleFactors)[v2];

                    if(distances[v2] > distances[v] + cost)
                    {
                        if(distances[v2] != distMax)
                        {
                            q.erase(q.find(DistanceToPoint(distances[v2],v2)));
                        }
                        distances[v2] = distances[v] + cost;
                        q.insert( DistanceToPoint(distances[v2],v2) );
                    }
                }
            }
        }

    }
    else
    {
        /// propagation algorithm
        Real d;
        unsigned int index1;
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
                    d=getDistance(index1,index2,fromLabel);
                    if(distanceScaleFactors)	d*=(*distanceScaleFactors)[index2];
                    if (distances[index2]>d+distances[index1])
                    {
                        distances[index2]=d+distances[index1];
                        fifo.push(index2);
                    }
                }
            }
            fifo.pop();
        }
        /// end propagation algorithm
    }

    updateMaxValues();

    if (verbose.getValue())
        std::cout << "INITTIME computeWeights: " << (clock() - timer) / (float)CLOCKS_PER_SEC << " sec." << std::endl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Geodesical_Distances");
#endif
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const VecSCoord& points, const Real distMax )
{
    if (!nbVoxels) return false;
    vector<int> indices;
    for (unsigned int i=0; i<points.size(); i++) indices.push_back(getIndex(points[i]));
    return computeGeodesicalDistances ( indices, distMax );
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistances ( const vector<int>& indices, const Real distMax )
{
    if (!nbVoxels) return false;
    unsigned int i,nbi=indices.size(),index2;
    distances.resize(this->nbVoxels);
    voronoi.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)
    {
        distances[i]=distMax;
        voronoi[i]=-1;
    }
    VUI neighbors;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Geodesical_Distances");
#endif

    if( useDijkstra.getValue()==true )
    {
        /// dijkstra algorithm
        typedef std::pair<Real,unsigned> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue
        for (i=0; i<nbi; i++)
        {
            if (indices[i]>=0 && indices[i]<(int)nbVoxels) if (grid.data()[indices[i]]!=0)
                {
                    q.insert( DistanceToPoint(0.,indices[i]) );
                    distances[indices[i]]=0;
                    voronoi[indices[i]]=i;
                }
        }

        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;
            int fromLabel=(int)grid.data()[indices[voronoi[v]]];

            get26Neighbors(v, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                {
                    Real d = distances[v] + getDistance(v,v2,fromLabel);
                    if(distances[v2] > d )
                    {
                        if(distances[v2] != distMax)
                        {
                            q.erase(q.find(DistanceToPoint(distances[v2],v2)));
                        }
                        voronoi[v2]=voronoi[v];
                        distances[v2] = d;
                        q.insert( DistanceToPoint(d,v2) );
                    }
                }
            }
        }

    }
    else
    {

        /// propagation algorithm
        Real d;
        unsigned int index1;
        std::queue<unsigned int> fifo;
        for (i=0; i<nbi; i++) if (indices[i]>=0 && indices[i]<(int)nbVoxels) if (grid.data()[indices[i]]!=0)
                {
                    distances[indices[i]]=0;
                    voronoi[indices[i]]=i;
                    fifo.push(indices[i]);
                }
        if (fifo.empty())
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode("Compute_Geodesical_Distances");
#endif
            return false; // all input voxels out of grid
        }
        while (!fifo.empty())
        {
            index1=fifo.front();
            int fromLabel=(int)grid.data()[indices[voronoi[index1]]];

            get26Neighbors(index1, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                index2=neighbors[i];
                if (grid.data()[index2]) // test if voxel is not void
                {
                    d=distances[index1]+getDistance(index1,index2,fromLabel);
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
        /// end propagation algorithm
    }

    updateMaxValues();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Geodesical_Distances");
#endif

    return true;
}





template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeVoronoiRecursive ( const unsigned int fromLabel )
// compute voronoi of voronoi "nbVoronoiSubdivisions" times
// input : a region initialized at 1 in "voronoi"
//       : distances used to compute the voronoi
//       : nbVoronoiSubdivisions data
// returns  : linearly decreasing weights from 0 to 1
//			: the distance from the region center in "distances"
{
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;

    unsigned int i,j;

    VUI neighbors;

    vector<bool> isaseed((int)this->nbVoxels);
    Real precision=(Real)1.;

        weights.resize((int)this->nbVoxels); for (i=0; i<nbVoxels; i++) if(voronoi[i]==1) weights[i]=1; else weights[i]=0;

    if( useDijkstra.getValue()==true )    // Dijkstra algorithm
    {

        typedef std::pair<Real,unsigned> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue
        typename std::set<DistanceToPoint>::iterator qit;

        for (unsigned int nbs=0; nbs<nbVoronoiSubdivisions.getValue(); nbs++)
        {
            // insert seeds
            for (i=0; i<nbVoxels; i++)
                if(weights[i]>0)
                {
                    isaseed[i]=false;
                    if(distances[i]!=0)	  // prevent from adding a seed at a frame
                    {
                        get6Neighbors((int)i, neighbors);
                        for (j=0; j<neighbors.size(); j++)
                            if(grid.data()[neighbors[j]])
                                if(weights[neighbors[j]]<weights[i])  // voronoi frontier
                                {
                                    Real d = 0;
                                    q.insert( DistanceToPoint(d,i) );
                                    distances[i]=d;
                                    j=neighbors.size();
                                    isaseed[i]=true;
                                }
                    }
                }
            if(q.empty()) {std::cout<<"Recursive voronoi has converged in "<<nbs<<" iterations"<<std::endl; return true;}
            // update voronoi with new regions at seeds
            for (i=0; i<nbVoxels; i++)
                if(weights[i]>0)
                {
                    weights[i]*=(Real)2.;
                    if(isaseed[i]) weights[i]-=precision;
                }

            // propagate distance from seeds
            while( !q.empty() )
            {
                DistanceToPoint top = *q.begin();
                q.erase(q.begin());
                unsigned v = top.second;

                get26Neighbors(v, neighbors);
                for (i=0; i<neighbors.size(); i++)
                {
                    unsigned v2 = neighbors[i];
                    if (grid.data()[v2])
                    {
                        Real d = distances[v] + getDistance(v,v2,fromLabel);
                        if(distances[v2] > d )
                        {
                            qit=q.find(DistanceToPoint(distances[v2],v2));
                            if(qit != q.end()) q.erase(qit);
                            distances[v2] = d;
                            q.insert( DistanceToPoint(d,v2) );
                            weights[v2]=weights[v];
                        }
                    }
                }
            }
            // renormalize weights
            for (i=0; i<nbVoxels; i++) if(weights[i]>0)  weights[i]/=(Real)2.;
            precision/=(Real)2.;
        }



        // linear interpolation of weights in each region (requires a final propagation slighlty different at the extremity)

        // new distance grid for interpolation and fill extermities with dmax
        vector<Real> distances2; distances2.resize((int)nbVoxels);
        vector<Real> weights2; weights2.resize((int)nbVoxels);

        //dmax
        Real dmax=(Real)0.;
        for (i=0; i<nbVoxels; i++) if(grid.data()[i]) if(dmax<distances[i]) dmax=distances[i];
        for (i=0; i<nbVoxels; i++) {distances2[i]=dmax; weights2[i]=0;}

        // insert seeds
        for (i=0; i<nbVoxels; i++)
            if(weights[i]>0)
            {
                isaseed[i]=false;
                if(distances[i]!=0)	  // prevent from adding a seed at a frame
                {
                    get6Neighbors((int)i, neighbors);
                    for (j=0; j<neighbors.size(); j++)
                        if(grid.data()[neighbors[j]])
                            if(weights[neighbors[j]]<weights[i])  //  frontier
                            {
                                Real d = 0;
                                q.insert( DistanceToPoint(d,i) );
                                distances2[i]=d;
                                j=neighbors.size();
                                isaseed[i]=true;
                            }
                }
            }
        precision/=(Real)2.;
        for (i=0; i<nbVoxels; i++) if(weights[i]>0) if(isaseed[i]) weights2[i]=weights[i]-precision;

        // propagate distance from seeds
        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;

            get26Neighbors(v, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                {
                    Real d = distances2[v] + getDistance(v,v2,fromLabel);
                    if(distances2[v2] > d )
                    {
                        qit=q.find(DistanceToPoint(distances2[v2],v2));
                        if(qit != q.end()) q.erase(qit);
                        distances2[v2] = d;
                        q.insert( DistanceToPoint(d,v2) );
                        weights2[v2]=weights2[v];
                    }
                }
            }
        }
        // particular case of the end extremity (don't want to have increasing weights behind the frame)
        // the weight is not interpolated by extrapolated

        for (i=0; i<nbVoxels; i++) if(grid.data()[i]) if(weights[i]==0) {distances[i]=dmax;}

        // insert seeds between 0 and the rest
        for (i=0; i<nbVoxels; i++)
            if(weights[i]>0)
            {
                isaseed[i]=false;
                if(distances[i]!=0)	  // prevent from adding a seed at a frame
                {
                    get6Neighbors((int)i, neighbors);
                    for (j=0; j<neighbors.size(); j++)
                        if(grid.data()[neighbors[j]])
                            if(weights[neighbors[j]]==0)
                            {
                                Real d = 0;
                                q.insert( DistanceToPoint(d,i) );
                                j=neighbors.size();
                            }
                }
            }
        // propagate distance from seeds
        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;
            get26Neighbors(v, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                    if(weights[v2]==0)
                    {
                        Real d = distances[v] + getDistance(v,v2,fromLabel);
                        if(distances[v2] > d )
                        {
                            qit=q.find(DistanceToPoint(distances[v2],v2));
                            if(qit != q.end()) q.erase(qit);
                            distances[v2] = d;
                            q.insert( DistanceToPoint(d,v2) );
                        }
                    }
            }
        }


        // interpolation
        for (i=0; i<nbVoxels; i++)
            if(weights[i]>0 || distances2[i]!=dmax)
                if(distances[i]!=0)
                {
                    if(weights[i]==0)
                    {
                        weights[i] +=precision*((Real)1. - (Real)2.* distances2[i]/distances[i]);
                    }
                    else
                    {
                        if(weights2[i]>weights[i]) weights[i] +=precision*distances[i]/(distances[i]+distances2[i]);
                        else weights[i] -=precision*distances[i]/(distances[i]+distances2[i]);
                    }
                }

    } // end Dijkstra



    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeGeodesicalDistancesToVoronoi ( const unsigned int fromLabel,const Real distMax , VUI* ancestors )
// compute distances from the border of the region initialized at 1 in "voronoi"
{
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Geodesical_Distances_To_Voronoi");
#endif

    unsigned int i,j,index2;
    distances.resize(this->nbVoxels);
    if(ancestors) ancestors->resize(this->nbVoxels);

    for (i=0; i<this->nbVoxels; i++) distances[i]=distMax;

    VUI neighbors;
    if( useDijkstra.getValue()==true )    // Dijkstra algorithm
    {
        typedef std::pair<Real,unsigned> DistanceToPoint;
        std::set<DistanceToPoint> q; // priority queue

        for (i=0; i<nbVoxels; i++)
        {
            if(voronoi[i]>0)
            {
                get6Neighbors((int)i, neighbors);
                for (j=0; j<neighbors.size(); j++)
                {
                    if(voronoi[neighbors[j]]<voronoi[i] && voronoi[neighbors[j]]!=-1) // voronoi frontier
                    {
                        Real d = 0; //getDistance(i,neighbors[j],fromLabel)/(Real)2.; // distance from border
                        q.insert( DistanceToPoint(d,i) );
                        distances[i]=d;
                        if(ancestors) (*ancestors)[i]=i;
                        j=neighbors.size();
                    }
                }
            }
        }

        while( !q.empty() )
        {
            DistanceToPoint top = *q.begin();
            q.erase(q.begin());
            unsigned v = top.second;

            get26Neighbors(v, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                unsigned v2 = neighbors[i];
                if (grid.data()[v2])
                {
                    Real d = distances[v] + getDistance(v,v2,fromLabel);
                    if(distances[v2] > d )
                    {
                        if(distances[v2] != distMax)
                        {
                            q.erase(q.find(DistanceToPoint(distances[v2],v2)));
                        }
                        distances[v2] = d;
                        q.insert( DistanceToPoint(d,v2) );
                        if(ancestors) (*ancestors)[v2]=(*ancestors)[v];
                    }
                }
            }
        }
    }
    else                       /// propagation algorithm
    {
        Real d;
        unsigned int index1;
        std::queue<int> fifo;
        for (i=0; i<nbVoxels; i++)
            if(voronoi[i]>0)
            {
                get6Neighbors((int)i, neighbors);
                for (j=0; j<neighbors.size(); j++)
                    if(voronoi[neighbors[j]]<voronoi[i] && voronoi[neighbors[j]]!=-1) // voronoi frontier
                    {
                        Real d = 0; //getDistance(i,neighbors[j],fromLabel)/(Real)2.; // distance from border
                        distances[i]=d; fifo.push((int)i);
                        if(ancestors) (*ancestors)[i]=i;
                        j=neighbors.size();
                    }
            }
        while (!fifo.empty())
        {
            index1=fifo.front();
            get26Neighbors(index1, neighbors);
            for (i=0; i<neighbors.size(); i++)
            {
                index2=neighbors[i];
                if (grid.data()[index2]) // test if voxel is not void
                {
                    d=distances[index1]+getDistance(index1,index2,fromLabel);
                    if (distances[index2]>d)
                    {
                        distances[index2]=d;
                        if(ancestors) (*ancestors)[index2]=(*ancestors)[index1];
                        fifo.push(index2);
                    }
                }
            }
            fifo.pop();
        }
    }

    updateMaxValues();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Geodesical_Distances_To_Voronoi");
#endif

    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::updateLinearityError(const bool allweights)
{
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;

    unsigned int i,j,ii;
    int k;

    // get voxel indices of different voronoi regions
    vector<unsigned int> indices;
    for (i=0; i<this->nbVoxels; i++)
        if(voronoi[i]!=-1)
        {
            k=-1;
            for (j=0; j<(unsigned int)indices.size() && k==-1; j++) if(voronoi[i]==voronoi[indices[j]]) k=j;
            if(k==-1) indices.push_back(i);
        }
    if(!indices.size()) return false;

    // compute error
    linearityError.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++) linearityError[i]=0;
    VUI ptlist; SCoord point,point2;
    Real w; SGradient dw;
    Real meanerror=0,minerror=1E10,maxerror=0; unsigned int count=0;
    unsigned int nbframes=1;

    for (i=0; i<indices.size(); i++)
    {
        if(allweights) // get the highest frame index influencing the region
        {
            nbframes=0;
            for (j=0; j<(unsigned int)nbVoxels; j++) if (voronoi[j]==voronoi[indices[i]]) for (ii=0; ii<nbRef; ii++) if(v_weights[j][ii]!=0) if(nbframes<v_index[j][ii]) nbframes=v_index[j][ii];
        }

        getCoord(indices[i],point);
        ptlist.clear();  for (j=0; j<(unsigned int)nbVoxels; j++) if (voronoi[j]==voronoi[indices[i]]) ptlist.push_back(j);

        for (ii=0; ii<nbframes; ii++)
        {
            if(allweights)  {for (j=0; j<ptlist.size(); j++) weights[ptlist[j]]=findWeightInRepartition(ptlist[j],ii+1); }
            Real err=0; bool allzero=true;
            for (j=0; j<ptlist.size(); j++) if(weights[ptlist[j]]!=0) {allzero=false; j=ptlist.size();}

            if(!allzero)
            {
                lumpWeights(ptlist,point,w,&dw,NULL);
                for (j=0; j<ptlist.size(); j++)
                {
                    getCoord(ptlist[j],point2);
                    err=weights[ptlist[j]] - dot(point2-point,dw) -w;
                    if(err<0) err*=(Real)-1;
                    if(err<1E-12) err=0; // clamp for visualization..
                    if(minerror>err) minerror=err;
                    if(maxerror<err) maxerror=err;
                    linearityError[ptlist[j]]+=err;

                    meanerror+=err; count++;
                }
            }
        }
    }
    if(count) meanerror/=(Real)count;
    std::cout<<"Min/Average/Max error in influenced regions per voxel="<<minerror<<"/"<<meanerror<<"/"<<maxerror<<std::endl;

    updateMaxValues();
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeLinearRegionsSampling ( VecSCoord& points, const unsigned int num_points)
{
    if (!nbVoxels) return false;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Linear_Regions_Sampling");
#endif
    clock_t timer = clock();

    unsigned int i,j,ii,initial_num_points=points.size();
    int k;

    // identify regions with similar repartitions and similar stiffness
    voronoi.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++) voronoi[i]=-1;

    vector<unsigned int> indices;
    vector<Real> stiffnesses;

    // insert initial points
    for (i=0; i<initial_num_points; i++)
    {
        j=getIndex(points[i]);
        if(grid.data()[j])
        {
            voronoi[j]=indices.size();
            indices.push_back(j);
            stiffnesses.push_back(getStiffness(grid.data()[j]));
        }
    }


    // visit all voxels and insert new reps/stiffness if necessary
    for (i=0; i<this->nbVoxels; i++)
        if(grid.data()[i])
        {
            Real stiff=getStiffness(grid.data()[i]);
            k=-1;
            for (j=0; j<(unsigned int)indices.size() && k==-1; j++) // detect similar already inserted repartitions and stiffness
                //if(stiffnesses[j]==stiff)
                if(areRepsSimilar(i,indices[j])) k=j;

            if(k==-1)   // insert
            {
                voronoi[i]=indices.size();
                indices.push_back(i);
                stiffnesses.push_back(stiff);
            }
            else voronoi[i]=k;
        }

    // check linearity in each region and subdivide region of highest error until num_points is reached
    vector<Real> errors(indices.size(),(Real)0.);
    Real w; SGradient dw; SHessian ddw; VUI ptlist;  SCoord point;

    for (j=0; j<indices.size(); j++)
    {
        getCoord(indices[j],point);
        ptlist.clear();  for (i=0; i<nbVoxels; i++) if (voronoi[i]==(int)j) ptlist.push_back(i);
        for (i=0; i<nbRef; i++)
            if(v_weights[indices[j]][i]!=0)
            {
                for (ii=0; ii<ptlist.size(); ii++) weights[ptlist[ii]]=findWeightInRepartition(ptlist[ii],v_index[indices[j]][i]);
                //pasteRepartioninWeight(v_index[indices[j]][i]);
                lumpWeights(ptlist,point,w,&dw,NULL,&errors[j]);
            }
    }

    while(indices.size()<num_points)
    {
        Real maxerr=0;
        for (j=0; j<indices.size(); j++) if(errors[j]>maxerr) {maxerr=errors[j]; i=j;}
        if (maxerr == 0) break;
        SubdivideVoronoiRegion(i,indices.size());
        j=0; while(j<(unsigned int)nbVoxels && (unsigned int)voronoi[j]!=indices.size()) j++;
        if(j==nbVoxels) {errors[i]=0; continue;} //unable to add region
        else
        {
            indices.push_back(j); errors.push_back(0);
            // update errors
            errors[i]=0;
            getCoord(indices[i],point);
            ptlist.clear();  for (j=0; j<(unsigned int)nbVoxels; j++) if ((unsigned int)voronoi[j]==i) ptlist.push_back(j);
            for (j=0; j<nbRef; j++)
                if(v_weights[indices[i]][j]!=0)
                {
                    for (ii=0; ii<ptlist.size(); ii++) weights[ptlist[ii]]=findWeightInRepartition(ptlist[ii],v_index[indices[i]][j]);
                    //pasteRepartioninWeight(v_index[indices[i]][j]);
                    lumpWeights(ptlist,point,w,&dw,NULL,&errors[i]);
                }
            i=indices.size()-1;
            errors[i]=0;
            getCoord(indices[i],point);
            ptlist.clear();  for (j=0; j<(unsigned int)nbVoxels; j++) if ((unsigned int)voronoi[j]==i) ptlist.push_back(j);
            for (j=0; j<nbRef; j++)
                if(v_weights[indices[i]][j]!=0)
                {
                    for (ii=0; ii<ptlist.size(); ii++) weights[ptlist[ii]]=findWeightInRepartition(ptlist[ii],v_index[indices[i]][j]);
                    //pasteRepartioninWeight(v_index[indices[i]][j]);
                    lumpWeights(ptlist,point,w,&dw,NULL,&errors[i]);
                }
        }
    }

    Real err=0; unsigned int nbinfluences=0;
    for (j=0; j<indices.size(); j++)
    {
        err+=errors[j];
        ptlist.clear();  for (i=0; i<nbVoxels; i++) if (voronoi[i]==(int)j) ptlist.push_back(i);
        for (i=0; i<nbRef; i++) if(v_weights[indices[j]][i]!=0)
                nbinfluences+=ptlist.size(); // nb influencing frames x nb voxels
    }
    err/=(Real)(nbinfluences);
    std::cout<<"Average kernel fitting error per voxel="<<err<<std::endl;

    // insert gauss points in the center of voronoi regions
    points.resize(indices.size());
    for (j=initial_num_points; j<indices.size(); j++)
    {
        points[j].fill(0);
        SCoord p; unsigned int count=0;
        for (i=0; i<this->nbVoxels; i++) if(voronoi[i]==(int)j) {getCoord(i,p); points[j]+=p; count++; }
        if(count!=0)
        {
            points[j]/=(Real)count;
        }
    }


    //std::cout<<"Added " << indices.size()-initial_num_points << " samples"<<std::endl;
    updateMaxValues();
    if (verbose.getValue())
        std::cout << "INITTIME computeLinearRegionsSampling: " << (clock() - timer) / (float)CLOCKS_PER_SEC << " sec." << std::endl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Linear_Regions_Sampling");
#endif
    return true;
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::SubdivideVoronoiRegion( const unsigned int voronoiindex, const unsigned int newvoronoiindex, const unsigned int max_iterations )
{
    if (!nbVoxels) return false;
    if (voronoi.size()!=nbVoxels) return false;

    unsigned int i,i1,i2=0;
    SCoord p1,p2;

    // initialization: take the first point and its farthest point inside the voronoi region
    i1=0; while(i1<nbVoxels && (unsigned int)voronoi[i1]!=voronoiindex) i1++;
    if(i1==nbVoxels) return false;
    getCoord(i1,p1);

    Real d,dmax=0;
    for (i=0; i<nbVoxels; i++)
        if((unsigned int)voronoi[i]==voronoiindex)
        {
            getCoord(i,p2); d=(p2-p1).norm2();
            if(d>dmax) {i2=i; dmax=d;}
        }
    if(dmax==0) return false;
    getCoord(i2,p2);

    // Lloyd relaxation
    Real d1,d2;
    SCoord p,cp1,cp2;
    unsigned int nb1,nb2,nbiterations=0;
    while (nbiterations<max_iterations)
    {
        // mode points to the centroid of their voronoi regions
        nb1=nb2=0;
        cp1.fill(0); cp2.fill(0);
        for (i=0; i<nbVoxels; i++)
            if((unsigned int)voronoi[i]==voronoiindex)
            {
                getCoord(i,p);
                d1=(p1-p).norm2(); d2=(p2-p).norm2();
                if(d1<d2) { cp1+=p; nb1++; }
                else { cp2+=p; nb2++; }
            }
        if(nb1!=0) cp1/=(Real)nb1;	nb1=getIndex(cp1);
        if(nb2!=0) cp2/=(Real)nb2;	nb2=getIndex(cp2);
        if(nb1==i1 && nb2==i2) nbiterations=max_iterations;
        else {i1=nb1; getCoord(i1,p1); i2=nb2; getCoord(i2,p2); nbiterations++; }
    }

    // fill one region with new index
    for (i=0; i<nbVoxels; i++)
        if((unsigned int)voronoi[i]==voronoiindex)
        {
            getCoord(i,p);
            d1=(p1-p).norm2(); d2=(p2-p).norm2();
            if(d1<d2) voronoi[i]=newvoronoiindex;
        }
    return true;
}




template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeRegularSampling ( VecSCoord& points, const unsigned int step)
{
    if (!nbVoxels) return false;
    if(step==0)  return false;

    unsigned int i,initial_num_points=points.size();
    vector<int> indices;
    for (i=0; i<initial_num_points; i++) indices.push_back(getIndex(points[i]));

    for(unsigned int z=0; z<(unsigned int)dimension.getValue()[2]; z+=step)
        for(unsigned int y=0; y<(unsigned int)dimension.getValue()[1]; y+=step)
            for(unsigned int x=0; x<(unsigned int)dimension.getValue()[0]; x+=step)
                if (grid(x,y,z)!=0)
                    indices.push_back(getIndex(GCoord(x,y,z)));

    computeGeodesicalDistances(indices); // voronoi

    // get points from indices
    points.resize(indices.size());
    for (i=initial_num_points; i<indices.size(); i++)     getCoord(indices[i],points[i]) ;

    //std::cout<<"Added " << indices.size()-initial_num_points << " regularly sampled points"<<std::endl;
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::isRigid(const voxelType label) const
{
    if(getStiffness(label)>=(Real)1.5E6) return true; // 1.5E6 N/cm^2 = bone stiffness
    else return false;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::rigidPartsSampling ( VecSCoord& points)
{
    if (!nbVoxels) return false;
    unsigned int i,k,initial_num_points=points.size();
    int index;

    // insert initial points and get labels of  the different rigid parts
    VUI labellist;
    for (i=0; i<initial_num_points; i++)
    {
        index=getIndex(points[i]);
        if (index == -1)
            serr << "Given Frame Coord of index " << i << " is outside of the grid !" << sendl;
        else if(isRigid(grid.data()[index])) labellist.push_back(grid.data()[index]);
    }
    unsigned int startlabelinsertion=labellist.size();
    for (i=0; i<this->nbVoxels; i++)
        if(grid.data()[i]!=0)
            if(isRigid(grid.data()[i]))
            {
                k=0; if(labellist.size()!=0) while(k<labellist.size() && grid.data()[i]!=labellist[k]) k++;
                if(k==labellist.size())	labellist.push_back(grid.data()[i]);
            }

    // insert new points in the center of rigid regions
    for (unsigned int l=startlabelinsertion; l<labellist.size(); l++)
    {
        unsigned int count=0;
        SCoord c(0,0,0),p;
        for (i=0; i<this->nbVoxels; i++)
            if(grid.data()[i]==labellist[l])
            {
                getCoord(i,p);
                c+=p;
                count++;
            }
        c/=(Real)count;
        index=getIndex(c);
        if((int)index!=-1) if(grid.data()[index]==labellist[l]) {points.push_back(c); continue;}

        // treat special case of concave regions
        Real d,dmin=1E10;
        for (i=0; i<this->nbVoxels; i++)
            if(grid.data()[i]==labellist[l])
            {
                getCoord(i,p);
                d=(p-c).norm2();
                if(d<dmin) {dmin=d; index=i;}
            }
        getCoord(index,c);

        points.push_back(c);
    }

    if(initial_num_points!=points.size()) std::cout<<"Added "<<points.size()-initial_num_points<<" frames in rigid parts"<<std::endl;
    return true;
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeUniformSampling ( VecSCoord& points, const unsigned int num_points)
{
    if (!nbVoxels) return false;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Uniform_Sampling");
#endif
    clock_t timer = clock();

    unsigned int i,k,initial_num_points=points.size(),nb_points=num_points;
    if(initial_num_points>num_points) nb_points=initial_num_points;

    biasFactor=distanceBiasFactor.getValue(); // use user-specified exponent for biased the distance

    vector<int> indices((int)nb_points,-1);
    for (i=0; i<initial_num_points; i++) indices[i]=getIndex(points[i]);
    points.resize(nb_points);

    // initialization: farthest point sampling (see [adams08])
    Real dmax;
    int indexmax;
    if (initial_num_points==0)
    {
        indices[0]=0;    // take the first not empty voxel as a random point
        while (grid.data()[indices[0]]==0)
        {
            indices[0]++;
            if (indices[0]==(int)nbVoxels)
            {
#ifdef SOFA_DUMP_VISITOR_INFO
                simulation::Visitor::printCloseNode("Compute_Uniform_Sampling");
#endif
                return false;
            }
        }
    }
    else if(initial_num_points==nb_points) computeGeodesicalDistances(indices);

    for (i=initial_num_points; i<nb_points; i++)
    {
        if (i==0) continue; // a random point has been inserted
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
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode("Compute_Uniform_Sampling");
#endif
            return false;    // unable to add point
        }
        indices[i]=indexmax;
    }

    // Lloyd relaxation
    if(initial_num_points!=nb_points)
    {
        SCoord pos,u,pos_point,pos_voxel;
        unsigned int count,nbiterations=0;
        bool ok=false,ok2;
        Real d,dmin;
        int indexmin;
        clock_t lloydItTimeBegin = clock(), lloydTimeBegin = clock();

        while (!ok && nbiterations<maxLloydIterations.getValue())
        {
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printNode("Lloyd_iteration");
#endif
            if (verbose.getValue())
                lloydItTimeBegin = clock();

            ok2=true;
            computeGeodesicalDistances(indices); // Voronoi
            vector<bool> flag((int)nbVoxels,false);
            for (i=initial_num_points; i<nb_points; i++) // move to centroid of Voronoi cells
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
                if (this->biasDistances.getValue() && biasFactor!=(Real)0.)
                {
                    Real stiff=getStiffness(grid.data()[indices[i]]);
                    if(biasFactor!=(Real)1.) stiff=(Real)pow(stiff,biasFactor);
                    pos*=stiff;
                }

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

            if (verbose.getValue())
                std::cout << "INITTIME Lloyd_iteration " << (clock() - lloydItTimeBegin) / (float)CLOCKS_PER_SEC << " sec." << std::endl;
#ifdef SOFA_DUMP_VISITOR_INFO
            simulation::Visitor::printCloseNode("Lloyd_iteration");
#endif

        }

        if (verbose.getValue())
            std::cout << "Lloyd total time " << (clock() - lloydTimeBegin) / (float)CLOCKS_PER_SEC << " sec." << std::endl;

        if (nbiterations==maxLloydIterations.getValue()) serr<<"Lloyd relaxation has not converged in "<<nbiterations<<" iterations"<<sendl;
        else std::cout<<"Lloyd relaxation completed in "<<nbiterations<<" iterations"<<std::endl;
    }

    // save voronoi/distances for further visualization
    voronoi_frames.clear(); voronoi_frames.insert(voronoi_frames.begin(),voronoi.begin(),voronoi.end());

    // get points from indices
    for (i=initial_num_points; i<nb_points; i++)	getCoord(indices[i],points[i]) ;

    biasFactor=(Real)1.; // restore compliance distance

    if (verbose.getValue())
        std::cout << "INITTIME computeUniformSampling: " << (clock() - timer) / (float)CLOCKS_PER_SEC << std::endl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Uniform_Sampling");
#endif
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeAnisotropicLinearWeightsInVoronoi ( const SCoord& point)
/// linearly decreasing weight with support*dist(point,closestVoronoiBorder)
{
    unsigned int i;
    weights.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)  weights[i]=0;
    if (!this->nbVoxels) return false;
    int index=getIndex(point);
    if(index==-1) return false;
    if (voronoi_frames.size()!=nbVoxels) return false;
    if (voronoi_frames[index]==-1) return false;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Compute_Anisotropic_Linear_Weights_In_Voronoi");
#endif

    unsigned int fromLabel=(unsigned int)grid.data()[index];

    // update voronoi of the considered frame
    for (i=0; i<this->nbVoxels; i++)
        if(voronoi_frames[i]==-1) voronoi[i]=-1;
        else if(voronoi_frames[i]==voronoi_frames[index]) voronoi[i]=1;
        else voronoi[i]=0;

    // update dmax in voronoi
    Real dmax=0; for (i=0; i<this->nbVoxels; i++) if(voronoi_frames[i]==voronoi_frames[index]) if(dmax<distances[i])  dmax=distances[i];

    vector<Real> backupdistance; backupdistance.clear();	backupdistance.insert(backupdistance.begin(),distances.begin(),distances.end());

    if(useDistanceScaleFactor.getValue()) // method based on DistanceScaleFactor -> weight = sum_point^voxel dl' with dl'=dl/(support*dist(point,closestVoronoiBorder))
    {
        // compute distance to voronoi border and track ancestors
        VUI ancestors;
        computeGeodesicalDistancesToVoronoi(fromLabel,dmax,&ancestors);

        // update DistanceScaleFactors
        vector<Real> distanceScaleFactors((int)nbVoxels,(Real)1E10);
        for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if(distances[i]<dmax)	if(backupdistance[ancestors[i]]!=0)
                        distanceScaleFactors[i]=(Real)1./(weightSupport.getValue()*backupdistance[ancestors[i]]);

        dmax=1;
        computeGeodesicalDistances(point,dmax,&distanceScaleFactors);
        for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (distances[i]<dmax)
                    weights[i]=(Real)1.-distances[i];

    }
    else
    {
        if(nbVoronoiSubdivisions.getValue()!=0)
        {
            // to do: implement weightsupport in the recursive method
            if(weightSupport.getValue()!=2) serr<<"Weightsupport not supported with voronoi subdivisions"<<sendl;
            computeVoronoiRecursive(fromLabel);
        }
        else
        {
            // method equivalent to nbsubdivision=0 but weightsupport (will be suppressed..)
            dmax*=weightSupport.getValue();
            // compute distance to voronoi border
            computeGeodesicalDistancesToVoronoi(fromLabel,dmax);
            vector<Real> dtovoronoi(distances);
            // compute distance to frame up to the weight function support size
            computeGeodesicalDistances(index,dmax);
            Real support=weightSupport.getValue();

            for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (distances[i]<dmax)
                    {
                        if(distances[i]==0) weights[i]=1;
                        //else if(voronoi_frames[i]==voronoi_frames[index]) weights[i]=1.-distances[i]/(support*(distances[i]+dtovoronoi[i])); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                        //else weights[i]=1.-distances[i]/(support*(distances[i]-dtovoronoi[i]));	// outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
                        else if(voronoi_frames[i]==voronoi_frames[index]) weights[i]=(support-1.)/support + dtovoronoi[i]/(support*(distances[i]+dtovoronoi[i])); // inside voronoi: dist(frame,closestVoronoiBorder)=d+disttovoronoi
                        else weights[i]=(support-1.)/support - dtovoronoi[i]/(support*(distances[i]-dtovoronoi[i]));	// outside voronoi: dist(frame,closestVoronoiBorder)=d-disttovoronoi
                        if(weights[i]<0) weights[i]=0;
                        else if(weights[i]>1) weights[i]=1;
                    }
        }
    }

    distances.clear(); distances.insert(distances.begin(),backupdistance.begin(),backupdistance.end()); // store initial distances from frames

    showedrepartition=-1;
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Compute_Anisotropic_Linear_Weights_In_Voronoi");
#endif
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::computeLinearWeightsInVoronoi ( const SCoord& point)
/// linearly decreasing weight with support=weightSupport*distmax_in_voronoi -> weight= weightSupport*(1-d/distmax)
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
    dmax*=weightSupport.getValue();
    vector<Real> backupdistance;
    backupdistance.swap(distances);
    computeGeodesicalDistances(point,dmax);
    for (i=0; i<nbVoxels; i++) if (grid.data()[i]) if (distances[i]<dmax) weights[i]=1.-distances[i]/dmax;
    backupdistance.swap(distances);
    showedrepartition=-1;
    return true;
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::offsetWeightsOutsideObject(unsigned int offestdist)
{
    if (!this->nbVoxels) return;
    unsigned int i,j;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Offset_Weights_Outside_Object");
#endif

    // get voxels within offsetdist
    VUI neighbors;
    vector<unsigned int> update((int)nbVoxels,0);
    // Liste l'ensemble des voxels null au bord du modele
    for (i=0; i<nbVoxels; i++)
        if(grid.data()[i])
        {
            get26Neighbors(i, neighbors);
            for (j=0; j<neighbors.size(); j++) if (!grid.data()[neighbors[j]]) update[neighbors[j]]=1;
        }
    // Etend l'ensemble a offsetDist
    for (unsigned int nbiterations = 0; nbiterations < offestdist; ++nbiterations)
    {
        for (i=0; i<nbVoxels; i++)
            if(update[i]==1)
            {
                get26Neighbors(i, neighbors);
                for (j=0; j<neighbors.size(); j++) if (!grid.data()[neighbors[j]]) update[neighbors[j]]=2;
                update[i]=3;
            }
        for (i=0; i<nbVoxels; i++) if(update[i]==2) update[i]=1;
    }

    // diffuse 2*offsetdist times for the selected voxels
    for (unsigned int nbiterations = 0; nbiterations < 2*offestdist; ++nbiterations)
        for (i=0; i<this->nbVoxels; i++)
            if (update[i])
            {
                Real val=0,W=0;
                get26Neighbors(i, neighbors);
                for (j=0; j<neighbors.size(); j++)
                {
                    val+=weights[neighbors[j]];
                    W+=1;
                }
                if (W!=0) val=val/W; // normalize value
                weights[i]=val;
            }
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Offset_Weights_Outside_Object");
#endif
}


template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::HeatDiffusion( const VecSCoord& points, const unsigned int hotpointindex,const bool fixdatavalue,const unsigned int max_iterations,const Real precision  )
{
    if (!this->nbVoxels) return false;
    unsigned int i,j,k,num_points=points.size();
    int index;
    weights.resize(this->nbVoxels);
    for (i=0; i<this->nbVoxels; i++)  weights[i]=0;

    vector<bool> isfixed((int)nbVoxels,false);
    vector<bool> update((int)nbVoxels,false);
    VUI neighbors;
    Real diffw,meanstiff;
    Real alphabias=5; // d^2/sigma^2 between a voxel and one its 6 neighbor.

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
        //  #pragma omp parallel for private(j,neighbors,diffw,meanstiff)
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
                            int ip,im;
                            GCoord icoord;
                            getiCoord(i,icoord);
                            neighbors.clear();
                            for (j=0; j<3 ; j++)
                            {
                                icoord[j]+=1; ip=getIndex(icoord); icoord[j]-=2; im=getIndex(icoord); icoord[j]+=1;
                                    if (ip!=-1) if (grid.data()[ip]) neighbors.push_back(ip); else ip=i; else ip=i;
                                    if (im!=-1) if (grid.data()[im]) neighbors.push_back(im); else im=i; else im=i;

                                if(biasDistances.getValue() && biasFactor!=(Real)0.)
                                {
                                    meanstiff=(getStiffness(grid.data()[ip])+getStiffness(grid.data()[im]))/2.;
                                    if(biasFactor!=(Real)1.) meanstiff=(Real)pow(meanstiff,biasFactor);
                                    diffw=(double)exp(-alphabias/(meanstiff*meanstiff));
                                }
                                else diffw=1;

                                dv2=diffw*(weights[ip]-weights[im])*(weights[ip]-weights[im])/4.;
                                val+=(weights[ip]+weights[im])*dv2;
                                W+=2*dv2;
                            }
                        }
                        else
                        {
                            get6Neighbors(i, neighbors);
                            for (j=0; j<neighbors.size(); j++)
                                if (grid.data()[neighbors[j]])
                                {

                                    if(biasDistances.getValue() && biasFactor!=(Real)0.)
                                    {
                                        meanstiff=(getStiffness(grid.data()[i])+getStiffness(grid.data()[neighbors[j]]))/2.;
                                        if(biasFactor!=(Real)1.) meanstiff=(Real)pow(meanstiff,biasFactor);
                                        diffw=(double)exp(-alphabias/(meanstiff*meanstiff));
                                    }
                                    else diffw=1.;
                                    val+=diffw*weights[neighbors[j]];
                                    W+=diffw;
                                }
                                else
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
void GridMaterial< MaterialTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!nbVoxels) return;

    unsigned int showvox=this->showVoxels.getValue().getSelectedId();

    if ( showvox!=SHOWVOXELS_NONE)
    {
        glPushAttrib( GL_LIGHTING_BIT);
        glEnable ( GL_LIGHTING );
        //glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) ;
        glLineWidth(1);

        unsigned int i;
        //        Real s=(voxelSize.getValue()[0]+voxelSize.getValue()[1]+voxelSize.getValue()[2])/3.;
        float color[4], specular[]= {0,0,0,0};
        showWireframe=vparams->displayFlags().getShowWireFrame();

        float label=-1;

        if (showvox==SHOWVOXELS_WEIGHTS)
        {
            if (v_weights.size()==nbVoxels && v_index.size()==nbVoxels) // paste v_weights into weights
            {
                if (showedrepartition!=(int)showWeightIndex.getValue()) pasteRepartioninWeight(showWeightIndex.getValue());
                showedrepartition=(int)showWeightIndex.getValue();
            }
        }
        else if(showvox==SHOWVOXELS_LINEARITYERROR)
        {
            if(linearityError.size()!=nbVoxels || showederror!=(int)showWeightIndex.getValue())
            {
                if(pasteRepartioninWeight(showWeightIndex.getValue())) updateLinearityError(false); else updateLinearityError(true);
            }
            showederror=(int)showWeightIndex.getValue();
            showedrepartition=(int)showWeightIndex.getValue();
        }



        bool slicedisplay=false;
        for (i=0; i<3; i++) if(showPlane.getValue()[i]>=0 && showPlane.getValue()[i]<dimension.getValue()[i]) slicedisplay=true;

        float labelMax = maxValues[showvox];
        cimg_forXYZ(grid,x,y,z)
        {
            //                        if (grid(x,y,z)==0) continue;
            if (slicedisplay)
            {
                if(x!=showPlane.getValue()[0] && y!=showPlane.getValue()[1] && z!=showPlane.getValue()[2])
                    continue;
            }
            else
            {
                //VUI neighbors;
                //get6Neighbors(getIndex(GCoord(x,y,z)), neighbors);
                //if (!wireframe && neighbors.size()==6) // disable internal voxels -> not working anymore (return neighbors outside objects)
                //    continue;
            }

            label = (labelMax>0)?getLabel(x,y,z):-1.0f;
            if (label<=0) continue;
            if (label>labelMax) label=labelMax;
            getColor( color, label);

            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
            glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
            glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);

            SCoord coord;
            getCoord(GCoord(x,y,z),coord);
            drawCube((double)coord[0],(double)coord[1],(double)coord[2]);
        }

        /*
        if (show3DValuesHeight.getValue() != 0 && slicedisplay && vboSupported)
        {
            displayValuesVBO();
        /*/
        if (show3DValuesHeight.getValue() != 0 && slicedisplay && vboSupported.getValue())
        {
            displayValues();
            //*/

            /*
            // Red BBox
            float color[] = {0.8,0.0,0.0,1.0};
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);

            if(showPlane.getValue()[0] != -1)
                drawPlaneBBox(0);
            if(showPlane.getValue()[1] != -1)
                drawPlaneBBox(1);
            if(showPlane.getValue()[2] != -1)
                drawPlaneBBox(2);
            */
        }

        glPopAttrib();
    }
}


template < class MaterialTypes>
float GridMaterial< MaterialTypes>::getLabel( const int&x, const int& y, const int& z) const
{
    float label = -1;
    const unsigned int& showvox=this->showVoxels.getValue().getSelectedId();

    if (showvox==SHOWVOXELS_DATAVALUE) label=(float)grid(x,y,z);
    else if (showvox==SHOWVOXELS_STIFFNESS) label=(float)getStiffness(grid(x,y,z));
    else if (showvox==SHOWVOXELS_DENSITY) label=(float)getDensity(grid(x,y,z));
    else if (showvox==SHOWVOXELS_BULKMODULUS) label=(float)getBulkModulus(grid(x,y,z));
    else if (showvox==SHOWVOXELS_POISSONRATIO) label=(float)getPoissonRatio(grid(x,y,z));
    else if (voronoi.size()==nbVoxels && showvox==SHOWVOXELS_VORONOI)  {if (voronoi.size()!=0) label=(float)voronoi[getIndex(GCoord(x,y,z))]+1.;}
    else if (distances.size()==nbVoxels && showvox==SHOWVOXELS_DISTANCES)  {if (grid(x,y,z)) label=(float)distances[getIndex(GCoord(x,y,z))]; else label=0; }
    else if (weights.size()==nbVoxels && showvox==SHOWVOXELS_WEIGHTS)  { if (grid(x,y,z)) label=(float)weights[getIndex(GCoord(x,y,z))]; else label=0; }
    else if (voronoi.size()==nbVoxels && showvox==SHOWVOXELS_VORONOI_FR)  {if (voronoi_frames.size()!=0) label=(float)voronoi_frames[getIndex(GCoord(x,y,z))]+1.; }
    else if ( linearityError.size()==nbVoxels && showvox==SHOWVOXELS_LINEARITYERROR) {if (grid(x,y,z)) label=(float)linearityError[getIndex(GCoord(x,y,z))]; else label=0; }
    else if ( v_index.size()==nbVoxels && showvox==SHOWVOXELS_FRAMESINDICES) {if (grid(x,y,z) && v_weights[getIndex(GCoord(x,y,z))][showWeightIndex.getValue()%nbRef]) label=(float)v_index[getIndex(GCoord(x,y,z))][showWeightIndex.getValue()%nbRef]+1.; else label=0; }
    return label;
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::getColor( float* color, const float& label) const
{
    unsigned int showvox=this->showVoxels.getValue().getSelectedId();
    float labelMax = maxValues[showvox];
    float value = 240.*(1.0-label/labelMax);
    if (showvox == SHOWVOXELS_DISTANCES) value = 240.*label/labelMax;
    helper::gl::Color::getHSVA(color, value,1.,.8,0.7);
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::drawCube(const double& x, const double& y, const double& z) const
{
    const SCoord& size = voxelSize.getValue();
    glPushMatrix();
    glTranslated (x,y,z);
    glScaled(size[0]*0.5, size[1]*0.5, size[2]*0.5);
    if(showWireframe) glCallList(wcubeList); else glCallList(cubeList);
    glPopMatrix();
}


template < class MaterialTypes>
GLuint GridMaterial< MaterialTypes>::createVBO(const void* data, int dataSize, GLenum target, GLenum usage)
{
    GLuint id = 0;  // 0 is reserved, glGenBuffersARB() will return non-zero id if success

    glGenBuffersARB(1, &id);                        // create a vbo
    glBindBufferARB(target, id);                    // activate vbo id to use
    glBufferDataARB(target, dataSize, data, usage); // upload data to video card

    // check data size in VBO is same as input array, if not return 0 and delete VBO
    int bufferSize = 0;
    glGetBufferParameterivARB(target, GL_BUFFER_SIZE_ARB, &bufferSize);
    if(dataSize != bufferSize)
    {
        glDeleteBuffersARB(1, &id);
        id = 0;
        std::cout << "[createVBO()] Data size is mismatch with input array\n";
    }

    return id;      // return VBO id
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::deleteVBO(const GLuint vboId)
{
    glDeleteBuffersARB(1, &vboId);
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::initVBO()
{
    if(vboSupported.getValue())
    {
        int bufferSize;

        unsigned int realGridSizeX = grid.width() - 2*gridOffset;
        unsigned int realGridSizeY = grid.height() - 2*gridOffset;
        unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;

        // Allocate a plane mesh for each axis
        unsigned int vertexSize = 3*(realGridSizeY * realGridSizeZ + realGridSizeX * realGridSizeZ + realGridSizeX * realGridSizeY);
        unsigned int normalSize = vertexSize;
        unsigned int colorSize = vertexSize;
        unsigned int indicesSize = 4*(((realGridSizeY-1)*(realGridSizeZ-1))+((realGridSizeX-1)*(realGridSizeZ-1))+((realGridSizeX-1)*(realGridSizeY-1)));
        GLfloat* valuesVertices = new GLfloat[vertexSize];
        GLfloat* valuesNormals = new GLfloat[normalSize];
        GLfloat* valuesColors = new GLfloat[colorSize];
        GLushort* valuesIndices = new GLushort[indicesSize];

        // Initialize
        int vOffSet1 = realGridSizeY * realGridSizeZ;
        int vOffSet2 = realGridSizeY * realGridSizeZ + realGridSizeX * realGridSizeZ;
        int iOffSet1 = (realGridSizeY-1) * (realGridSizeZ-1);
        int iOffSet2 = (realGridSizeY-1) * (realGridSizeZ-1) + (realGridSizeX-1) * (realGridSizeZ-1);
        initPlaneGeometry ( valuesVertices, valuesNormals, valuesColors, valuesIndices, 0, 0, 0);
        initPlaneGeometry ( valuesVertices, valuesNormals, valuesColors, valuesIndices, 1, vOffSet1, iOffSet1);
        initPlaneGeometry ( valuesVertices, valuesNormals, valuesColors, valuesIndices, 2, vOffSet2, iOffSet2);

        // Allocate on GPU
        glGenBuffersARB(1, &vboValuesId1);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboValuesId1);
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, sizeof(GLfloat)*(vertexSize+normalSize+colorSize), 0, GL_STREAM_DRAW_ARB);
        glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, sizeof(GLfloat)*vertexSize, valuesVertices);
        glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, sizeof(GLfloat)*vertexSize, sizeof(GLfloat)*normalSize, valuesNormals);
        glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, sizeof(GLfloat)*(vertexSize+normalSize), sizeof(GLfloat)*colorSize, valuesColors);
        glGetBufferParameterivARB(GL_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bufferSize);
        //std::cout << "Vertex and Normal Array in VBO: " << bufferSize << " bytes\n";

        vboValuesId2 = createVBO(valuesIndices, sizeof(GLushort)*indicesSize, GL_ELEMENT_ARRAY_BUFFER_ARB, GL_STATIC_DRAW_ARB);
        glGetBufferParameterivARB(GL_ELEMENT_ARRAY_BUFFER_ARB, GL_BUFFER_SIZE_ARB, &bufferSize);
        //std::cout << "Index Array in VBO: " << bufferSize << " bytes\n";

        delete [] valuesVertices;
        delete [] valuesNormals;
        delete [] valuesColors;
        delete [] valuesIndices;
    }
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::initPlaneGeometry( GLfloat* valuesVertices, GLfloat* valuesNormals, GLfloat* valuesColors, GLushort* valuesIndices, const int& axis, const int nbVerticesOffset, const int nbIndicesOffset)
{
    const SCoord& ori = origin.getValue() + voxelSize.getValue() * gridOffset;
    unsigned int realGridSizeX = grid.width() - 2*gridOffset;
    unsigned int realGridSizeY = grid.height() - 2*gridOffset;
    unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;
    SCoord dimGrid(realGridSizeX, realGridSizeY, realGridSizeZ);

    unsigned int iX = (axis+1)%3;
    unsigned int iY = (axis+2)%3;
    unsigned int iZ = axis;

    float vSX = voxelSize.getValue()[iX];
    float vSY = voxelSize.getValue()[iY];
    float vSZ = voxelSize.getValue()[iZ];
    unsigned int nbVertices = 0;
    unsigned int nbFaces = 0;
    int maxX = dimGrid[iX];
    int maxY = dimGrid[iY];
    for (int x = 0; x < maxX; ++x)
    {
        for (int y = 0; y < maxY; ++y)
        {
            if( x > 0 && y > 0)
            {
                valuesIndices[4*nbIndicesOffset+4*nbFaces  ] = nbVerticesOffset+nbVertices;
                valuesIndices[4*nbIndicesOffset+4*nbFaces+1] = nbVerticesOffset+nbVertices-maxY;
                valuesIndices[4*nbIndicesOffset+4*nbFaces+2] = nbVerticesOffset+nbVertices-maxY-1;
                valuesIndices[4*nbIndicesOffset+4*nbFaces+3] = nbVerticesOffset+nbVertices-1;
                nbFaces++;
            }
            valuesVertices[3*nbVerticesOffset+3*nbVertices+iX] = ori[iX] + x*vSX;
            valuesVertices[3*nbVerticesOffset+3*nbVertices+iY] = ori[iY] + y*vSY;
            valuesVertices[3*nbVerticesOffset+3*nbVertices+iZ] = ori[iZ] + (showPlane.getValue()[iZ]-gridOffset+.5)*vSZ;
            nbVertices++;
        }
    }

    /*
    if( axis == 2)
    {
        std::cout << "nbVerticesOffset: " << nbVerticesOffset << std::endl;
        std::cout << "nbIndicesOffset: " << nbIndicesOffset << std::endl;
        for( int i = 0; i < maxX*maxY; ++i)
            std::cout << "vertex["<<i<<"]" << valuesVertices[3*nbVerticesOffset+3*i] << ", " << valuesVertices[3*nbVerticesOffset+3*i+1] << ", " << valuesVertices[3*nbVerticesOffset+3*i+2] << std::endl;
        for( int i = 0; i < (maxX-1)*(maxY-1); ++i)
            std::cout << "face["<<i<<"]" << valuesIndices[4*nbIndicesOffset+4*i] << ", " << valuesIndices[4*nbIndicesOffset+4*i+1] << ", " << valuesIndices[4*nbIndicesOffset+4*i+2] << ", " << valuesIndices[4*nbIndicesOffset+4*i+3] << std::endl;
    }*/
    /*
    // Compute face normals
    GLfloat faceNormals[3*(maxX-1)*(maxY-1)];
    nbFaces = 0;
    for (int x = 0; x < maxX; ++x)
    {
        for (int y = 0; y < maxY; ++y)
        {
            if( x > 0 && y > 0)
            {
                faceNormals[3*nbFaces  ] = ;
                faceNormals[3*nbFaces+1] = ;
                faceNormals[3*nbFaces+2] = ;
                nbFaces++;
            }
        }
    }*/

    unsigned int nbElt = 0;
    for (int x = 0; x < maxX; ++x)
    {
        for (int y = 0; y < maxY; ++y)
        {
            // TODO compute each face normal, then vertex normal
            for (int i = 0; i < 3; ++i)
            {
                valuesNormals[3*nbVerticesOffset+3*nbElt+i] = (i == axis)?1.0:0.0;
            }
            valuesColors[3*nbVerticesOffset+3*nbElt+0] = 0.5;
            valuesColors[3*nbVerticesOffset+3*nbElt+1] = 0.5;
            valuesColors[3*nbVerticesOffset+3*nbElt+2] = 0.5;
            nbElt++;

            /*
            // Compute Laplacian
            float n[3];
            unsigned int index = nbNormals;
            for (unsigned int w = 0; w < 3; ++w)
            {
                n[w] = 0.0f;
                float nbContrib = 0;
                if (x >   0   ) {n[w] += valuesVertices[vertexOffset+index-maxY]; nbContrib++;}
                if (x < maxX-1) {n[w] += valuesVertices[vertexOffset+index+maxY]; nbContrib++;}
                if (y >   0   ) {n[w] += valuesVertices[vertexOffset+index-1]; nbContrib++;}
                if (y < maxY-1) {n[w] += valuesVertices[vertexOffset+index+1]; nbContrib++;}
                n[w] /= nbContrib; index++;
            }
            // Normalize
            float norm = sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
            valuesNormals[vertexOffset+nbNormals] = n[0] / norm; nbNormals++;
            valuesNormals[vertexOffset+nbNormals] = n[1] / norm; nbNormals++;
            valuesNormals[vertexOffset+nbNormals] = n[2] / norm; nbNormals++;
            */
        }
    }
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::updateValuesVBO() const
{
    unsigned int realGridSizeX = grid.width() - 2*gridOffset;
    unsigned int realGridSizeY = grid.height() - 2*gridOffset;
    unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;
    unsigned int vertexSize = 3*(realGridSizeY * realGridSizeZ + realGridSizeX * realGridSizeZ + realGridSizeX * realGridSizeY);
    unsigned int showvox=this->showVoxels.getValue().getSelectedId();
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboValuesId1);
    float *ptr = (float*)glMapBufferARB(GL_ARRAY_BUFFER_ARB, GL_WRITE_ONLY_ARB);
    if(ptr)
    {
        unsigned int offSet[] = {0,(realGridSizeY * realGridSizeZ),(realGridSizeY * realGridSizeZ + realGridSizeX * realGridSizeZ)};
        const SCoord& ori = origin.getValue() + voxelSize.getValue() * gridOffset;
        SCoord dimGrid(realGridSizeX, realGridSizeY, realGridSizeZ);
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            unsigned int iX = (axis+1)%3;
            unsigned int iY = (axis+2)%3;
            unsigned int iZ = axis;
            int zCoord = showPlane.getValue()[iZ]-gridOffset;
            if(zCoord < 0 || zCoord > dimGrid[axis]) continue;
            float vSZ = voxelSize.getValue()[iZ];
            float scaleZ = show3DValuesHeight.getValue() * vSZ;
            int maxX = dimGrid[iX];
            int maxY = dimGrid[iY];
            unsigned int nbVertices = 0;
            for (int x = 0; x < maxX; ++x)
            {
                for (int y = 0; y < maxY; ++y)
                {
                    // Vertices
                    SCoord coord;
                    coord[iX] = gridOffset+x;
                    coord[iY] = gridOffset+y;
                    coord[iZ] = gridOffset+zCoord;
                    float value = getLabel(coord[0],coord[1],coord[2]);
                    if( value < 0 ) value = 0;
                    if( value > maxValues[showvox] ) value = maxValues[showvox];
                    ptr[3*offSet[axis]+3*nbVertices+iZ] = ori[iZ] + (zCoord+.5)*vSZ + value / maxValues[showvox] * scaleZ;

                    // Normals
                    //ptr[vertexSize+3*offSet[axis]+3*nbVertices+iX] = ;
                    //ptr[vertexSize+3*offSet[axis]+3*nbVertices+iY] = ;
                    //ptr[vertexSize+3*offSet[axis]+3*nbVertices+iZ] = ;

                    // Colors
                    float color[4];
                    getColor( color, value);
                    ptr[2*vertexSize+3*offSet[axis]+3*nbVertices+0] = color[0];
                    ptr[2*vertexSize+3*offSet[axis]+3*nbVertices+1] = color[1];
                    ptr[2*vertexSize+3*offSet[axis]+3*nbVertices+2] = color[2];

                    nbVertices++;
                }
            }
        }
        glUnmapBufferARB(GL_ARRAY_BUFFER_ARB);
    }
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::displayValuesVBO() const
{
    unsigned int realGridSizeX = grid.width() - 2*gridOffset;
    unsigned int realGridSizeY = grid.height() - 2*gridOffset;
    unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;
    unsigned int vertexSize = 3*(realGridSizeY * realGridSizeZ + realGridSizeX * realGridSizeZ + realGridSizeX * realGridSizeY);

    glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT); // Save context

    updateValuesVBO();

    // before draw, specify vertex and index arrays with their offsets
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, vboValuesId1);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glNormalPointer(GL_FLOAT, 0, (void*)vertexSize);
    glColorPointer(3, GL_FLOAT, 0, (void*)(2*vertexSize));
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboValuesId2);
    glIndexPointer(GL_UNSIGNED_SHORT, 0, 0);

    // Enable VBO
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    //float colorPlane[] = {1.0, 1.0, 1.0, 1.0};
    //glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,colorPlane);

    // use only offset here instead of absolute pointer addresses
    if(showPlane.getValue()[0] != -1)
    {
        glDrawElements(GL_QUADS, 4*(realGridSizeY-1)*(realGridSizeZ-1), GL_UNSIGNED_SHORT, (GLushort*)0+0);
    }
    if(showPlane.getValue()[1] != -1)
    {
        glDrawElements(GL_QUADS, 4*(realGridSizeX-1)*(realGridSizeZ-1), GL_UNSIGNED_SHORT, (GLushort*)0+4*(realGridSizeY-1)*(realGridSizeZ-1));
    }
    if(showPlane.getValue()[2] != -1)
    {
        glDrawElements(GL_QUADS, 4*(realGridSizeX-1)*(realGridSizeY-1), GL_UNSIGNED_SHORT, (GLushort*)0+4*((realGridSizeY-1)*(realGridSizeZ-1)+(realGridSizeX-1)*(realGridSizeZ-1)));
    }

    // Disable VBO
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    // Bind the buffers to 0 by safety and restore ARRAY context
    glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
    glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);
    glPopClientAttrib();
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::drawPlaneBBox( const int& axis) const
{
    const SCoord& ori = origin.getValue() + voxelSize.getValue() * gridOffset - voxelSize.getValue()*.5;
    unsigned int iX = (axis+1)%3;
    unsigned int iY = (axis+2)%3;
    unsigned int iZ = axis;

    unsigned int realGridSizeX = grid.width() - 2*gridOffset;
    unsigned int realGridSizeY = grid.height() - 2*gridOffset;
    unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;
    SCoord dimGrid(realGridSizeX, realGridSizeY, realGridSizeZ);

    const SCoord& vSize = voxelSize.getValue();

    SCoord minBox, maxBox;
    minBox[iX] = ori[iX];
    minBox[iY] = ori[iY];
    minBox[iZ] = ori[iZ] + (showPlane.getValue()[iZ]-gridOffset+1)*vSize[iZ];
    maxBox[iX] = ori[iX] + dimGrid[iX] * vSize[iX];
    maxBox[iY] = ori[iY] + dimGrid[iY] * vSize[iY];
    maxBox[iZ] = ori[iZ] + (showPlane.getValue()[iZ]-gridOffset+1+show3DValuesHeight.getValue())*vSize[iZ];
    glLineWidth(2.0);
    glBegin (GL_LINES);
    glVertex3f(minBox[0],minBox[1],minBox[2]);
    glVertex3f(maxBox[0],minBox[1],minBox[2]);
    glVertex3f(minBox[0],minBox[1],minBox[2]);
    glVertex3f(minBox[0],maxBox[1],minBox[2]);
    glVertex3f(minBox[0],minBox[1],minBox[2]);
    glVertex3f(minBox[0],minBox[1],maxBox[2]);
    glVertex3f(maxBox[0],maxBox[1],maxBox[2]);
    glVertex3f(minBox[0],maxBox[1],maxBox[2]);
    glVertex3f(maxBox[0],maxBox[1],maxBox[2]);
    glVertex3f(maxBox[0],minBox[1],maxBox[2]);
    glVertex3f(maxBox[0],maxBox[1],maxBox[2]);
    glVertex3f(maxBox[0],maxBox[1],minBox[2]);
    glVertex3f(minBox[0],maxBox[1],minBox[2]);
    glVertex3f(minBox[0],maxBox[1],maxBox[2]);
    glVertex3f(minBox[0],maxBox[1],minBox[2]);
    glVertex3f(maxBox[0],maxBox[1],minBox[2]);
    glVertex3f(maxBox[0],minBox[1],minBox[2]);
    glVertex3f(maxBox[0],maxBox[1],minBox[2]);
    glVertex3f(maxBox[0],minBox[1],minBox[2]);
    glVertex3f(maxBox[0],minBox[1],maxBox[2]);
    glVertex3f(minBox[0],minBox[1],maxBox[2]);
    glVertex3f(maxBox[0],minBox[1],maxBox[2]);
    glVertex3f(minBox[0],minBox[1],maxBox[2]);
    glVertex3f(minBox[0],maxBox[1],maxBox[2]);
    glEnd();
    glLineWidth(1.0);
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::genListCube()
{
    cubeList = glGenLists(1);
    glNewList(cubeList, GL_COMPILE);

    glBegin(GL_QUADS);
    glNormal3f(1,0,0); glVertex3d(1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(1.0,1.0,-1.0);
    glNormal3f(-1,0,0); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(-1.0,-1.0,1.0); glVertex3d(-1.0,1.0,1.0); glVertex3d(-1.0,1.0,-1.0);
    glNormal3f(0,1,0); glVertex3d(-1.0,1.0,-1.0); glVertex3d(1.0,1.0,-1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(-1.0,1.0,1.0);
    glNormal3f(0,-1,0); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,1.0); glVertex3d(-1.0,-1.0,1.0);
    glNormal3f(0,0,1); glVertex3d(-1.0,-1.0,1.0); glVertex3d(-1.0,1.0,1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(1.0,-1.0,1.0);
    glNormal3f(0,0,-1); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(-1.0,1.0,-1.0); glVertex3d(1.0,1.0,-1.0); glVertex3d(1.0,-1.0,-1.0);
    glEnd ();
    glEndList();

    wcubeList = glGenLists(1);
    glNewList(wcubeList, GL_COMPILE);
    glBegin(GL_LINE_LOOP);
    glNormal3f(1,0,0); glVertex3d(1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(1.0,1.0,-1.0);
    glEnd ();
    glBegin(GL_LINE_LOOP);
    glNormal3f(-1,0,0); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(-1.0,-1.0,1.0); glVertex3d(-1.0,1.0,1.0); glVertex3d(-1.0,1.0,-1.0);
    glEnd ();
    glBegin(GL_LINE_LOOP);
    glNormal3f(0,1,0); glVertex3d(-1.0,1.0,-1.0); glVertex3d(1.0,1.0,-1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(-1.0,1.0,1.0);
    glEnd ();
    glBegin(GL_LINE_LOOP);
    glNormal3f(0,-1,0); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,-1.0); glVertex3d(1.0,-1.0,1.0); glVertex3d(-1.0,-1.0,1.0);
    glEnd ();
    glBegin(GL_LINE_LOOP);
    glNormal3f(0,0,1); glVertex3d(-1.0,-1.0,1.0); glVertex3d(-1.0,1.0,1.0); glVertex3d(1.0,1.0,1.0); glVertex3d(1.0,-1.0,1.0);
    glEnd ();
    glBegin(GL_LINE_LOOP);
    glNormal3f(0,0,-1); glVertex3d(-1.0,-1.0,-1.0); glVertex3d(-1.0,1.0,-1.0); glVertex3d(1.0,1.0,-1.0); glVertex3d(1.0,-1.0,-1.0);
    glEnd ();
    glEndList();

}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::displayValues() const
{
    for(unsigned int i = 0; i < 3; ++i)
        if(showPlane.getValue()[i] != -1)
            displayPlane(i);
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::displayPlane( const int& axis) const
{
    unsigned int showvox=this->showVoxels.getValue().getSelectedId();
    const SCoord& ori = origin.getValue() + voxelSize.getValue() * gridOffset;
    unsigned int realGridSizeX = grid.width() - 2*gridOffset;
    unsigned int realGridSizeY = grid.height() - 2*gridOffset;
    unsigned int realGridSizeZ = grid.depth() - 2*gridOffset;
    SCoord dimGrid(realGridSizeX, realGridSizeY, realGridSizeZ);

    unsigned int iX = (axis+1)%3;
    unsigned int iY = (axis+2)%3;
    unsigned int iZ = axis;

    float vSX = voxelSize.getValue()[iX];
    float vSY = voxelSize.getValue()[iY];
    float vSZ = voxelSize.getValue()[iZ];
    int maxX = dimGrid[iX];
    int maxY = dimGrid[iY];

    // Simple normal
    float normal[3];
    normal[iZ] = (show3DValuesHeight.getValue() > 0)?1.0:-1.0;
    glNormal3f(normal[0],normal[1],normal[2]);

    float halfVoxelOffset = (show3DValuesHeight.getValue() > 0)?0.51:-0.51;
    float scaleZ = show3DValuesHeight.getValue() * vSZ;
    float zCoord = showPlane.getValue()[iZ]-gridOffset;
    SCoord coord, coord2;
    float value;
    float color[4];
    glBegin( GL_QUADS);
    for (int x = 1; x < maxX; ++x)
    {
        for (int y = 1; y < maxY; ++y)
        {
            // Vertex 0 //
            // Value
            coord2[iX] = gridOffset+x;
            coord2[iY] = gridOffset+y;
            coord2[iZ] = gridOffset+zCoord;
            value = getLabel(coord2[0],coord2[1],coord2[2]);
            if( value < 0 ) value = 0;
            if( value > maxValues[showvox] ) value = maxValues[showvox];

            // Color
            getColor( color, value);
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);

            // Position
            coord[iX] = ori[iX] + x*vSX;
            coord[iY] = ori[iY] + y*vSY;
            coord[iZ] = ori[iZ] + (zCoord+halfVoxelOffset)*vSZ + value / maxValues[showvox] * scaleZ;
            glVertex3f(coord[0],coord[1],coord[2]);

            // Vertex 1 //
            // Value
            coord2[iX] = gridOffset+x-1;
            coord2[iY] = gridOffset+y;
            coord2[iZ] = gridOffset+zCoord;
            value = getLabel(coord2[0],coord2[1],coord2[2]);
            if( value < 0 ) value = 0;
            if( value > maxValues[showvox] ) value = maxValues[showvox];

            // Color
            getColor( color, value);
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);

            // Position
            coord[iX] = ori[iX] + (x-1)*vSX;
            coord[iY] = ori[iY] + y*vSY;
            coord[iZ] = ori[iZ] + (zCoord+halfVoxelOffset)*vSZ + value / maxValues[showvox] * scaleZ;
            glVertex3f(coord[0],coord[1],coord[2]);

            // Vertex 2 //
            // Value
            coord2[iX] = gridOffset+x-1;
            coord2[iY] = gridOffset+y-1;
            coord2[iZ] = gridOffset+zCoord;
            value = getLabel(coord2[0],coord2[1],coord2[2]);
            if( value < 0 ) value = 0;
            if( value > maxValues[showvox] ) value = maxValues[showvox];

            // Color
            getColor( color, value);
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);

            // Position
            coord[iX] = ori[iX] + (x-1)*vSX;
            coord[iY] = ori[iY] + (y-1)*vSY;
            coord[iZ] = ori[iZ] + (zCoord+halfVoxelOffset)*vSZ + value / maxValues[showvox] * scaleZ;
            glVertex3f(coord[0],coord[1],coord[2]);

            // Vertex 3 //
            // Value
            coord2[iX] = gridOffset+x;
            coord2[iY] = gridOffset+y-1;
            coord2[iZ] = gridOffset+zCoord;
            value = getLabel(coord2[0],coord2[1],coord2[2]);
            if( value < 0 ) value = 0;
            if( value > maxValues[showvox] ) value = maxValues[showvox];

            // Color
            getColor( color, value);
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);

            // Position
            coord[iX] = ori[iX] + x*vSX;
            coord[iY] = ori[iY] + (y-1)*vSY;
            coord[iZ] = ori[iZ] + (zCoord+halfVoxelOffset)*vSZ + value / maxValues[showvox] * scaleZ;
            glVertex3f(coord[0],coord[1],coord[2]);
        }
    }
    glEnd();
}


template < class MaterialTypes>
void GridMaterial< MaterialTypes>::updateMaxValues()
{
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Update_Max_Values");
#endif

    // Determine max values
    unsigned int i;
    for (i=0; i < 11; ++i) maxValues[i] = -1.0f;
    for (i=0; i<nbVoxels; i++) if ((float)grid.data()[i]>maxValues[SHOWVOXELS_DATAVALUE]) maxValues[SHOWVOXELS_DATAVALUE]=(float)grid.data()[i];
    bool rigidPart = false;
    for (typename mapLabelType::const_iterator it = labelToStiffnessPairs.getValue().begin(); it != labelToStiffnessPairs.getValue().end(); ++it)
    {
        if (isRigid(it->second))
            rigidPart = true;
        else if (it->second > maxValues[SHOWVOXELS_STIFFNESS])
            maxValues[SHOWVOXELS_STIFFNESS]=(float)it->second;
    }
    if (rigidPart) maxValues[SHOWVOXELS_STIFFNESS] *= 1.2;
    for (typename mapLabelType::const_iterator it = labelToDensityPairs.getValue().begin(); it != labelToDensityPairs.getValue().end(); ++it) if (it->second > maxValues[SHOWVOXELS_DENSITY]) maxValues[SHOWVOXELS_DENSITY]=(float)it->second;
    for (typename mapLabelType::const_iterator it = labelToBulkModulusPairs.getValue().begin(); it != labelToBulkModulusPairs.getValue().end(); ++it) if (it->second > maxValues[SHOWVOXELS_BULKMODULUS]) maxValues[SHOWVOXELS_BULKMODULUS]=(float)it->second;
    for (typename mapLabelType::const_iterator it = labelToPoissonRatioPairs.getValue().begin(); it != labelToPoissonRatioPairs.getValue().end(); ++it) if (it->second > maxValues[SHOWVOXELS_POISSONRATIO]) maxValues[SHOWVOXELS_POISSONRATIO]=(float)it->second;
    for (i=0; i<voronoi.size(); i++) if (grid.data()[i] && voronoi[i]+1>maxValues[SHOWVOXELS_VORONOI]) maxValues[SHOWVOXELS_VORONOI]=(float)voronoi[i]+1;
    for (i=0; i<distances.size(); i++) if (grid.data()[i] && distances[i]>maxValues[SHOWVOXELS_DISTANCES]) maxValues[SHOWVOXELS_DISTANCES]=(float)distances[i];
    for (i=0; i<voronoi_frames.size(); i++) if (grid.data()[i] && voronoi_frames[i]+1>maxValues[SHOWVOXELS_VORONOI_FR]) maxValues[SHOWVOXELS_VORONOI_FR]=(float)voronoi_frames[i]+1;
    maxValues[SHOWVOXELS_WEIGHTS]=1.0f;
    for (i=0; i<linearityError.size(); i++) if (grid.data()[i] && linearityError[i]>maxValues[SHOWVOXELS_LINEARITYERROR]) maxValues[SHOWVOXELS_LINEARITYERROR]=(float)linearityError[i];
    maxValues[SHOWVOXELS_FRAMESINDICES]=nbRef+1.;
    for (unsigned int i = 0; i < showVoxels.getValue().size(); ++i)
        sout << "maxValues["<<showVoxels.getValue()[i]<<"]: " << maxValues[i] << sendl;

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Update_Max_Values");
#endif
}


/*************************/
/*         Utils         */
/*************************/
template < class MaterialTypes>
int GridMaterial< MaterialTypes>::getIndex(const GCoord& icoord) const
{
    if (!nbVoxels) return -1;
    for (int i=0; i<3; i++) if (icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return -1; // invalid icoord (out of grid)
    return icoord[0]+dimension.getValue()[0]*(icoord[1]+dimension.getValue()[1]*icoord[2]);
}

template < class MaterialTypes>
int GridMaterial< MaterialTypes>::getIndex(const SCoord& coord) const
{
    if (!nbVoxels) return -1;
    GCoord icoord;
    if (!getiCoord(coord,icoord)) return -1;
    return getIndex(icoord);
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getiCoord(const SCoord& coord, GCoord& icoord) const
{
    if (!nbVoxels) return false;
    Real val;
//                cerr<<"GridMaterial< MaterialTypes>::getiCoord, coord = "<< coord <<", origin = "<< origin.getValue() <<", voxelSize = " << voxelSize.getValue() << endl;
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
bool GridMaterial< MaterialTypes>::getiCoord(const int& index, GCoord& icoord) const
{
    if (!nbVoxels) return false;
    if (index<0 || index>=(int)nbVoxels) return false; // invalid index
    icoord[2]=index/(dimension.getValue()[0]*dimension.getValue()[1]);
    icoord[1]=(index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1])/dimension.getValue()[0];
    icoord[0]=index-icoord[2]*dimension.getValue()[0]*dimension.getValue()[1]-icoord[1]*dimension.getValue()[0];
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getCoord(const GCoord& icoord, SCoord& coord) const
{
    if (!nbVoxels) return false;
    for (unsigned int i=0; i<3; i++) if (icoord[i]<0 || icoord[i]>=dimension.getValue()[i]) return false; // invalid icoord (out of grid)
    coord=this->origin.getValue();
    for (unsigned int i=0; i<3; i++) coord[i]+=(Real)this->voxelSize.getValue()[i]*(Real)icoord[i];
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::getCoord(const int& index, SCoord& coord) const
{
    if (!nbVoxels) return false;
    GCoord icoord;
    if (!getiCoord(index,icoord)) return false;
    else return getCoord(icoord,coord);
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get6Neighbors ( const int& index, VUI& neighbors ) const
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    GCoord icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]+=1;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get18Neighbors ( const int& index, VUI& neighbors ) const
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    GCoord icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]+=1;
    }

    for (unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]-=2;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    return true;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::get26Neighbors ( const int& index, VUI& neighbors ) const
{
    neighbors.clear();
    if (!nbVoxels) return false;
    int i;
    GCoord icoord;
    if (!getiCoord(index,icoord)) return false;
    for (unsigned int j=0; j<3 ; j++)
    {
        icoord[j]+=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]-=1;
        icoord[j]-=1;
        i=getIndex(icoord);
        if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
        icoord[j]+=1;
    }
    for (unsigned int k=0; k<3 ; k++)
    {
        icoord[k]+=1;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]-=2;
        for (unsigned int j=k+1; j<3 ; j++)
        {
            icoord[j]+=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]-=1;
            icoord[j]-=1;
            i=getIndex(icoord);
            if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
            icoord[j]+=1;
        }
        icoord[k]+=1;
    }
    icoord[0]+=1;
    icoord[1]+=1;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]-=2;
    icoord[1]+=1;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]-=2;
    icoord[2]+=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]-=1;
    icoord[2]-=1;
    i=getIndex(icoord);
    if (i!=-1) /*if (grid.data()[i])*/ neighbors.push_back(i);
    icoord[2]+=1;
    icoord[1]+=1;
    icoord[0]+=1;
    return true;
}



template < class MaterialTypes>
void GridMaterial< MaterialTypes>::getCompleteBasis(const SCoord& p,const unsigned int order,vector<Real>& basis) const
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'
    basis.resize(dim);
    for (j=0; j<dim; j++) basis[j]=0;

    SCoord p2;
    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    SCoord p3;
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
void GridMaterial< MaterialTypes>::getCompleteBasisDeriv(const SCoord& p,const unsigned int order,vector<SGradient>& basisDeriv) const
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    basisDeriv.resize(dim);
    for (j=0; j<dim; j++) basisDeriv[j].fill(0);

    SCoord p2;
    for (j=0; j<3; j++) p2[j]=p[j]*p[j];
    SCoord p3;
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
void GridMaterial< MaterialTypes>::getCompleteBasisDeriv2(const SCoord& p,const unsigned int order,vector<SHessian>& basisDeriv) const
{
    unsigned int j,k,count=0,dim=(order+1)*(order+2)*(order+3)/6; // complete basis of order 'order'

    basisDeriv.resize(dim);
    for (k=0; k<dim; k++) basisDeriv[k].fill(0);

    SCoord p2;
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

template < class MaterialTypes>
typename GridMaterial< MaterialTypes>::Real GridMaterial< MaterialTypes>::findWeightInRepartition(const unsigned int& pointIndex, const unsigned int& frameIndex)
{
    if (v_index.size()<=pointIndex) return 0;
    for ( unsigned int j = 0; j < nbRef && v_weights[pointIndex][j]!=0 ; ++j)
        if ( v_index[pointIndex][j] == frameIndex)
            return v_weights[pointIndex][j];
    return 0;
}

template < class MaterialTypes>
bool GridMaterial< MaterialTypes>::areRepsSimilar(const unsigned int i1,const unsigned int i2)
{
    Vec<nbRef,bool> checked; checked.fill(false);
    unsigned int i,j;
    for(i=0; i<nbRef; i++)
        if(v_weights[i1][i])
        {
            j=0; while(j<nbRef && v_index[i1][i]!=v_index[i2][j]) j++;
            if(j==nbRef) return false;
            if(v_weights[i2][j]==0) return false;
            checked[j]=true;
        }
    for(j=0; j<nbRef; j++) if(!checked[j] && v_weights[i2][j]!=0) return false;
    return true;
}

} // namespace material

} // namespace component

} // namespace sofa

#endif


