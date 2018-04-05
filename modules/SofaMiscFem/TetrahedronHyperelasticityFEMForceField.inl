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

#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD_INL

#include <sofa/helper/system/gl.h>
#include <SofaMiscFem/BoyceAndArruda.h>
#include <SofaMiscFem/NeoHookean.h>
#include <SofaMiscFem/MooneyRivlin.h>
#include <SofaMiscFem/VerondaWestman.h>
#include <SofaMiscFem/STVenantKirchhoff.h>
#include <SofaMiscFem/HyperelasticMaterial.h>
#include <SofaMiscFem/Costa.h>
#include <SofaMiscFem/Ogden.h>
#include "TetrahedronHyperelasticityFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <sofa/core/behavior/ForceField.inl>
#include <SofaBaseTopology/TopologyData.inl>
#include <algorithm>
#include <iterator>
namespace sofa
{
namespace component
{
namespace forcefield
{
using namespace sofa::defaulttype;
using namespace	sofa::component::topology;
using namespace core::topology;


template< class DataTypes >
void TetrahedronHyperelasticityFEMForceField<DataTypes>::TetrahedronHandler::applyCreateFunction(unsigned int tetrahedronIndex,
                                                                                              TetrahedronRestInformation &tinfo,
                                                                                              const Tetrahedron &,
                                                                                              const sofa::helper::vector<unsigned int> &,
                                                                                              const sofa::helper::vector<double> &)
{

  if (ff) {
      const vector< Tetrahedron > &tetrahedronArray=ff->m_topology->getTetrahedra() ;
      const std::vector< Edge> &edgeArray=ff->m_topology->getEdges() ;
      unsigned int j;
//      int k;
      typename DataTypes::Real volume;
      typename DataTypes::Coord point[4];
      const typename DataTypes::VecCoord restPosition = ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

      ///describe the indices of the 4 tetrahedron vertices
      const Tetrahedron &t= tetrahedronArray[tetrahedronIndex];
      BaseMeshTopology::EdgesInTetrahedron te=ff->m_topology->getEdgesInTetrahedron(tetrahedronIndex);

      // store the point position

      for(j=0;j<4;++j)
          point[j]=(restPosition)[t[j]];
      /// compute 6 times the rest volume
      volume=dot(cross(point[2]-point[0],point[3]-point[0]),point[1]-point[0]);
      /// store the rest volume
      tinfo.m_volScale =(Real)(1.0/volume);
      tinfo.m_restVolume = fabs(volume/6);
      // store shape vectors at the rest configuration
      for(j=0;j<4;++j) {
          if (!(j%2))
              tinfo.m_shapeVector[j]=-cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/ volume;
          else
              tinfo.m_shapeVector[j]=cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/ volume;;
      }


      for(j=0;j<6;++j) {
          Edge e=ff->m_topology->getLocalEdgesInTetrahedron(j);
          int k=e[0];
          //int l=e[1];
          if (edgeArray[te[j]][0]!=t[k]) {
              k=e[1];
              //l=e[0];
          }
      }


  }//end if(ff)

}

template <class DataTypes> TetrahedronHyperelasticityFEMForceField<DataTypes>::TetrahedronHyperelasticityFEMForceField() 
    : m_topology(0)
    , m_initialPoints(0)
    , m_updateMatrix(true)
    , m_meshSaved( false)
    , d_stiffnessMatrixRegularizationWeight(initData(&d_stiffnessMatrixRegularizationWeight, (bool)false,"matrixRegularization","Regularization of the Stiffness Matrix (between true or false)"))
    , d_materialName(initData(&d_materialName,std::string("ArrudaBoyce"),"materialName","the name of the material to be used"))
    , d_parameterSet(initData(&d_parameterSet,"ParameterSet","The global parameters specifying the material"))
    , d_anisotropySet(initData(&d_anisotropySet,"AnisotropyDirections","The global directions of anisotropy of the material"))
    , m_tetrahedronInfo(initData(&m_tetrahedronInfo, "tetrahedronInfo", "Internal tetrahedron data"))
    , m_edgeInfo(initData(&m_edgeInfo, "edgeInfo", "Internal edge data"))
    , m_tetrahedronHandler(NULL)
{
    m_tetrahedronHandler = new TetrahedronHandler(this,&m_tetrahedronInfo);
}

template <class DataTypes> TetrahedronHyperelasticityFEMForceField<DataTypes>::~TetrahedronHyperelasticityFEMForceField()
{
    if(m_tetrahedronHandler) delete m_tetrahedronHandler;
}

template <class DataTypes> void TetrahedronHyperelasticityFEMForceField<DataTypes>::init()
{
    if (this->f_printLog.getValue())
        msg_info() << "initializing TetrahedronHyperelasticityFEMForceField";

    this->Inherited::init();

    /** parse the parameter set */
    SetParameterArray paramSet=d_parameterSet.getValue();
    if (paramSet.size()>0) {
            globalParameters.parameterArray.resize(paramSet.size());
            copy(paramSet.begin(), paramSet.end(),globalParameters.parameterArray.begin());
    }
    /** parse the anisotropy Direction set */
    SetAnisotropyDirectionArray anisotropySet=d_anisotropySet.getValue();
    if (anisotropySet.size()>0) {
            globalParameters.anisotropyDirection.resize(anisotropySet.size());
            copy(anisotropySet.begin(), anisotropySet.end(),globalParameters.anisotropyDirection.begin());
    }

    m_topology = this->getContext()->getMeshTopology();


    /** parse the input material name */
    string material = d_materialName.getValue();
    if (material=="ArrudaBoyce")
    {
        fem::BoyceAndArruda<DataTypes> *BoyceAndArrudaMaterial = new fem::BoyceAndArruda<DataTypes>;
        m_myMaterial = BoyceAndArrudaMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else if (material=="StVenantKirchhoff")
    {
        fem::STVenantKirchhoff<DataTypes> *STVenantKirchhoffMaterial = new fem::STVenantKirchhoff<DataTypes>;
        m_myMaterial = STVenantKirchhoffMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else if (material=="NeoHookean")
    {
        fem::NeoHookean<DataTypes> *NeoHookeanMaterial = new fem::NeoHookean<DataTypes>;
        m_myMaterial = NeoHookeanMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else if (material=="MooneyRivlin")
    {
        fem::MooneyRivlin<DataTypes> *MooneyRivlinMaterial = new fem::MooneyRivlin<DataTypes>;
        m_myMaterial = MooneyRivlinMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else if (material=="VerondaWestman")
    {
        fem::VerondaWestman<DataTypes> *VerondaWestmanMaterial = new fem::VerondaWestman<DataTypes>;
        m_myMaterial = VerondaWestmanMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }

    else if (material=="Costa")
    {
        fem::Costa<DataTypes> *CostaMaterial = new fem::Costa<DataTypes>;
        m_myMaterial = CostaMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else if (material=="Ogden")
    {
        fem::Ogden<DataTypes> *OgdenMaterial = new fem::Ogden<DataTypes>;
        m_myMaterial = OgdenMaterial;
        if (this->f_printLog.getValue())
            msg_info()<<"The model is "<<material;
    }
    else
    {
        msg_error() << "material name " << material << " is not valid (should be ArrudaBoyce, StVenantKirchhoff, MooneyRivlin, VerondaWestman, Costa or Ogden)";
    }


    if (!m_topology->getNbTetrahedra())
    {
        msg_error() << "ERROR(TetrahedronHyperelasticityFEMForceField): object must have a Tetrahedral Set Topology.\n";
        return;
    }

    helper::vector<typename TetrahedronHyperelasticityFEMForceField<DataTypes>::TetrahedronRestInformation>& tetrahedronInf = *(m_tetrahedronInfo.beginEdit());

    /// prepare to store info in the triangle array
    tetrahedronInf.resize(m_topology->getNbTetrahedra());

    helper::vector<typename TetrahedronHyperelasticityFEMForceField<DataTypes>::EdgeInformation>& edgeInf = *(m_edgeInfo.beginEdit());

    edgeInf.resize(m_topology->getNbEdges());
    m_edgeInfo.createTopologicalEngine(m_topology);

    m_edgeInfo.registerTopologicalData();

    m_edgeInfo.endEdit();

    // get restPosition
    if (m_initialPoints.size() == 0)
    {
    const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
            m_initialPoints=p;
    }
    int i;

    /// initialize the data structure associated with each tetrahedron
    for (i=0;i<m_topology->getNbTetrahedra();++i)
    {
        m_tetrahedronHandler->applyCreateFunction(i, tetrahedronInf[i],
                                                m_topology->getTetrahedron(i),  (const vector< unsigned int > )0,
                                                (const vector< double >)0);
    }

    /// set the call back function upon creation of a tetrahedron
    m_tetrahedronInfo.createTopologicalEngine(m_topology,m_tetrahedronHandler);
    m_tetrahedronInfo.registerTopologicalData();

    m_tetrahedronInfo.endEdit();
    //testDerivatives();

}

template <class DataTypes> 
void TetrahedronHyperelasticityFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();


    const bool printLog = this->f_printLog.getValue();
    if (printLog && !m_meshSaved)
    {
        saveMesh( "D:/Steph/sofa-result.stl" );
        printf( "Mesh saved.\n" );
        m_meshSaved = true;
    }
    unsigned int i=0,j=0,k=0,l=0;
    unsigned int nbTetrahedra=m_topology->getNbTetrahedra();

    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(m_tetrahedronInfo.beginEdit());


    TetrahedronRestInformation *tetInfo;

    assert(this->mstate);

    Coord dp[3],x0,sv;


    for(i=0; i<nbTetrahedra; i++ )
    {
        tetInfo=&tetrahedronInf[i];
        const Tetrahedron &ta= m_topology->getTetrahedron(i);

        x0=x[ta[0]];

        // compute the deformation gradient
        // deformation gradient = sum of tensor product between vertex position and shape vector
        // optimize by using displacement with first vertex
        dp[0]=x[ta[1]]-x0;
        sv=tetInfo->m_shapeVector[1];
        for (k=0;k<3;++k) {
                for (l=0;l<3;++l) {
                        tetInfo->m_deformationGradient[k][l]=dp[0][k]*sv[l];
                }
        }
        for (j=1;j<3;++j) {
                dp[j]=x[ta[j+1]]-x0;
                sv=tetInfo->m_shapeVector[j+1];
                for (k=0;k<3;++k) {
                        for (l=0;l<3;++l) {
                                tetInfo->m_deformationGradient[k][l]+=dp[j][k]*sv[l];
                        }
                }
        }

        /// compute the right Cauchy-Green deformation matrix
        for (k=0;k<3;++k) {
            for (l=k;l<3;++l) {
                tetInfo->deformationTensor(k,l)=(tetInfo->m_deformationGradient(0,k)*tetInfo->m_deformationGradient(0,l)+
                tetInfo->m_deformationGradient(1,k)*tetInfo->m_deformationGradient(1,l)+
                tetInfo->m_deformationGradient(2,k)*tetInfo->m_deformationGradient(2,l));
            }
        }

        if(globalParameters.anisotropyDirection.size()>0)
        {
            tetInfo->m_fiberDirection=globalParameters.anisotropyDirection[0];
            Coord vectCa=tetInfo->deformationTensor*tetInfo->m_fiberDirection;
            Real aDotCDota=dot(tetInfo->m_fiberDirection,vectCa);
            tetInfo->lambda=(Real)sqrt(aDotCDota);
        }
        Coord areaVec = cross( dp[1], dp[2] );

        tetInfo->J = dot( areaVec, dp[0] ) * tetInfo->m_volScale;
        tetInfo->trC = (Real)( tetInfo->deformationTensor(0,0) + tetInfo->deformationTensor(1,1) + tetInfo->deformationTensor(2,2));
        tetInfo->m_SPKTensorGeneral.clear();
        m_myMaterial->deriveSPKTensor(tetInfo,globalParameters,tetInfo->m_SPKTensorGeneral);
        for(l=0;l<4;++l)
        {
            f[ta[l]]-=tetInfo->m_deformationGradient*(tetInfo->m_SPKTensorGeneral*tetInfo->m_shapeVector[l])*tetInfo->m_restVolume;
        }
    }


    /// indicates that the next call to addDForce will need to update the stiffness matrix
    m_updateMatrix=true;
    m_tetrahedronInfo.endEdit();

    d_f.endEdit();
}

template <class DataTypes>
void TetrahedronHyperelasticityFEMForceField<DataTypes>::updateTangentMatrix()
{
    unsigned int i=0,j=0,k=0,l=0;
    unsigned int nbEdges=m_topology->getNbEdges();
    const vector< Edge> &edgeArray=m_topology->getEdges() ;

    helper::vector<EdgeInformation>& edgeInf = *(m_edgeInfo.beginEdit());
    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(m_tetrahedronInfo.beginEdit());

    EdgeInformation *einfo;
    TetrahedronRestInformation *tetInfo;
    unsigned int nbTetrahedra=m_topology->getNbTetrahedra();
    const std::vector< Tetrahedron> &tetrahedronArray=m_topology->getTetrahedra() ;

    for(l=0; l<nbEdges; l++ )edgeInf[l].DfDx.clear();
    for(i=0; i<nbTetrahedra; i++ )
    {
        tetInfo=&tetrahedronInf[i];
        Matrix3 &df=tetInfo->m_deformationGradient;
//			Matrix3 Tdf=df.transposed();
        BaseMeshTopology::EdgesInTetrahedron te=m_topology->getEdgesInTetrahedron(i);

        /// describe the jth vertex index of triangle no i
        const Tetrahedron &ta= tetrahedronArray[i];
        for(j=0;j<6;j++) {
            einfo= &edgeInf[te[j]];
            Edge e=m_topology->getLocalEdgesInTetrahedron(j);

            k=e[0];
            l=e[1];
            if (edgeArray[te[j]][0]!=ta[k]) {
                k=e[1];
                l=e[0];
            }
            Matrix3 &edgeDfDx = einfo->DfDx;


            Coord svl=tetInfo->m_shapeVector[l];
            Coord svk=tetInfo->m_shapeVector[k];

            Matrix3  M, N;
            MatrixSym outputTensor;
            N.clear();
            vector<MatrixSym> inputTensor;
            inputTensor.resize(3);
            //	MatrixSym input1,input2,input3,outputTensor;
            for(int m=0; m<3;m++){
                for (int n=m;n<3;n++){
                    inputTensor[0](m,n)=svl[m]*df[0][n]+df[0][m]*svl[n];
                    inputTensor[1](m,n)=svl[m]*df[1][n]+df[1][m]*svl[n];
                    inputTensor[2](m,n)=svl[m]*df[2][n]+df[2][m]*svl[n];
                }
            }

            for(int m=0; m<3; m++){

                m_myMaterial->applyElasticityTensor(tetInfo,globalParameters,inputTensor[m],outputTensor);
                Coord vectortemp=df*(outputTensor*svk);
                Matrix3 Nv;
                //Nv.clear();
                for(int u=0; u<3;u++){
                    Nv[u][m]=vectortemp[u];
                }
                N+=Nv.transposed();
            }


            //Now M
            Real productSD=0;

            Coord vectSD=tetInfo->m_SPKTensorGeneral*svk;
            productSD=dot(vectSD,svl);
            M[0][1]=M[0][2]=M[1][0]=M[1][2]=M[2][0]=M[2][1]=0;
            M[0][0]=M[1][1]=M[2][2]=(Real)productSD;

            edgeDfDx += (M+N)*tetInfo->m_restVolume;


        }// end of for j
    }//end of for i
    m_updateMatrix=false;
}


template <class DataTypes> 
void TetrahedronHyperelasticityFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int l=0;
    unsigned int nbEdges=m_topology->getNbEdges();
    const vector< Edge> &edgeArray=m_topology->getEdges() ;

    helper::vector<EdgeInformation>& edgeInf = *(m_edgeInfo.beginEdit());

    EdgeInformation *einfo;


    /// if the  matrix needs to be updated
    if (m_updateMatrix) {
    this->updateTangentMatrix();
    }// end of if


    /// performs matrix vector computation
    unsigned int v0,v1;
    Deriv deltax;	Deriv dv0,dv1;

    for(l=0; l<nbEdges; l++ )
    {
        einfo=&edgeInf[l];
        v0=edgeArray[l][0];
        v1=edgeArray[l][1];

        deltax= dx[v0] - dx[v1];
        dv0 = einfo->DfDx * deltax;
        // do the transpose multiply:
        dv1[0] = (Real)(deltax[0]*einfo->DfDx[0][0] + deltax[1]*einfo->DfDx[1][0] + deltax[2]*einfo->DfDx[2][0]);
        dv1[1] = (Real)(deltax[0]*einfo->DfDx[0][1] + deltax[1]*einfo->DfDx[1][1] + deltax[2]*einfo->DfDx[2][1]);
        dv1[2] = (Real)(deltax[0]*einfo->DfDx[0][2] + deltax[1]*einfo->DfDx[1][2] + deltax[2]*einfo->DfDx[2][2]);
        // add forces
        df[v0] += dv1 * kFactor;
        df[v1] -= dv0 * kFactor;
    }
    m_edgeInfo.endEdit();
    m_tetrahedronInfo.endEdit();

    d_df.endEdit();
}

template<class DataTypes>
SReal TetrahedronHyperelasticityFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams*, const DataVecCoord&) const
{
    msg_error() << "ERROR("<<this->getClassName()<<"): getPotentialEnergy( const MechanicalParams*, const DataVecCoord& ) not implemented.";
    return 0.0;
}

template <class DataTypes>
void TetrahedronHyperelasticityFEMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal k, unsigned int &offset)
{

    /// if the  matrix needs to be updated
    if (m_updateMatrix)
    {
        this->updateTangentMatrix();
    }

    unsigned int nbEdges=m_topology->getNbEdges();
    const vector< Edge> &edgeArray=m_topology->getEdges() ;
    helper::vector<EdgeInformation>& edgeInf = *(m_edgeInfo.beginEdit());
    EdgeInformation *einfo;
    unsigned int i,j,N0, N1, l;
        Index noeud0, noeud1;

    for(l=0; l<nbEdges; l++ )
    {
        einfo=&edgeInf[l];
        noeud0=edgeArray[l][0];
        noeud1=edgeArray[l][1];
        N0 = offset+3*noeud0;
        N1 = offset+3*noeud1;

        for (i=0; i<3; i++)
        {
            for(j=0; j<3; j++)
            {
                mat->add(N0+i, N0+j,  einfo->DfDx[j][i]*k);
                mat->add(N0+i, N1+j, - einfo->DfDx[j][i]*k);
                mat->add(N1+i, N0+j, - einfo->DfDx[i][j]*k);
                mat->add(N1+i, N1+j, + einfo->DfDx[i][j]*k);
            }
        }
    }
    m_edgeInfo.endEdit();
}


template<class DataTypes>
Mat<3,3,double> TetrahedronHyperelasticityFEMForceField<DataTypes>::getPhi(int TetrahedronIndex)
{
    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(m_tetrahedronInfo.beginEdit());
	TetrahedronRestInformation *tetInfo;
	tetInfo=&tetrahedronInf[TetrahedronIndex];
    return tetInfo->m_deformationGradient;

}

template<class DataTypes>
void TetrahedronHyperelasticityFEMForceField<DataTypes>::testDerivatives()
{
    DataVecCoord d_pos;
    VecCoord &pos = *d_pos.beginEdit();
    pos =  this->mstate->read(core::ConstVecCoordId::position())->getValue();

    // perturbate original state:
    srand( 0 );
    for (unsigned int idx=0; idx<pos.size(); idx++) {
            for (unsigned int d=0; d<3; d++) pos[idx][d] += (Real)0.01 * ((Real)rand()/(Real)(RAND_MAX - 0.5));
    }

    DataVecDeriv d_force1;
    VecDeriv &force1 = *d_force1.beginEdit();
    force1.resize( pos.size() );

    DataVecDeriv d_deltaPos;
    VecDeriv &deltaPos = *d_deltaPos.beginEdit();
    deltaPos.resize( pos.size() );

    DataVecDeriv d_deltaForceCalculated;
    VecDeriv &deltaForceCalculated = *d_deltaForceCalculated.beginEdit();
    deltaForceCalculated.resize( pos.size() );

    DataVecDeriv d_force2;
    VecDeriv &force2 = *d_force2.beginEdit();
    force2.resize( pos.size() );

    Coord epsilon, zero;
    Real cs = (Real)0.00001;
    Real errorThresh = (Real)200.0*cs*cs;
    Real errorNorm;
    Real avgError=0.0;
    int count=0;

    helper::vector<TetrahedronRestInformation> &tetrahedronInf = *(m_tetrahedronInfo.beginEdit());

    for (unsigned int moveIdx=0; moveIdx<pos.size(); moveIdx++)
    {
        for (unsigned int i=0; i<pos.size(); i++)
        {
                deltaForceCalculated[i] = zero;
                force1[i] = zero;
                force2[i] = zero;
        }

        d_force1.setValue(force1);
        d_pos.setValue(pos);

        //this->addForce( force1, pos, force1 );
        this->addForce( core::MechanicalParams::defaultInstance() /* PARAMS FIRST */, d_force1, d_pos, d_force1 );

        // get current energy around
        Real energy1 = 0;
        BaseMeshTopology::TetrahedraAroundVertex vTetras = m_topology->getTetrahedraAroundVertex( moveIdx );
        for(unsigned int i = 0; i < vTetras.size(); ++i)
        {
            energy1 += tetrahedronInf[vTetras[i]].m_strainEnergy * tetrahedronInf[vTetras[i]].m_restVolume;
        }
        // generate random delta
        epsilon[0]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        epsilon[1]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        epsilon[2]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        deltaPos[moveIdx] = epsilon;
        // calc derivative
        this->addDForce( core::MechanicalParams::defaultInstance() /* PARAMS FIRST */, d_deltaForceCalculated, d_deltaPos );
        deltaPos[moveIdx] = zero;
        // calc factual change
        pos[moveIdx] = pos[moveIdx] + epsilon;

        DataVecCoord d_force2;
        d_force2.setValue(force2);
        //this->addForce( force2, pos, force2 );
        this->addForce( core::MechanicalParams::defaultInstance() /* PARAMS FIRST */, d_force2, d_pos, d_force2 );

        pos[moveIdx] = pos[moveIdx] - epsilon;
        // check first derivative:
        Real energy2 = 0;
        for(unsigned int i = 0; i < vTetras.size(); ++i)
        {
                energy2 += tetrahedronInf[vTetras[i]].m_strainEnergy * tetrahedronInf[vTetras[i]].m_restVolume;
        }
        Coord forceAtMI = force1[moveIdx];
        Real deltaEnergyPredicted = -dot( forceAtMI, epsilon );
        Real deltaEnergyFactual = (energy2 - energy1);
        Real energyError = fabs( deltaEnergyPredicted - deltaEnergyFactual );
        if (energyError > 0.05*fabs(deltaEnergyFactual))
        { // allow up to 5% error
            printf("Error energy %i = %f%%\n", moveIdx, 100.0*energyError/fabs(deltaEnergyFactual) );
        }

        // check 2nd derivative for off-diagonal elements:
        BaseMeshTopology::EdgesAroundVertex vEdges = m_topology->getEdgesAroundVertex( moveIdx );
        for (unsigned int eIdx=0; eIdx<vEdges.size(); eIdx++)
        {
            BaseMeshTopology::Edge edge = m_topology->getEdge( vEdges[eIdx] );
            unsigned int testIdx = edge[0];
            if (testIdx==moveIdx) testIdx = edge[1];
            Coord deltaForceFactual = force2[testIdx] - force1[testIdx];
            Coord deltaForcePredicted = deltaForceCalculated[testIdx];
            Coord error = deltaForcePredicted - deltaForceFactual;
            errorNorm = error.norm();
            errorThresh = (Real) 0.05 * deltaForceFactual.norm(); // allow up to 5% error

            if (deltaForceFactual.norm() > 0.0)
            {
                    avgError += (Real)100.0*errorNorm/deltaForceFactual.norm();
                    count++;
            }
            if (errorNorm > errorThresh)
            {
                    printf("Error move %i test %i = %f%%\n", moveIdx, testIdx, 100.0*errorNorm/deltaForceFactual.norm() );
            }
        }
        // check 2nd derivative for diagonal elements:
        unsigned int testIdx = moveIdx;
        Coord deltaForceFactual = force2[testIdx] - force1[testIdx];
        Coord deltaForcePredicted = deltaForceCalculated[testIdx];
        Coord error = deltaForcePredicted - deltaForceFactual;
        errorNorm = error.norm();
        errorThresh = (Real)0.05 * deltaForceFactual.norm(); // allow up to 5% error
        if (errorNorm > errorThresh)
        {
                printf("Error move %i test %i = %f%%\n", moveIdx, testIdx, 100.0*errorNorm/deltaForceFactual.norm() );
        }
    }

    m_tetrahedronInfo.endEdit();
    printf( "testDerivatives passed!\n" );
    avgError /= (Real)count;
    printf( "Average error = %.2f%%\n", avgError );

    d_pos.endEdit();
    d_force1.endEdit();
    d_force2.endEdit();
    d_deltaPos.endEdit();
    d_deltaForceCalculated.endEdit();
}


template<class DataTypes>
void TetrahedronHyperelasticityFEMForceField<DataTypes>::saveMesh( const char *filename )
{
    VecCoord pos( this->mstate->read(core::ConstVecCoordId::position())->getValue());
    core::topology::BaseMeshTopology::SeqTriangles triangles = m_topology->getTriangles();
    FILE *file = fopen( filename, "wb" );

    if (!file) return;

    // write header
    char header[81];

    size_t errResult;
    errResult = fwrite( (void*)&(header[0]),1, 80, file );
    unsigned int numTriangles = triangles.size();
    errResult = fwrite( &numTriangles, 4, 1, file );
    // write poly data
    float vertex[3][3];
    float normal[3] = { 1,0,0 };
    short stlSeperator = 0;

    for (unsigned int triangleId=0; triangleId<triangles.size(); triangleId++)
    {
        if (m_topology->getTetrahedraAroundTriangle( triangleId ).size()==1)
        {
            // surface triangle, save it
            unsigned int p0 = m_topology->getTriangle( triangleId )[0];
            unsigned int p1 = m_topology->getTriangle( triangleId )[1];
            unsigned int p2 = m_topology->getTriangle( triangleId )[2];
            for (int d=0; d<3; d++)
            {
                    vertex[0][d] = (float)pos[p0][d];
                    vertex[1][d] = (float)pos[p1][d];
                    vertex[2][d] = (float)pos[p2][d];
            }
            errResult = fwrite( (void*)&(normal[0]), sizeof(float), 3, file );
            errResult = fwrite( (void*)&(vertex[0][0]), sizeof(float), 9, file );
            errResult = fwrite( (void*)&(stlSeperator), 2, 1, file );
        }
    }
    errResult -= errResult; // ugly trick to avoid warnings

	fclose( file );
}

template<class DataTypes>
void TetrahedronHyperelasticityFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    //	unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
          vparams->drawTool()->setPolygonMode(0,true);


    std::vector< Vector3 > points[4];
    for(int i = 0 ; i<m_topology->getNbTetrahedra();++i)
    {
        const Tetrahedron t=m_topology->getTetrahedron(i);

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];
        Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
        Coord pa = (x[a]+center)*(Real)0.666667;
        Coord pb = (x[b]+center)*(Real)0.666667;
        Coord pc = (x[c]+center)*(Real)0.666667;
        Coord pd = (x[d]+center)*(Real)0.666667;

        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);

        points[1].push_back(pb);
        points[1].push_back(pc);
        points[1].push_back(pd);

        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(pa);

        points[3].push_back(pd);
        points[3].push_back(pa);
        points[3].push_back(pb);
    }

    Vec<4,float> color1;
    Vec<4,float> color2;
    Vec<4,float> color3;
    Vec<4,float> color4;

    std::string material = d_materialName.getValue();
    if (material=="ArrudaBoyce") {
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.5,1.0,0.0,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,1.0,0.5,1.0);
    }
    else if (material=="StVenantKirchhoff"){
        color1 = Vec<4,float>(1.0,0.0,0.0,1.0);
        color2 = Vec<4,float>(1.0,0.0,0.5,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,0.5,1.0,1.0);
    }
    else if (material=="NeoHookean"){
        color1 = Vec<4,float>(0.0,1.0,1.0,1.0);
        color2 = Vec<4,float>(0.5,0.0,1.0,1.0);
        color3 = Vec<4,float>(1.0,0.0,1.0,1.0);
        color4 = Vec<4,float>(1.0,0.5,1.0,1.0);
    }
    else if (material=="MooneyRivlin"){
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.0,1.0,0.5,1.0);
        color3 = Vec<4,float>(0.0,1.0,1.0,1.0);
        color4 = Vec<4,float>(0.5,1.0,1.0,1.0);
    }
    else if (material=="VerondaWestman"){
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.5,1.0,0.0,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,1.0,0.5,1.0);
    }
    else if (material=="Costa"){
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.5,1.0,0.0,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,1.0,0.5,1.0);
    }
    else if (material=="Ogden"){
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.5,1.0,0.0,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,1.0,0.5,1.0);
    }
    else {
        color1 = Vec<4,float>(0.0,1.0,0.0,1.0);
        color2 = Vec<4,float>(0.5,1.0,0.0,1.0);
        color3 = Vec<4,float>(1.0,1.0,0.0,1.0);
        color4 = Vec<4,float>(1.0,1.0,0.5,1.0);
    }


    vparams->drawTool()->drawTriangles(points[0], color1);
    vparams->drawTool()->drawTriangles(points[1], color2);
    vparams->drawTool()->drawTriangles(points[2], color3);
    vparams->drawTool()->drawTriangles(points[3], color4);

    if (vparams->displayFlags().getShowWireFrame())
          vparams->drawTool()->setPolygonMode(0,false);

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONHYPERELASTICITYFEMFORCEFIELD_INL
