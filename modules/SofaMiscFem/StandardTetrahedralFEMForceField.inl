/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_STANDARDTETRAHEDRALFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_STANDARDTETRAHEDRALFEMFORCEFIELD_INL

#include <SofaMiscFem/BoyceAndArruda.h>
#include <SofaMiscFem/NeoHookean.h>
#include <SofaMiscFem/MooneyRivlin.h>
#include <SofaMiscFem/VerondaWestman.h>
#include <SofaMiscFem/STVenantKirchhoff.h>
#include <SofaMiscFem/HyperelasticMaterial.h>
#include <SofaMiscFem/Costa.h>
#include <SofaMiscFem/Ogden.h>
#include "StandardTetrahedralFEMForceField.h"
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/ObjectFactory.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#ifndef SOFA_NO_OPENGL
#if defined (__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif
#endif
#include <SofaBaseTopology/TopologyData.inl>
#include <algorithm>
#include <iterator>
#include <sofa/helper/AdvancedTimer.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

namespace sofa
{
namespace component
{
namespace forcefield
{

template< class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::GHTetrahedronHandler::applyCreateFunction(unsigned int tetrahedronIndex,
                                                                                            TetrahedronRestInformation & tinfo,
                                                                                            const core::topology::BaseMeshTopology::Tetrahedron &,
                                                                                            const sofa::helper::vector<unsigned int> &,
                                                                                            const sofa::helper::vector<double> &)
{
    if (ff) {
        const helper::vector< core::topology::BaseMeshTopology::Tetrahedron > &tetrahedronArray=ff->_topology->getTetrahedra() ;
        const std::vector< core::topology::BaseMeshTopology::Edge> &edgeArray=ff->_topology->getEdges() ;
        unsigned int j;
        /*int l*/;
        typename DataTypes::Real volume;
        typename DataTypes::Coord point[4];
        const typename DataTypes::VecCoord restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        ///describe the indices of the 4 tetrahedron vertices
        const core::topology::BaseMeshTopology::Tetrahedron &t= tetrahedronArray[tetrahedronIndex];
        core::topology::BaseMeshTopology::EdgesInTetrahedron te=ff->_topology->getEdgesInTetrahedron(tetrahedronIndex);

        //store point indices
        tinfo.tetraIndices[0] = (float)t[0];
        tinfo.tetraIndices[1] = (float)t[1];
        tinfo.tetraIndices[2] = (float)t[2];
        tinfo.tetraIndices[3] = (float)t[3];
        //store edges
        tinfo.tetraEdges[0] = (float)te[0];
        tinfo.tetraEdges[1] = (float)te[1];
        tinfo.tetraEdges[2] = (float)te[2];
        tinfo.tetraEdges[3] = (float)te[3];
        tinfo.tetraEdges[4] = (float)te[4];
        tinfo.tetraEdges[5] = (float)te[5];

        // store the point position
        for(j=0;j<4;++j)
        {
            point[j]=(restPosition)[t[j]];
        }
        /// compute 6 times the rest volume
        volume=dot(cross(point[2]-point[0],point[3]-point[0]),point[1]-point[0]);
        /// store the rest volume
        tinfo.volScale =(Real)(1.0/volume);
        tinfo.restVolume = fabs(volume/6);
        // store shape vectors at the rest configuration
        for(j=0;j<4;++j) {
            if (!(j%2))
                tinfo.shapeVector[j]=-cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/ volume;
            else
                tinfo.shapeVector[j]=cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/ volume;;
        }


        for(j=0;j<6;++j) {
            core::topology::BaseMeshTopology::Edge e=ff->_topology->getLocalEdgesInTetrahedron(j);
            int k=e[0];
            //l=e[1];
            if (edgeArray[te[j]][0]!=t[k]) {
                k=e[1];
                //l=e[0];
            }
        }




    }//end if(ff)

}


template <class DataTypes> StandardTetrahedralFEMForceField<DataTypes>::StandardTetrahedralFEMForceField()
: _topology(0)
    , _initialPoints(0)
    , updateMatrix(true)
    , _meshSaved( false)
    , f_materialName(initData(&f_materialName,std::string("ArrudaBoyce"),"materialName","the name of the material to be used"))
    , f_parameterSet(initData(&f_parameterSet,"ParameterSet","The global parameters specifying the material"))
    , f_anisotropySet(initData(&f_anisotropySet,"AnisotropyDirections","The global directions of anisotropy of the material"))
    , f_parameterFileName(initData(&f_parameterFileName,std::string("myFile.param"),"ParameterFile","the name of the file describing the material parameters for all tetrahedra"))
, tetrahedronInfo(initData(&tetrahedronInfo, "tetrahedronInfo", "Internal tetrahedron data"))
, edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
{
    tetrahedronHandler = new GHTetrahedronHandler(this,&tetrahedronInfo);
}

template <class DataTypes> StandardTetrahedralFEMForceField<DataTypes>::~StandardTetrahedralFEMForceField()
{
    if (tetrahedronHandler) delete tetrahedronHandler;
}

template <class DataTypes> void StandardTetrahedralFEMForceField<DataTypes>::init()
{
    this->Inherited::init();
    _topology = this->getContext()->getMeshTopology();

        tetrahedronInfo.createTopologicalEngine(_topology,tetrahedronHandler);
        tetrahedronInfo.registerTopologicalData();

        edgeInfo.createTopologicalEngine(_topology);
        edgeInfo.registerTopologicalData();

    /** parse the parameter set */
    SetParameterArray paramSet=f_parameterSet.getValue();
    if (paramSet.size()>0) {
        globalParameters.parameterArray.resize(paramSet.size());
        copy(paramSet.begin(), paramSet.end(),globalParameters.parameterArray.begin());
    }
    /** parse the anisotropy Direction set */
    SetAnisotropyDirectionArray anisotropySet=f_anisotropySet.getValue();
    if (anisotropySet.size()>0) {
        globalParameters.anisotropyDirection.resize(anisotropySet.size());
        copy(anisotropySet.begin(), anisotropySet.end(),globalParameters.anisotropyDirection.begin());
    }
    //vector<HyperelasticMaterialTerm *> materialTermArray;
    /** parse the input material name */
    std::string material = f_materialName.getValue();
    if(material=="ArrudaBoyce") {
        fem::BoyceAndArruda<DataTypes> *BoyceAndArrudaMaterial = new fem::BoyceAndArruda<DataTypes>;
        myMaterial = BoyceAndArrudaMaterial;
        msg_info() << "The model is "<<material ;
    }
    else if (material=="StVenantKirchhoff"){
        fem::STVenantKirchhoff<DataTypes> *STVenantKirchhoffMaterial = new fem::STVenantKirchhoff<DataTypes>;
        myMaterial = STVenantKirchhoffMaterial;
        msg_info() << "The model is "<<material ;
    }
    else if (material=="NeoHookean"){
        fem::NeoHookean<DataTypes> *NeoHookeanMaterial = new fem::NeoHookean<DataTypes>;
        myMaterial = NeoHookeanMaterial;
        msg_info() <<"The model is "<<material ;
    }
    else if (material=="MooneyRivlin"){
        fem::MooneyRivlin<DataTypes> *MooneyRivlinMaterial = new fem::MooneyRivlin<DataTypes>;
        myMaterial = MooneyRivlinMaterial;
        msg_info() << "The model is "<<material ;
    }
    else if (material=="VerondaWestman"){
        fem::VerondaWestman<DataTypes> *VerondaWestmanMaterial = new fem::VerondaWestman<DataTypes>;
        myMaterial = VerondaWestmanMaterial;
        msg_info() << "The model is "<<material ;
    }
    else if (material=="Costa"){
        fem::Costa<DataTypes> *CostaMaterial = new fem::Costa<DataTypes>();
        myMaterial =CostaMaterial;
        msg_info() <<"The model is "<<material ;
    }
    else if (material=="Ogden"){
        fem::Ogden<DataTypes> *OgdenMaterial = new fem::Ogden<DataTypes>();
        myMaterial =OgdenMaterial;
        msg_info()<<"The model is "<<material ;
    }
    else {
        msg_error() << "Material name '" << material << "' is not valid." ;
    }

    if (!_topology->getNbTetrahedra())
    {
        msg_error() << "Object must have a Tetrahedral Set Topology.";
        return;
    }

    tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    /// prepare to store info in the triangle array
    tetrahedronInf.resize(_topology->getNbTetrahedra());

    edgeInformationVector& edgeInf = *(edgeInfo.beginEdit());

    edgeInf.resize(_topology->getNbEdges());
    edgeInfo.endEdit();

    if (_initialPoints.size() == 0)
    {
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints=p;
    }
    int i;

    /// initialize the data structure associate with each tetrahedron
    for (int i=0; i<_topology->getNbEdges(); i++)
    {
        edgeInf[i].vertices[0] = (float) _topology->getEdge(i)[0];
        edgeInf[i].vertices[1] = (float) _topology->getEdge(i)[1];
    }

    /// initialize the data structure associated with each tetrahedron
    for (i=0;i<_topology->getNbTetrahedra();++i) {
            tetrahedronHandler->applyCreateFunction(i, tetrahedronInf[i],
                        _topology->getTetrahedron(i),  (const helper::vector< unsigned int > )0,
                        (const helper::vector< double >)0);
    }
    /// set the call back function upon creation of a tetrahedron

    tetrahedronInfo.endEdit();

    /// FOR CUDA
    /// Save the neighbourhood for points (in case of CudaTypes)
    this->initNeighbourhoodPoints();
    this->initNeighbourhoodEdges();
}

template <class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::initNeighbourhoodPoints(){}

template <class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::initNeighbourhoodEdges(){}

template <class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::addForce(const core::MechanicalParams*  mparams , DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    sofa::helper::AdvancedTimer::stepBegin("addForceStandardTetraFEM");

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();

    unsigned int i=0,j=0,k=0,l=0;
    unsigned int nbTetrahedra=_topology->getNbTetrahedra();

    tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
    unsigned int nbEdges=_topology->getNbEdges();
    const helper::vector< core::topology::BaseMeshTopology::Edge> &edgeArray=_topology->getEdges() ;
    TetrahedronRestInformation *tetInfo;
    EdgeInformation *einfo;


    myposition=x;

    Coord dp[3],x0,sv;


    if (mparams->implicit()) {
        // if implicit solver recompute the stiffness matrix stored at each edge
        // starts by reseting each matrix to 0
        for(l=0; l<nbEdges; l++ )edgeInf[l].DfDx.clear();
    }
    Matrix3 deformationGradient;
    Matrix63 matB[4];
    MatrixSym SPK;
    for(i=0; i<nbTetrahedra; i++ )
    {
        tetInfo=&tetrahedronInf[i];
        const core::topology::BaseMeshTopology::Tetrahedron &ta= _topology->getTetrahedron(i);

        x0=x[ta[0]];

        // compute the deformation gradient
        // deformation gradient = sum of tensor product between vertex position and shape vector
        // optimize by using displacement with first vertex
        dp[0]=x[ta[1]]-x0;
        sv=tetInfo->shapeVector[1];
        for (k=0;k<3;++k) {
            for (l=0;l<3;++l) {
                deformationGradient[k][l]=dp[0][k]*sv[l];
            }
        }
        for (j=1;j<3;++j) {
            dp[j]=x[ta[j+1]]-x0;
            sv=tetInfo->shapeVector[j+1];
            for (k=0;k<3;++k) {
                for (l=0;l<3;++l) {
                    deformationGradient[k][l]+=dp[j][k]*sv[l];
                }
            }
        }

        //Compute the matrix strain displacement B 6*3
        for (int alpha=0; alpha<4; ++alpha){
            Coord sva=tetInfo->shapeVector[alpha];
            Matrix63 matBa;
            matBa[0][0]=deformationGradient[0][0]*sva[0];
            matBa[0][1]=deformationGradient[1][0]*sva[0];
            matBa[0][2]=deformationGradient[2][0]*sva[0];

            matBa[2][0]=deformationGradient[0][1]*sva[1];
            matBa[2][1]=deformationGradient[1][1]*sva[1];
            matBa[2][2]=deformationGradient[2][1]*sva[1];

            matBa[5][0]=deformationGradient[0][2]*sva[2];
            matBa[5][1]=deformationGradient[1][2]*sva[2];
            matBa[5][2]=deformationGradient[2][2]*sva[2];

            matBa[1][0]=(deformationGradient[0][0]*sva[1]+deformationGradient[0][1]*sva[0]);
            matBa[1][1]=(deformationGradient[1][0]*sva[1]+deformationGradient[1][1]*sva[0]);
            matBa[1][2]=(deformationGradient[2][0]*sva[1]+deformationGradient[2][1]*sva[0]);

            matBa[3][0]=(deformationGradient[0][2]*sva[0]+deformationGradient[0][0]*sva[2]);
            matBa[3][1]=(deformationGradient[1][2]*sva[0]+deformationGradient[1][0]*sva[2]);
            matBa[3][2]=(deformationGradient[2][2]*sva[0]+deformationGradient[2][0]*sva[2]);

            matBa[4][0]=(deformationGradient[0][1]*sva[2]+deformationGradient[0][2]*sva[1]);
            matBa[4][1]=(deformationGradient[1][1]*sva[2]+deformationGradient[1][2]*sva[1]);
            matBa[4][2]=(deformationGradient[2][1]*sva[2]+deformationGradient[2][2]*sva[1]);

            matB[alpha]=matBa;
        }


        /// compute the right Cauchy-Green deformation matrix
        for (k=0;k<3;++k) {
            for (l=k;l<3;++l) {
                //	if (l>=k) {
                tetInfo->deformationTensor(k,l)=(deformationGradient(0,k)*deformationGradient(0,l)+
                    deformationGradient(1,k)*deformationGradient(1,l)+
                    deformationGradient(2,k)*deformationGradient(2,l));
                //	}
                //	else
                //		tetInfo->deformationTensor[k][l]=tetInfo->deformationTensor[l][k];
            }
        }


        //in case of transversaly isotropy

        if(globalParameters.anisotropyDirection.size()>0){
            tetInfo->fiberDirection=globalParameters.anisotropyDirection[0];
            Coord vectCa=tetInfo->deformationTensor*tetInfo->fiberDirection;
            Real aDotCDota=dot(tetInfo->fiberDirection,vectCa);
            tetInfo->lambda=(Real)sqrt(aDotCDota);
        }
        Coord areaVec = cross( dp[1], dp[2] );

        tetInfo->J = dot( areaVec, dp[0] ) * tetInfo->volScale;
        tetInfo->trC = (Real)( tetInfo->deformationTensor(0,0) + tetInfo->deformationTensor(1,1) + tetInfo->deformationTensor(2,2));
        //tetInfo->strainEnergy=myMaterial->getStrainEnergy(tetInfo,globalParameters); // to uncomment for test derivatives
    //	tetInfo->SPKTensorGeneral.clear();
        SPK.clear();
        myMaterial->deriveSPKTensor(tetInfo,globalParameters,SPK); // calculate the SPK tensor of the chosen material
        //tetInfo->SPKTensorGeneral=SPK;
        for(l=0;l<4;++l){
            f[ta[l]]-=matB[l].transposed()*SPK*tetInfo->restVolume;
        }
        if (mparams->implicit()) {
            // if implicit solver then computes the stifffness on each edge
             core::topology::BaseMeshTopology::EdgesInTetrahedron te=_topology->getEdgesInTetrahedron(i);

            /// describe the jth edge index of tetrahedron i no i
            for(j=0;j<6;j++) {
                einfo= &edgeInf[te[j]];
                core::topology::BaseMeshTopology::Edge e=_topology->getLocalEdgesInTetrahedron(j);

                k=e[0];
                l=e[1];
                if (edgeArray[te[j]][0]!=ta[k]) {
                    k=e[1];
                    l=e[0];
                }
                Matrix3 &edgeDfDx = einfo->DfDx;


                Coord svl=tetInfo->shapeVector[l];
                Coord svk=tetInfo->shapeVector[k];

                Matrix3  M, N;
                Matrix6 outputTensor;
                N.clear();

                // Calculates the dS/dC tensor 6*6
                myMaterial->ElasticityTensor(tetInfo,globalParameters,outputTensor);
                Matrix63 mBl=matB[l];
                mBl[1][0]/=2;mBl[1][1]/=2;mBl[1][2]/=2;mBl[3][0]/=2;mBl[3][1]/=2;mBl[3][2]/=2;mBl[4][0]/=2;mBl[4][1]/=2;mBl[4][2]/=2;

                N=(matB[k].transposed()*outputTensor*mBl);

                //Now M
                Real productSD=0;

                Coord vectSD=SPK*svk;
                productSD=dot(vectSD,svl);
                M[0][1]=M[0][2]=M[1][0]=M[1][2]=M[2][0]=M[2][1]=0;
                M[0][0]=M[1][1]=M[2][2]=(Real)productSD;

                edgeDfDx += (M+N.transposed())*tetInfo->restVolume;


            }// end of for j
        }
    }


    /// indicates that the next call to addDForce will need to update the stiffness matrix
    updateMatrix=true;
    tetrahedronInfo.endEdit();
    edgeInfo.endEdit();
    d_f.endEdit();

    sofa::helper::AdvancedTimer::stepEnd("addForceStandardTetraFEM");
}


template <class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    sofa::helper::AdvancedTimer::stepBegin("addDForceStandardTetraFEM");

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int l=0;
    unsigned int nbEdges=_topology->getNbEdges();
    const helper::vector< core::topology::BaseMeshTopology::Edge> &edgeArray=_topology->getEdges() ;

    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
//	tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    EdgeInformation *einfo;

    /*
    /// if the  matrix needs to be updated
    if (updateMatrix) {

        TetrahedronRestInformation *tetInfo;
        unsigned int nbTetrahedra=_topology->getNbTetrahedra();
        const std::vector< topology::Tetrahedron> &tetrahedronArray=_topology->getTetrahedra() ;
        unsigned int i=0, j=0, k=0;
        for(l=0; l<nbEdges; l++ )edgeInf[l].DfDx.clear();
        for(i=0; i<nbTetrahedra; i++ )
        {
            tetInfo=&tetrahedronInf[i];
//			Matrix3 &df=tetInfo->deformationGradient;
//			Matrix3 Tdf=df.transposed();
            core::topology::BaseMeshTopology::EdgesInTetrahedron te=_topology->getEdgesInTetrahedron(i);

            /// describe the jth vertex index of triangle no i
            const topology::Tetrahedron &ta= tetrahedronArray[i];
            for(j=0;j<6;j++) {
                einfo= &edgeInf[te[j]];
                topology::Edge e=_topology->getLocalEdgesInTetrahedron(j);

                k=e[0];
                l=e[1];
                if (edgeArray[te[j]][0]!=ta[k]) {
                    k=e[1];
                    l=e[0];
                }
                Matrix3 &edgeDfDx = einfo->DfDx;


                Coord svl=tetInfo->shapeVector[l];
                Coord svk=tetInfo->shapeVector[k];

                Matrix3  M, N;
                Matrix6 outputTensor;
                N.clear();

                // Calculates the dS/dC tensor 6*6
                myMaterial->ElasticityTensor(tetInfo,globalParameters,outputTensor);
                Matrix63 mBl=tetInfo->matB[l];
                mBl[1][0]/=2;mBl[1][1]/=2;mBl[1][2]/=2;mBl[3][0]/=2;mBl[3][1]/=2;mBl[3][2]/=2;mBl[4][0]/=2;mBl[4][1]/=2;mBl[4][2]/=2;

                N=(tetInfo->matB[k].transposed()*outputTensor*mBl);




                //Now M
                Real productSD=0;

                Coord vectSD=tetInfo->SPKTensorGeneral*svk;
                productSD=dot(vectSD,svl);
                M[0][1]=M[0][2]=M[1][0]=M[1][2]=M[2][0]=M[2][1]=0;
                M[0][0]=M[1][1]=M[2][2]=(Real)productSD;

                edgeDfDx += (M+N.transposed())*tetInfo->restVolume;


            }// end of for j
        }//end of for i
        updateMatrix=false;
    }// end of if

    */
    /// performs matrix vector computation
    Deriv deltax;	Deriv dv0,dv1;

    for(l=0; l<nbEdges; l++ )
    {
        einfo=&edgeInf[l];
        unsigned int v0=edgeArray[l][0];
        unsigned int v1=edgeArray[l][1];

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
    edgeInfo.endEdit();
//	tetrahedronInfo.endEdit();
    d_df.beginEdit();

    sofa::helper::AdvancedTimer::stepEnd("addDForceStandardTetraFEM");
}

template<class DataTypes>
void  StandardTetrahedralFEMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * mat, SReal kFact, unsigned int &offset)
{
    unsigned int nbEdges=_topology->getNbEdges();
    const helper::vector< Edge> &edgeArray=_topology->getEdges() ;
    edgeInformationVector& edgeInf = *(edgeInfo.beginEdit());
    EdgeInformation *einfo;
    unsigned int i,j,N0, N1, l;
    Index noeud0, noeud1;
    if (sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,Real> > * crsmat = dynamic_cast<sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,Real> > * >(mat))
    {
        int offd3 = offset/3;

        for(l=0; l<nbEdges; l++ )
        {
            einfo=&edgeInf[l];
            noeud0=edgeArray[l][0];
            noeud1=edgeArray[l][1];
            N0 = offd3+noeud0;
            N1 = offd3+noeud1;
            Matrix3 stiff= einfo->DfDx*(Real)kFact;
            Matrix3 stiffTransposed= einfo->DfDx.transposed()*(Real)kFact;

            *crsmat->wbloc(N0,N0,true) += stiffTransposed;
            *crsmat->wbloc(N1,N1,true) += stiff;
            *crsmat->wbloc(N0,N1,true) -= stiffTransposed;
            *crsmat->wbloc(N1,N0,true) -= stiff;
        }
    } else {
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

                    mat->add(N0+i, N0+j,  einfo->DfDx[j][i]*kFact);
                    mat->add(N0+i, N1+j, - einfo->DfDx[j][i]*kFact);
                    mat->add(N1+i, N0+j, - einfo->DfDx[i][j]*kFact);
                    mat->add(N1+i, N1+j, + einfo->DfDx[i][j]*kFact);

                }
            }

        }
    }
    edgeInfo.endEdit();
}

template<class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
#ifndef SOFA_NO_OPENGL
    //	unsigned int i;
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    if (vparams->displayFlags().getShowWireFrame())
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

#endif /* SOFA_NO_OPENGL */
}

/*
template<class DataTypes>
defaulttype::Mat<3,3,double> StandardTetrahedralFEMForceField<DataTypes>::getPhi(int TetrahedronIndex)
{
    tetrahedronRestInfoVector& tetrahedronInf = *(tetrahedronInfo.beginEdit());
    TetrahedronRestInformation *tetInfo;
    tetInfo=&tetrahedronInf[TetrahedronIndex];
    return tetInfo->deformationGradient;

}
*/
template<class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::testDerivatives()
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

    tetrahedronRestInfoVector &tetrahedronInf = *(tetrahedronInfo.beginEdit());

    for (unsigned int moveIdx=0; moveIdx<pos.size(); moveIdx++)
    {
        for (unsigned int i=0; i<pos.size(); i++) {
            deltaForceCalculated[i] = zero;
            force1[i] = zero;
            force2[i] = zero;
        }

        //this->addForce( force1, pos, force1 );
        this->addForce( core::MechanicalParams::defaultInstance(), d_force1, d_pos, d_force1 );

        // get current energy around
        Real energy1 = 0;
        core::topology::BaseMeshTopology::TetrahedraAroundVertex vTetras = _topology->getTetrahedraAroundVertex( moveIdx );
        for(unsigned int i = 0; i < vTetras.size(); ++i)
        {
            energy1 += tetrahedronInf[vTetras[i]].strainEnergy * tetrahedronInf[vTetras[i]].restVolume;
        }
        // generate random delta
        epsilon[0]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        epsilon[1]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        epsilon[2]= cs * ((Real)rand()/(Real)(RAND_MAX - 0.5));
        deltaPos[moveIdx] = epsilon;
        // calc derivative
        //this->addDForce( deltaForceCalculated, deltaPos);
        this->addDForce( core::MechanicalParams::defaultInstance(), d_deltaForceCalculated, d_deltaPos );

        deltaPos[moveIdx] = zero;
        // calc factual change
        pos[moveIdx] = pos[moveIdx] + epsilon;
        //this->addForce( force2, pos, force2 );
        this->addForce( core::MechanicalParams::defaultInstance(), d_force2, d_pos, d_force2 );

        pos[moveIdx] = pos[moveIdx] - epsilon;
        // check first derivative:
        Real energy2 = 0;
        for(unsigned int i = 0; i < vTetras.size(); ++i)
        {
            energy2 += tetrahedronInf[vTetras[i]].strainEnergy * tetrahedronInf[vTetras[i]].restVolume;
        }
        Coord forceAtMI = force1[moveIdx];
        Real deltaEnergyPredicted = -dot( forceAtMI, epsilon );
        Real deltaEnergyFactual = (energy2 - energy1);
        Real energyError = fabs( deltaEnergyPredicted - deltaEnergyFactual );
        if (energyError > 0.05*fabs(deltaEnergyFactual)) { // allow up to 5% error
            printf("Error energy %ui = %f%%\n", moveIdx, 100.0*energyError/fabs(deltaEnergyFactual) );
        }

        // check 2nd derivative for off-diagonal elements:
        core::topology::BaseMeshTopology::EdgesAroundVertex vEdges = _topology->getEdgesAroundVertex( moveIdx );
        for (unsigned int eIdx=0; eIdx<vEdges.size(); eIdx++)
        {
            core::topology::BaseMeshTopology::Edge edge = _topology->getEdge( vEdges[eIdx] );
            unsigned int testIdx = edge[0];
            if (testIdx==moveIdx) testIdx = edge[1];
            Coord deltaForceFactual = force2[testIdx] - force1[testIdx];
            Coord deltaForcePredicted = deltaForceCalculated[testIdx];
            Coord error = deltaForcePredicted - deltaForceFactual;
            errorNorm = error.norm();
            errorThresh = (Real) 0.05 * deltaForceFactual.norm(); // allow up to 5% error
            if (deltaForceFactual.norm() > 0.0) {
                avgError += (Real)100.0*errorNorm/deltaForceFactual.norm();
                count++;
            }
            if (errorNorm > errorThresh) {
                printf("Error move %ui test %ui = %f%%\n", moveIdx, testIdx, 100.0*errorNorm/deltaForceFactual.norm() );
            }
        }
        // check 2nd derivative for diagonal elements:
        unsigned int testIdx = moveIdx;
        Coord deltaForceFactual = force2[testIdx] - force1[testIdx];
        Coord deltaForcePredicted = deltaForceCalculated[testIdx];
        Coord error = deltaForcePredicted - deltaForceFactual;
        errorNorm = error.norm();
        errorThresh = (Real)0.05 * deltaForceFactual.norm(); // allow up to 5% error
        if (errorNorm > errorThresh) {
            printf("Error move %ui test %ui = %f%%\n", moveIdx, testIdx, 100.0*errorNorm/deltaForceFactual.norm() );
        }
    }

    tetrahedronInfo.endEdit();
    printf( "testDerivatives passed!\n" );
    avgError /= (Real)count;
    printf( "Average error = %.2f%%\n", avgError );
}

template<class DataTypes>
void StandardTetrahedralFEMForceField<DataTypes>::saveMesh( const char *filename )
{
    VecCoord pos( this->mstate->read(core::ConstVecCoordId::position())->getValue() );
    core::topology::BaseMeshTopology::SeqTriangles triangles = _topology->getTriangles();
    FILE *file = fopen( filename, "wb" );
    if (!file) return;
    // write header
    char header[81];
    //	strcpy( header, "STL generated by SOFA." );
    fwrite( (void*)&(header[0]),1, 80, file );
    unsigned int numTriangles = triangles.size();
    fwrite( &numTriangles, 4, 1, file );
    // write poly data
    float vertex[3][3];
    float normal[3] = { 1,0,0 };
    short stlSeperator = 0;

    for (unsigned int triangleId=0; triangleId<triangles.size(); triangleId++) {
        if (_topology->getTetrahedraAroundTriangle( triangleId ).size()==1) {
            // surface triangle, save it
            unsigned int p0 = _topology->getTriangle( triangleId )[0];
            unsigned int p1 = _topology->getTriangle( triangleId )[1];
            unsigned int p2 = _topology->getTriangle( triangleId )[2];
            for (int d=0; d<3; d++) {
                vertex[0][d] = (float)pos[p0][d];
                vertex[1][d] = (float)pos[p1][d];
                vertex[2][d] = (float)pos[p2][d];
            }

            fwrite( (void*)&(normal[0]), sizeof(float), 3, file );
            fwrite( (void*)&(vertex[0][0]), sizeof(float), 9, file );
            fwrite( (void*)&(stlSeperator), 2, 1, file );
        }
    }

    fclose( file );
}


} // namespace forcefield
} // namespace component
} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_STANDARDTETRAHEDRALFEMFORCEFIELD_INL
