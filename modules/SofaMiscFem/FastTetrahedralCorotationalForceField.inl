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
#ifndef SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_INL

#include <SofaMiscFem/FastTetrahedralCorotationalForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <sofa/helper/gl/template.h>
#include <SofaBaseTopology/TopologyData.inl>
#include <sofa/helper/decompose.h>

namespace sofa
{

namespace component
{

namespace forcefield
{




template< class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::FTCFTetrahedronHandler::applyCreateFunction(unsigned int tetrahedronIndex,
        TetrahedronRestInformation &my_tinfo,
        const core::topology::BaseMeshTopology::Tetrahedron &,
        const sofa::helper::vector<unsigned int> &,
        const sofa::helper::vector<double> &)
{
    if (ff)
    {
        const std::vector< Tetrahedron > &tetrahedronArray=ff->_topology->getTetrahedra() ;
        //		const std::vector< Edge> &edgeArray=ff->_topology->getEdges() ;
        unsigned int j,k,l,m,n;
        typename DataTypes::Real lambda=ff->getLambda();
        typename DataTypes::Real mu=ff->getMu();
        typename DataTypes::Real volume,val;
        typename DataTypes::Coord point[4]; //shapeVector[4];
        const typename DataTypes::VecCoord restPosition=ff->mstate->read(core::ConstVecCoordId::restPosition())->getValue();

        ///describe the indices of the 4 tetrahedron vertices
        const Tetrahedron &t= tetrahedronArray[tetrahedronIndex];
//    BaseMeshTopology::EdgesInTetrahedron te=ff->_topology->getEdgesInTetrahedron(tetrahedronIndex);


        // store the point position
        for(j=0; j<4; ++j)
            point[j]=(restPosition)[t[j]];
        /// compute 6 times the rest volume
        volume=dot(cross(point[1]-point[0],point[2]-point[0]),point[0]-point[3]);
        /// store the rest volume
        my_tinfo.restVolume=volume/6;
        mu*=fabs(volume)/6;
        lambda*=fabs(volume)/6;

        // store shape vectors at the rest configuration
        for(j=0; j<4; ++j)
        {
            if ((j%2)==0)
                my_tinfo.shapeVector[j]=cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
            else
                my_tinfo.shapeVector[j]= -cross(point[(j+2)%4] - point[(j+1)%4],point[(j+3)%4] - point[(j+1)%4])/volume;
        }

        /// compute the vertex stiffness of the linear elastic material, needed for addKToMatrix
        for(j=0; j<4; ++j)
        {
            // the linear stiffness matrix using shape vectors and Lame coefficients
            val=mu*dot(my_tinfo.shapeVector[j],my_tinfo.shapeVector[j]);
            for(m=0; m<3; ++m)
            {
                for(n=m; n<3; ++n)
                {
                    my_tinfo.linearDfDxDiag[j][m][n]=lambda*my_tinfo.shapeVector[j][n]*my_tinfo.shapeVector[j][m]+
                            mu*my_tinfo.shapeVector[j][n]*my_tinfo.shapeVector[j][m];

                    if (m==n)
                    {
                        my_tinfo.linearDfDxDiag[j][m][m]+=(Real)val;
                    } else
                        my_tinfo.linearDfDxDiag[j][n][m]=my_tinfo.linearDfDxDiag[j][m][n];
                }
            }
        }

        /// compute the edge stiffness of the linear elastic material
        for(j=0; j<6; ++j)
        {
            core::topology::BaseMeshTopology::Edge e=ff->_topology->getLocalEdgesInTetrahedron(j);
            k=e[0];
            l=e[1];

            // store the rest edge vector
            my_tinfo.restEdgeVector[j]=point[l]-point[k];

            // the linear stiffness matrix using shape vectors and Lame coefficients
            val=mu*dot(my_tinfo.shapeVector[l],my_tinfo.shapeVector[k]);
            for(m=0; m<3; ++m)
            {
                for(n=0; n<3; ++n)
                {
                    my_tinfo.linearDfDx[j][m][n]=lambda*my_tinfo.shapeVector[k][n]*my_tinfo.shapeVector[l][m]+
                            mu*my_tinfo.shapeVector[l][n]*my_tinfo.shapeVector[k][m];

                    if (m==n)
                    {
                        my_tinfo.linearDfDx[j][m][m]+=(Real)val;
                    }
                }
            }
        }
        if (ff->decompositionMethod==QR_DECOMPOSITION) {
            // compute the rotation matrix of the initial tetrahedron for the QR decomposition
            computeQRRotation(my_tinfo.restRotation,my_tinfo.restEdgeVector);
        } else 	if (ff->decompositionMethod==POLAR_DECOMPOSITION_MODIFIED) {
            Mat3x3 Transformation;
            Transformation[0]=point[1]-point[0];
            Transformation[1]=point[2]-point[0];
            Transformation[2]=point[3]-point[0];
            helper::Decompose<Real>::polarDecomposition( Transformation, my_tinfo.restRotation );
        }
    }
}

template <class DataTypes> FastTetrahedralCorotationalForceField<DataTypes>::FastTetrahedralCorotationalForceField()
    : pointInfo(initData(&pointInfo, "pointInfo", "Internal point data"))
    , edgeInfo(initData(&edgeInfo, "edgeInfo", "Internal edge data"))
    , tetrahedronInfo(initData(&tetrahedronInfo, "tetrahedronInfo", "Internal tetrahedron data"))
    , _initialPoints(0)
    , updateMatrix(true)
    , f_method(initData(&f_method,std::string("qr"),"method"," method for rotation computation :\"qr\" (by QR) or \"polar\" or \"polar2\" or \"none\" (Linear elastic) "))
    , f_poissonRatio(initData(&f_poissonRatio,(Real)0.3,"poissonRatio","Poisson ratio in Hooke's law"))
    , f_youngModulus(initData(&f_youngModulus,(Real)1000.,"youngModulus","Young modulus in Hooke's law"))
    , lambda(0)
    , mu(0)
    , f_drawing(initData(&f_drawing, true, "drawing", " draw the forcefield if true"))
    , drawColor1(initData(&drawColor1, defaulttype::Vec4f(0.0f, 0.0f, 1.0f, 1.0f), "drawColor1", " draw color for faces 1"))
    , drawColor2(initData(&drawColor2, defaulttype::Vec4f(0.0f, 0.5f, 1.0f, 1.0f), "drawColor2", " draw color for faces 2"))
    , drawColor3(initData(&drawColor3, defaulttype::Vec4f(0.0f, 1.0f, 1.0f, 1.0f), "drawColor3", " draw color for faces 3"))
    , drawColor4(initData(&drawColor4, defaulttype::Vec4f(0.5f, 1.0f, 1.0f, 1.0f), "drawColor4", " draw color for faces 4"))
    , tetrahedronHandler(NULL)
{
    tetrahedronHandler = new FTCFTetrahedronHandler(this,&tetrahedronInfo);
}

template <class DataTypes> FastTetrahedralCorotationalForceField<DataTypes>::~FastTetrahedralCorotationalForceField()
{
    if (tetrahedronHandler) delete tetrahedronHandler;
}

template <class DataTypes> void FastTetrahedralCorotationalForceField<DataTypes>::init()
{
    this->Inherited::init();

    _topology = this->getContext()->getMeshTopology();

    if (_topology->getNbTetrahedra()==0)
    {
        serr << "ERROR(FastTetrahedralCorotationalForceField): object must have a Tetrahedral Set Topology."<<sendl;
        return;
    }
    updateLameCoefficients();


    if (f_method.getValue() == "polar")
        decompositionMethod= POLAR_DECOMPOSITION;
     else if ((f_method.getValue() == "qr") || (f_method.getValue() == "large"))
        decompositionMethod= QR_DECOMPOSITION;
    else if (f_method.getValue() == "polar2")
        decompositionMethod= POLAR_DECOMPOSITION_MODIFIED;
     else if ((f_method.getValue() == "none") || (f_method.getValue() == "linear"))
        decompositionMethod= LINEAR_ELASTIC;
    else
    {
        serr << "cannot recognize method "<< f_method.getValue() << ". Must be either qr, polar, polar2 or none" << sendl;
    }


    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());
    tetrahedronInf.resize(_topology->getNbTetrahedra());


    helper::vector<Mat3x3>& edgeInf = *(edgeInfo.beginEdit());
    /// prepare to store info in the edge array
    edgeInf.resize(_topology->getNbEdges());
    edgeInfo.createTopologicalEngine(_topology);
    edgeInfo.registerTopologicalData();
    edgeInfo.endEdit();

    helper::vector<Mat3x3>& pointInf = *(pointInfo.beginEdit());
    /// prepare to store info in the point array
    pointInf.resize(_topology->getNbPoints());
    pointInfo.createTopologicalEngine(_topology);
    pointInfo.registerTopologicalData();
    pointInfo.endEdit();

    if (_initialPoints.size() == 0)
    {
        // get restPosition
        const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
        _initialPoints=p;
    }

    int i;

    /// initialize the data structure associated with each tetrahedron
    for (i=0; i<_topology->getNbTetrahedra(); ++i)
    {
        tetrahedronHandler->applyCreateFunction(i,tetrahedronInf[i],_topology->getTetrahedron(i),
                (const helper::vector< unsigned int > )0,
                (const helper::vector< double >)0);
    }
    /// set the call back function upon creation of a tetrahedron
    tetrahedronInfo.createTopologicalEngine(_topology,tetrahedronHandler);
    tetrahedronInfo.registerTopologicalData();
    tetrahedronInfo.endEdit();

    updateTopologyInfo=true;
}


template <class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::updateTopologyInformation()
{
    int i;
    unsigned int j;

    int nbTetrahedra=_topology->getNbTetrahedra();

    TetrahedronRestInformation *tetinfo;

    helper::vector<typename FastTetrahedralCorotationalForceField<DataTypes>::TetrahedronRestInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());

    for(i=0; i<nbTetrahedra; i++ )
    {
        tetinfo=&tetrahedronInf[i];
        /// describe the jth edge index of triangle no i
        const core::topology::BaseMeshTopology::EdgesInTetrahedron &tea= _topology->getEdgesInTetrahedron(i);
        /// describe the jth vertex index of triangle no i
        const core::topology::BaseMeshTopology::Tetrahedron &ta= _topology->getTetrahedron(i);

        for (j=0; j<6; ++j)
        {
            /// store the information about the orientation of the edge : 1 if the edge orientation matches the orientation in getLocalEdgesInTetrahedron
            /// ie edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};
            if (ta[ _topology->getLocalEdgesInTetrahedron(j)[0]]== _topology->getEdge(tea[j])[0])
                tetinfo->edgeOrientation[j]=1;
            else
                tetinfo->edgeOrientation[j]= -1;
        }

    }    
    tetrahedronInfo.endEdit();

    updateTopologyInfo = false;
}
template<class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::computeQRRotation( Mat3x3 &r, const Coord *dp)
{
    // first vector on first edge
    // second vector in the plane of the two first edges
    // third vector orthogonal to first and second

    Coord edgex = dp[0];
    edgex.normalize();

    Coord edgey = dp[1];

    Coord edgez = cross( edgex, edgey );
    edgez.normalize();

    edgey = cross( edgez, edgex );
    edgey.normalize();

    r[0][0] = edgex[0];
    r[0][1] = edgex[1];
    r[0][2] = edgex[2];
    r[1][0] = edgey[0];
    r[1][1] = edgey[1];
    r[1][2] = edgey[2];
    r[2][0] = edgez[0];
    r[2][1] = edgez[1];
    r[2][2] = edgez[2];
}

template <class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::addForce(const sofa::core::MechanicalParams* /*mparams*/, DataVecDeriv &  dataF, const DataVecCoord &  dataX , const DataVecDeriv & /*dataV*/ )
{

    VecDeriv& f        = *(dataF.beginEdit());
    const VecCoord& x  =   dataX.getValue()  ;


    unsigned int j,k,l;
    int nbTetrahedra=_topology->getNbTetrahedra();
    int i;
    const unsigned int edgesInTetrahedronArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};


    if (updateTopologyInfo)
    {
        updateTopologyInformation();
    }
    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());
    TetrahedronRestInformation *tetinfo;

    Coord dp[6],sv;
    Mat3x3 deformationGradient,S,R;

    for(i=0; i<nbTetrahedra; i++ )
    {
        tetinfo=&tetrahedronInf[i];
        const core::topology::BaseMeshTopology::Tetrahedron &ta = _topology->getTetrahedron(i);
        for (j=0; j<6; ++j)
        {
            dp[j]=x[ta[edgesInTetrahedronArray[j][1]]]-x[ta[edgesInTetrahedronArray[j][0]]];
        }

        if (decompositionMethod==POLAR_DECOMPOSITION)
        {
            // compute the deformation gradient
            // deformation gradient = sum of tensor product between vertex position and shape vector
            // optimize by using displacement with first vertex
            sv=tetinfo->shapeVector[1];
            for (k=0; k<3; ++k)
            {
                for (l=0; l<3; ++l)
                {
                    deformationGradient[k][l]=dp[0][k]*sv[l];
                }
            }
            for (j=1; j<3; ++j)
            {
                sv=tetinfo->shapeVector[j+1];
                for (k=0; k<3; ++k)
                {
                    for (l=0; l<3; ++l)
                    {
                        deformationGradient[k][l]+=dp[j][k]*sv[l];
                    }
                }
            }
            // polar decomposition of the transformation
            helper::Decompose<Real>::polarDecomposition(deformationGradient,R);
        }
        else if (decompositionMethod==QR_DECOMPOSITION)
        {

            /// perform QR decomposition
            computeQRRotation(S,dp);
            R=S.transposed()*tetinfo->restRotation;

        } else if (decompositionMethod==POLAR_DECOMPOSITION_MODIFIED) {

            S[0]=dp[0];
            S[1]=dp[1];
            S[2]=dp[2];
            helper::Decompose<Real>::polarDecomposition( S, R );
            R=R.transposed()*tetinfo->restRotation;
        }  else if (decompositionMethod==LINEAR_ELASTIC) {
            R.identity();
        }
        // store transpose of rotation
        tetinfo->rotation=R.transposed();
        Coord force[4];


        for (j=0; j<6; ++j)
        {
            // displacement in the rest configuration
            dp[j]=tetinfo->rotation*dp[j]-tetinfo->restEdgeVector[j];

            // force on first vertex in the rest configuration
            force[edgesInTetrahedronArray[j][1]]+=tetinfo->linearDfDx[j]*dp[j];

            // force on second vertex in the rest configuration
            force[edgesInTetrahedronArray[j][0]]-=tetinfo->linearDfDx[j].multTranspose(dp[j]);
        }
        for (j=0; j<4; ++j)
        {
            f[ta[j]]+=R*force[j];
        }


    }

    updateMatrix=true; // next time assemble the matrix
    tetrahedronInfo.endEdit();

    dataF.endEdit();

}


template <class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::addDForce(const sofa::core::MechanicalParams* mparams, DataVecDeriv&   datadF , const DataVecDeriv&   datadX )
{
    dmsg_info() << "[" << this->getName() << "]: calling addDForce " ;
    VecDeriv& df       = *(datadF.beginEdit());
    const VecCoord& dx =   datadX.getValue()  ;
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    unsigned int j;
    int i;
    int nbEdges=_topology->getNbEdges();

    if (updateMatrix==true)
    {
        // the matrix must be stored in edges
        helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());
        helper::vector<Mat3x3>& edgeDfDx = *(edgeInfo.beginEdit());

        TetrahedronRestInformation *tetinfo;
        int nbTetrahedra=_topology->getNbTetrahedra();
        Mat3x3 tmp;

        updateMatrix=false;

        // reset all edge matrices
        for(unsigned int j=0; j<edgeDfDx.size(); j++)
        {
            edgeDfDx[j].clear();
        }

        for(i=0; i<nbTetrahedra; i++ )
        {
            tetinfo=&tetrahedronInf[i];
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &tea = _topology->getEdgesInTetrahedron(i);

            for (j=0; j<6; ++j)
            {
                unsigned int edgeID = tea[j];

                // test if the tetrahedron edge has the same orientation as the global edge
                tmp=tetinfo->linearDfDx[j]*tetinfo->rotation;

                if (tetinfo->edgeOrientation[j]==1)
                {
                    // store the two edge matrices since the stiffness matrix is not symmetric
                    edgeDfDx[edgeID] += tetinfo->rotation.transposed()*tmp;
                }
                else
                {
                    edgeDfDx[edgeID] += tmp.transposed()*tetinfo->rotation;
                }
            }
        }

        tetrahedronInfo.endEdit();
        edgeInfo.endEdit();
    }

    const helper::vector<Mat3x3> &edgeDfDx = edgeInfo.getValue();
    Coord deltax;

    // use the already stored matrix
    for(i=0; i<nbEdges; i++ )
    {
        const core::topology::BaseMeshTopology::Edge& edge = _topology->getEdge(i);

        deltax= dx[edge[1]] -dx[edge[0]];
        df[edge[1]]+= edgeDfDx[i] *(deltax * kFactor);
        df[edge[0]]-= edgeDfDx[i].multTranspose(deltax * kFactor);
    }

    datadF.endEdit();
}

template<class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::addKToMatrix(const core::MechanicalParams* mparams, const sofa::core::behavior::MultiMatrixAccessor* matrix )
{
    sofa::core::behavior::MultiMatrixAccessor::MatrixRef r = matrix->getMatrix(this->mstate);
    if (r)
        addKToMatrix(r.matrix, mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()), r.offset);
    else
        serr<<"addKToMatrix found no valid matrix accessor." << sendl;
}


template<class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix *mat, SReal kFactor, unsigned int &offset)
{
    dmsg_info() << "[" << this->getName() << "]: calling addKToMatrix " ;

    unsigned int j;
    int i, matCol, matRow;
    int nbEdges=_topology->getNbEdges();
    int nbPoints=_topology->getNbPoints();
    int nbTetrahedra=_topology->getNbTetrahedra();

    helper::vector<TetrahedronRestInformation>& tetrahedronInf = *(tetrahedronInfo.beginEdit());
    helper::vector<Mat3x3>& edgeDfDx = *(edgeInfo.beginEdit());
    helper::vector<Mat3x3>& pointDfDx = *(pointInfo.beginEdit());

    TetrahedronRestInformation *tetinfo;
    Mat3x3 tmp;

    if (updateMatrix==true) {
        /// if not done in addDForce: update off-diagonal blocks ("edges") of each element matrix
        updateMatrix=false;
        // reset all edge matrices
        for(j=0; j<edgeDfDx.size(); j++)
        {
            edgeDfDx[j].clear();
        }

        for(i=0; i<nbTetrahedra; i++ )
        {
            tetinfo=&tetrahedronInf[i];
            const core::topology::BaseMeshTopology::EdgesInTetrahedron &tea = _topology->getEdgesInTetrahedron(i);

            for (j=0; j<6; ++j)
            {
                unsigned int edgeID = tea[j];

                // test if the tetrahedron edge has the same orientation as the global edge
                tmp=tetinfo->linearDfDx[j]*tetinfo->rotation;

                if (tetinfo->edgeOrientation[j]==1) {
                    // store the two edge matrices since the stiffness sub-matrix is not symmetric
                    edgeDfDx[edgeID] += tetinfo->rotation.transposed()*tmp;
                }
                else {
                    edgeDfDx[edgeID] += tmp.transposed()*tetinfo->rotation;
                }

            }
        }
    }

    /// must update point data since these are not computed in addDForce
    for (j=0; j < pointDfDx.size(); j++)
        pointDfDx[j].clear();

    for(i=0; i<nbTetrahedra; i++ ) {
        tetinfo=&tetrahedronInf[i];
        const core::topology::BaseMeshTopology::Tetrahedron& t = _topology->getTetrahedron(i);

        for (j = 0; j < 4; ++j) {
            unsigned int Id = t[j];
            
            tmp = tetinfo->rotation.transposed() * tetinfo->linearDfDxDiag[j] * tetinfo->rotation;
            pointDfDx[Id] += tmp;
        }
    }

    /// construct the diagonal blocks from point data
    for (i=0; i<nbPoints; i++) {
        tmp = pointDfDx[i];

        for (int m = 0; m < 3; m++) {
            matRow = offset + 3*i + m;
            for (int n = 0; n < 3; n++) {
                matCol = offset + 3*i + n;
                mat->add(matRow, matCol, -kFactor*tmp[m][n]);
            }
        }
    }

    /// construct the off-diagonal blocks from edge data
    for(i=0; i<nbEdges; i++ )
    {
        tmp = edgeDfDx[i];

        const core::topology::BaseMeshTopology::Edge& edge = _topology->getEdge(i);

        for (int m = 0; m < 3; m++) {
            matRow = offset + 3*edge[0] + m;
            for (int n = 0; n < 3; n++) {
                matCol = offset + 3*edge[1] + n;
                mat->add(matRow, matCol, -kFactor*tmp[n][m]);
                mat->add(matCol, matRow, -kFactor*tmp[n][m]);

            }
        }
    }

    tetrahedronInfo.endEdit();
    edgeInfo.endEdit();
    pointInfo.endEdit();
}

template<class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::updateLameCoefficients()
{
    lambda= f_youngModulus.getValue()*f_poissonRatio.getValue()/((1-2*f_poissonRatio.getValue())*(1+f_poissonRatio.getValue()));
    mu = f_youngModulus.getValue()/(2*(1+f_poissonRatio.getValue()));

}


template<class DataTypes>
void FastTetrahedralCorotationalForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (!f_drawing.getValue()) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, true);


    std::vector< defaulttype::Vector3 > points[4];
    for (int i = 0; i<_topology->getNbTetrahedra(); ++i)
    {
        const core::topology::BaseMeshTopology::Tetrahedron t = _topology->getTetrahedron(i);

        Index a = t[0];
        Index b = t[1];
        Index c = t[2];
        Index d = t[3];
        Coord center = (x[a] + x[b] + x[c] + x[d])*0.125;
        Coord pa = (x[a] + center)*(Real)0.666667;
        Coord pb = (x[b] + center)*(Real)0.666667;
        Coord pc = (x[c] + center)*(Real)0.666667;
        Coord pd = (x[d] + center)*(Real)0.666667;

        // 		glColor4f(0,0,1,1);
        points[0].push_back(pa);
        points[0].push_back(pb);
        points[0].push_back(pc);

        // 		glColor4f(0,0.5,1,1);
        points[1].push_back(pb);
        points[1].push_back(pc);
        points[1].push_back(pd);

        // 		glColor4f(0,1,1,1);
        points[2].push_back(pc);
        points[2].push_back(pd);
        points[2].push_back(pa);

        // 		glColor4f(0.5,1,1,1);
        points[3].push_back(pd);
        points[3].push_back(pa);
        points[3].push_back(pb);
    }

    vparams->drawTool()->drawTriangles(points[0], drawColor1.getValue());
    vparams->drawTool()->drawTriangles(points[1], drawColor2.getValue());
    vparams->drawTool()->drawTriangles(points[2], drawColor3.getValue());
    vparams->drawTool()->drawTriangles(points[3], drawColor4.getValue());

    if (vparams->displayFlags().getShowWireFrame())
        vparams->drawTool()->setPolygonMode(0, false);

}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_FASTTETRAHEDRALCOROTATIONALFORCEFIELD_INL
