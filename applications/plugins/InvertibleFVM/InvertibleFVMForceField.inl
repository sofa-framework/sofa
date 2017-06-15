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
#ifndef SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL

#include "InvertibleFVMForceField.h"
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseTopology/GridTopology.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/helper/decompose.h>
#include <sofa/helper/gl/template.h>
#include <assert.h>
#include <iostream>
#include <set>
//#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>




namespace sofa
{

namespace component
{

namespace forcefield
{


using std::set;


using namespace sofa::defaulttype;




template <class DataTypes>
void InvertibleFVMForceField<DataTypes>::reset()
{
    //serr<<"InvertibleFVMForceField<DataTypes>::reset"<<sendl;
}

template <class DataTypes>
void InvertibleFVMForceField<DataTypes>::init()
{
    const VecReal& youngModulus = _youngModulus.getValue();
    minYoung=youngModulus[0];
    maxYoung=youngModulus[0];
    for (unsigned i=0; i<youngModulus.size(); i++)
    {
        if (youngModulus[i]<minYoung) minYoung=youngModulus[i];
        if (youngModulus[i]>maxYoung) maxYoung=youngModulus[i];
    }

    // ParallelDataThrd is used to build the matrix asynchronusly (when listening = true)
    // This feature is activated when callin handleEvent with ParallelizeBuildEvent
    // At init parallelDataSimu == parallelDataThrd (and it's the case since handleEvent is called)

    this->core::behavior::ForceField<DataTypes>::init();
    _mesh = this->getContext()->getMeshTopology();
    if (_mesh==NULL)
    {
        serr << "ERROR(InvertibleFVMForceField): object must have a BaseMeshTopology."<<sendl;
        return;
    }
#ifdef SOFA_NEW_HEXA
    if (_mesh==NULL || (_mesh->getNbTetrahedra()<=0 && _mesh->getNbHexahedra()<=0))
#else
    if (_mesh==NULL || (_mesh->getNbTetrahedra()<=0 && _mesh->getNbCubes()<=0))
#endif
    {
        serr << "ERROR(InvertibleFVMForceField): object must have a tetrahedric BaseMeshTopology."<<sendl;
        return;
    }
    if (!_mesh->getTetrahedra().empty())
    {
        _indexedTetra = & (_mesh->getTetrahedra());
    }
    else
    {
        core::topology::BaseMeshTopology::SeqTetrahedra* tetrahedra = new core::topology::BaseMeshTopology::SeqTetrahedra;
#ifdef SOFA_NEW_HEXA
        int nbcubes = _mesh->getNbHexahedra();
#else
        int nbcubes = _mesh->getNbCubes();
#endif
        // These values are only correct if the mesh is a grid topology
        int nx = 2;
        int ny = 1;
//		int nz = 1;
        {
            topology::GridTopology* grid = dynamic_cast<topology::GridTopology*>(_mesh);
            if (grid != NULL)
            {
                nx = grid->getNx()-1;
                ny = grid->getNy()-1;
//				nz = grid->getNz()-1;
            }
        }

        // Tesselation of each cube into 6 tetrahedra
        tetrahedra->reserve(nbcubes*6);
        for (int i=0; i<nbcubes; i++)
        {
            // if (flags && !flags->isCubeActive(i)) continue;
#ifdef SOFA_NEW_HEXA
            core::topology::BaseMeshTopology::Hexa c = _mesh->getHexahedron(i);
#define swap(a,b) { int t = a; a = b; b = t; }
            if (!((i%nx)&1))
            {
                // swap all points on the X edges
                swap(c[0],c[1]);
                swap(c[3],c[2]);
                swap(c[4],c[5]);
                swap(c[7],c[6]);
            }
            if (((i/nx)%ny)&1)
            {
                // swap all points on the Y edges
                swap(c[0],c[3]);
                swap(c[1],c[2]);
                swap(c[4],c[7]);
                swap(c[5],c[6]);
            }
            if ((i/(nx*ny))&1)
            {
                // swap all points on the Z edges
                swap(c[0],c[4]);
                swap(c[1],c[5]);
                swap(c[2],c[6]);
                swap(c[3],c[7]);
            }
#undef swap
            typedef core::topology::BaseMeshTopology::Tetra Tetra;
            tetrahedra->push_back(Tetra(c[0],c[5],c[1],c[6]));
            tetrahedra->push_back(Tetra(c[0],c[1],c[3],c[6]));
            tetrahedra->push_back(Tetra(c[1],c[3],c[6],c[2]));
            tetrahedra->push_back(Tetra(c[6],c[3],c[0],c[7]));
            tetrahedra->push_back(Tetra(c[6],c[7],c[0],c[5]));
            tetrahedra->push_back(Tetra(c[7],c[5],c[4],c[0]));
#else
            core::topology::BaseMeshTopology::Cube c = _mesh->getCube(i);
            int sym = 0;
            if (!((i%nx)&1)) sym+=1;
            if (((i/nx)%ny)&1) sym+=2;
            if ((i/(nx*ny))&1) sym+=4;
            typedef core::topology::BaseMeshTopology::Tetra Tetra;
            tetrahedra->push_back(Tetra(c[0^sym],c[5^sym],c[1^sym],c[7^sym]));
            tetrahedra->push_back(Tetra(c[0^sym],c[1^sym],c[2^sym],c[7^sym]));
            tetrahedra->push_back(Tetra(c[1^sym],c[2^sym],c[7^sym],c[3^sym]));
            tetrahedra->push_back(Tetra(c[7^sym],c[2^sym],c[0^sym],c[6^sym]));
            tetrahedra->push_back(Tetra(c[7^sym],c[6^sym],c[0^sym],c[5^sym]));
            tetrahedra->push_back(Tetra(c[6^sym],c[5^sym],c[4^sym],c[0^sym]));
#endif
        }

        _indexedTetra = tetrahedra;
    }


    reinit(); // compute per-element stiffness matrices and other precomputed values

//     sout << "InvertibleFVMForceField: init OK, "<<_indexedTetra->size()<<" tetra."<<sendl;
}





template <class DataTypes>
inline void InvertibleFVMForceField<DataTypes>::reinit()
{
    if (!this->mstate) return;
    if (!_mesh->getTetrahedra().empty())
    {
        _indexedTetra = & (_mesh->getTetrahedra());
    }

    //serr<<"InvertibleFVMForceField<DataTypes>::reinit"<<sendl;

    const VecCoord& p = this->mstate->read(core::ConstVecCoordId::restPosition())->getValue();
    _initialPoints.setValue(p);

    _initialTransformation.resize( _indexedTetra->size() );
    _initialRotation.resize( _indexedTetra->size() );
    _U.resize( _indexedTetra->size() );
    _V.resize( _indexedTetra->size() );
    _b.resize( _indexedTetra->size() );

    unsigned int i=0;
    typename VecTetra::const_iterator it;
    for( it = _indexedTetra->begin(), i = 0 ; it != _indexedTetra->end() ; ++it, ++i )
    {
        const Index &a = (*it)[0];
        const Index &b = (*it)[1];
        const Index &c = (*it)[2];
        const Index &d = (*it)[3];

        const VecCoord &initialPoints=_initialPoints.getValue();

        // edges
        Coord ab = initialPoints[b]-initialPoints[a];
        Coord ac = initialPoints[c]-initialPoints[a];
        Coord ad = initialPoints[d]-initialPoints[a];
        Coord bc = initialPoints[c]-initialPoints[b];
        Coord bd = initialPoints[d]-initialPoints[b];
        //Coord cd = initialPoints[d]-initialPoints[c];



        // the initial edge matrix
        Transformation A;
        A[0] = ab;
        A[1] = ac;
        A[2] = ad;

        serr<<"A"<< A<<sendl;



        //Transformation R_0_1;
        helper::Decompose<Real>::polarDecomposition( A, _initialRotation[i] );
        _initialRotation[i].transpose();
        //_initialRotation[i] = R_0_1;
        if( _verbose.getValue() ) serr<<"InvertibleFVMForceField initialRotation "<<_initialRotation[i]<<sendl;

        _initialTransformation[i].invert( _initialRotation[i] * A );


        /*serr<<"R_0_1 * A "<<R_0_1 * A<<sendl;
        serr<<"R_0_1.transposed() * A "<<R_0_1.transposed() * A<<sendl;
        serr<<" A * R_0_1 "<< A*R_0_1<<sendl;
        serr<<" A * R_0_1.transposed() "<< A*R_0_1.transposed()<<sendl;


        ab = R_0_1.transposed() * ab;
        ac = R_0_1.transposed() * ac;
        ad = R_0_1.transposed() * ad;
        bc = R_0_1.transposed() * bc;
        bd = R_0_1.transposed() * bd;

        A[0] = ab;
        A[1] = ac;
        A[2] = ad;

        serr<<"R_0_1 * edges "<<A<<sendl;

        Coord rotateda = R_0_1.transposed() * initialPoints[a];
        Coord rotatedb = R_0_1.transposed() * initialPoints[b];
        Coord rotatedc = R_0_1.transposed() * initialPoints[c];
        Coord rotatedd = R_0_1.transposed() * initialPoints[d];

        serr<<"rotateda "<<rotateda<<sendl;
        serr<<"rotatedb "<<rotatedb<<sendl;
        serr<<"rotatedc "<<rotatedc<<sendl;
        serr<<"rotatedd "<<rotatedd<<sendl;

        ab = rotatedb-rotateda;
        ac = rotatedc-rotateda;
        ad = rotatedd-rotateda;
        bc = rotatedc-rotatedb;
        bd = rotatedd-rotatedb;

        A[0] = ab;
        A[1] = ac;
        A[2] = ad;

        serr<<"rotated A"<< A<<sendl;*/






        /* ab = R_0_1 * ab;
         ac = R_0_1 * ac;
         ad = R_0_1 * ad;
         bc = R_0_1 * bc;
         bd = R_0_1 * bd;*/




        if( _verbose.getValue() ) serr<<"InvertibleFVMForceField _initialTransformation "<<A<<" "<<_initialTransformation[i]<<sendl;


        // the normals (warning: the cross product gives a normal weighted by 2 times the area of the triangle)
        Coord N3 = cross( ab, ac ); // face (a,b,c)
        Coord N2 = cross( ad, ab ); // face (a,d,b)
        Coord N1 = cross( ac, ad ); // face (a,c,d)
        Coord N0 = cross( bd, bc ); // face (b,c,d)

        // the node ordering changes the normal directions
        Real coef = determinant(A)>0 ? (Real)(1/6.0) : (Real)(-1/6.0);

        //_initialRotation[i].transpose();

        ////// compute b_i = -(Nj+Nk+Nl)/3 where N_j are the area-weighted normals of the triangles incident to the node i
        _b[i][0] = /*_initialRotation[i] **/ ( N1 + N2 + N3 ) * coef;
        _b[i][1] = /*_initialRotation[i] **/ ( N0 + N2 + N3 ) * coef;
        _b[i][2] = /*_initialRotation[i] **/ ( N0 + N1 + N3 ) * coef;
        //_b[i][3] = ( N0 + N1 + N2 ) * coef;

        if( _verbose.getValue() && determinant(A) < 0 ) serr<<"detA "<<determinant(A)<<sendl;


        serr<<"InvertibleFVMForceField b "<<sendl;
        serr<<_b[i][0]<<sendl;
        serr<<_b[i][1]<<sendl;
        serr<<_b[i][2]<<sendl;

    }
}


template<class DataTypes>
inline void InvertibleFVMForceField<DataTypes>::addForce (const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& /* d_v */)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& p = d_x.getValue();


    f.resize(p.size());


    unsigned int elementIndex;
    typename VecTetra::const_iterator it;

    for( it=_indexedTetra->begin(), elementIndex=0 ; it!=_indexedTetra->end() ; ++it,++elementIndex )
    {

        const Index &a = (*it)[0];
        const Index &b = (*it)[1];
        const Index &c = (*it)[2];
        const Index &d = (*it)[3];

        Transformation A;
        A[0] = p[b]-p[a];
        A[1] = p[c]-p[a];
        A[2] = p[d]-p[a];

        // tests
        //A[0] = Coord(2,0,0)-p[a];
        //A[1] = Coord(0,2,0)-p[a];
        //A[2] = Coord(0,0,-2)-p[a];
        //A[0] = Coord(1,1,0)-p[a];
        //A[0] = Coord(1,0,-1)-p[a];
        //A[0] = Coord(0.5,0,0)-p[a];
        //A[0] = Coord(1,2,0)-p[a];
        //A[1] = Coord(0.5,0,0)-p[a];
        //A[2] = Coord(1,0,2)-p[a];

        if( _verbose.getValue() ) serr<<"InvertibleFVMForceField currentTransf "<<A<<sendl;



        Mat<3,3,Real> F = /*_initialRotation[elementIndex] **/ A * _initialTransformation[elementIndex];



        if( _verbose.getValue() )  serr<<"InvertibleFVMForceField F "<<F<<" (det= "<<determinant(F)<<")"<<sendl;


        Mat<3,3,Real> U, V; // the two rotations
        Vec<3,Real> F_diagonal, P_diagonal; // diagonalized strain, diagonalized stress

        helper::Decompose<Real>::SVD_stable( F, U, F_diagonal, V );


        // isotrope hookean material defined by P_diag = 2*mu*(F_diag-Id)+lambda*tr(F_diag-Id)*Id
        const VecReal& localStiffnessFactor = _localStiffnessFactor.getValue();
        Real youngModulusElement;
        if (_youngModulus.getValue().size() == _indexedTetra->size()) youngModulusElement = _youngModulus.getValue()[elementIndex];
        else if (_youngModulus.getValue().size() > 0) youngModulusElement = _youngModulus.getValue()[0];
        else
        {
            setYoungModulus(500.0f);
            youngModulusElement = _youngModulus.getValue()[0];
        }
        const Real youngModulus = (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[elementIndex*localStiffnessFactor.size()/_indexedTetra->size()])*youngModulusElement;
        const Real poissonRatio = _poissonRatio.getValue();

        Real lambda = (youngModulus*poissonRatio) / ((1+poissonRatio)*(1-2*poissonRatio));
        Real mu     = youngModulus / (/*2**/(1+poissonRatio));


        //Real volume = helper::rabs( determinant( A ) ) / (Real)6.;
        //lambda *= volume;
        //mu *= volume;

        //serr<<F_diagonal<<sendl;

        F_diagonal[0] -= 1;
        F_diagonal[1] -= 1;
        F_diagonal[2] -= 1;
        P_diagonal = F_diagonal* /*2**/ mu;
        Real tmp = lambda*(F_diagonal[0]+F_diagonal[1]+F_diagonal[2]);
        P_diagonal[0] += tmp;
        P_diagonal[1] += tmp;
        P_diagonal[2] += tmp;


        /*for( int i = 0 ; i<3; ++i )
            if( fabs(P_diagonal[i]) < 1e-6 )
                P_diagonal[i] = 0;*/

        if( _verbose.getValue() ) serr<<"InvertibleFVMForceField P_diagonal "<<P_diagonal<<sendl;




        // TODO optimize this computation without having to use a 3x3 matrix
        Mat<3,3,Real> P; //P_diag_M.clear();
        P[0][0] = P_diagonal[0];
        P[1][1] = P_diagonal[1];
        P[2][2] = P_diagonal[2];

        //if( _verbose.getValue() ) serr<<"InvertibleFVMForceField P "<<P<<sendl;

        P = _initialRotation[elementIndex] * U * P * V.transposed();

        _U[elementIndex].transpose( U );
        _V[elementIndex] = V;



        //serr<<F<<sendl;
//        //F.invert(F);
//        F[0][0] -= 1;
//        F[1][1] -= 1;
//        F[2][2] -= 1;
//        P = F*/*2**/mu;
//        tmp = lambda*(F[0][0]+F[1][1]+F[2][2]);
//        P[0][0] += tmp;
//        P[1][1] += tmp;
//        P[2][2] += tmp;
//        P = _initialRotation[elementIndex].transposed() * P;



        Deriv G0 = P * _b[elementIndex][0];
        Deriv G1 = P * _b[elementIndex][1];
        Deriv G2 = P * _b[elementIndex][2];
        //Deriv G3 = P * _b[elementIndex][3];

        if( _verbose.getValue() )
        {
            serr<<"InvertibleFVMForceField forcesG "<<sendl;
            serr<<G0<<sendl;
            serr<<G1<<sendl;
            serr<<G2<<sendl;
            serr<<(-G0-G1-G2)<<sendl;
        }



        f[a] += G0;
        f[b] += G1;
        f[c] += G2;
        f[d] += (-G0-G1-G2); // null force sum
        //f[d] += G3;

        //if( (G3 +G0+G1+G2).norm() > 1e-5 ) serr<<"force sum not null  "<<G3<<" "<<(-(G0+G1+G2))<<sendl;

    }

    d_f.endEdit();
}

template<class DataTypes>
inline void InvertibleFVMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    return;
    //serr<<"InvertibleFVMForceField::addDForce is not implemented\n";

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());
    unsigned int i;
    typename VecTetra::const_iterator it;


    for(it = _indexedTetra->begin(), i = 0 ; it != _indexedTetra->end() ; ++it, ++i)
    {
        Index a = (*it)[0];
        Index b = (*it)[1];
        Index c = (*it)[2];
        Index d = (*it)[3];


        // edges
        //const VecCoord &initialPoints=_initialPoints.getValue();
        /*Coord ab = initialPoints[b]+dx[b]-(initialPoints[a]+dx[a]);
        Coord ac = initialPoints[c]+dx[c]-(initialPoints[a]+dx[a]);
        Coord ad = initialPoints[d]+dx[d]-(initialPoints[a]+dx[a]);*/
        Coord ab = dx[b]-(dx[a]);
        Coord ac = dx[c]-(dx[a]);
        Coord ad = dx[d]-(dx[a]);

        // the initial edge matrix
        Transformation A;
        A[0] = ab;
        A[1] = ac;
        A[2] = ad;

        Mat<3,3,Real> F = A * _initialTransformation[i];

        //serr<<"addDForce F "<<F<<sendl;

        Mat<3,3,Real> F_diagonal = _U[i] * F * _V[i];



        const VecReal& localStiffnessFactor = _localStiffnessFactor.getValue();
        Real youngModulusElement;
        if (_youngModulus.getValue().size() == _indexedTetra->size()) youngModulusElement = _youngModulus.getValue()[i];
        else if (_youngModulus.getValue().size() > 0) youngModulusElement = _youngModulus.getValue()[0];
        else
        {
            setYoungModulus(500.0f);
            youngModulusElement = _youngModulus.getValue()[0];
        }
        const Real youngModulus = (localStiffnessFactor.empty() ? 1.0f : localStiffnessFactor[i*localStiffnessFactor.size()/_indexedTetra->size()])*youngModulusElement;
        const Real poissonRatio = _poissonRatio.getValue();

        Real lambda = (youngModulus*poissonRatio) / ((1+poissonRatio)*(1-2*poissonRatio));
        Real mu     = youngModulus / (/*2**/(1+poissonRatio));

        //serr<<F_diagonal<<sendl;

        Mat<3,3,Real> P_diagonal;

        F_diagonal[0][0] -= 1;
        F_diagonal[1][1] -= 1;
        F_diagonal[2][2] -= 1;
        P_diagonal = F_diagonal* /*2**/ mu;
        Real tmp = lambda*(F_diagonal[0][0]+F_diagonal[1][1]+F_diagonal[2][2]);
        P_diagonal[0][0] += tmp;
        P_diagonal[1][1] += tmp;
        P_diagonal[2][2] += tmp;


        //serr<<"addDForce F_diagonal "<<F_diagonal<<sendl;
        //serr<<"addDForce P_diagonal "<<P_diagonal<<sendl;


        //serr<<"InvertibleFVMForceField P_diagonal "<<P_diagonal<<sendl;


        Mat<3,3,Real> P = _initialRotation[i] * _U[i].transposed() * P_diagonal * _V[i] * kFactor;


        Deriv G0 = P * _b[i][0];
        Deriv G1 = P * _b[i][1];
        Deriv G2 = P * _b[i][2];
        //Deriv G3 = forceTmp * _b[elementIndex][3];

        df[a] += G0;
        df[b] += G1;
        df[c] += G2;
        df[d] += (-G0-G1-G2);


    }


    d_df.endEdit();
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

template<class DataTypes>
void InvertibleFVMForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;

    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();

    const bool edges = (drawAsEdges.getValue() || vparams->displayFlags().getShowWireFrame());
    const bool heterogeneous = (drawHeterogeneousTetra.getValue() && minYoung!=maxYoung);

    const VecReal & youngModulus = _youngModulus.getValue();
    vparams->drawTool()->setLightingEnabled(false);

    if (edges)
    {
        std::vector< Vector3 > points[3];
        typename VecTetra::const_iterator it;
        int i;
        for(it = _indexedTetra->begin(), i = 0 ; it != _indexedTetra->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            Coord pa = x[a];
            Coord pb = x[b];
            Coord pc = x[c];
            Coord pd = x[d];

            // 		glColor4f(0,0,1,1);
            points[0].push_back(pa);
            points[0].push_back(pb);
            points[0].push_back(pc);
            points[0].push_back(pd);

            // 		glColor4f(0,0.5,1,1);
            points[1].push_back(pa);
            points[1].push_back(pc);
            points[1].push_back(pb);
            points[1].push_back(pd);

            // 		glColor4f(0,1,1,1);
            points[2].push_back(pa);
            points[2].push_back(pd);
            points[2].push_back(pb);
            points[2].push_back(pc);

            if(heterogeneous)
            {
                float col = (float)((youngModulus[i]-minYoung) / (maxYoung-minYoung));
                float fac = col * 0.5f;
                Vec<4,float> color2 = Vec<4,float>(col      , 0.5f - fac , 1.0f-col,1.0f);
                Vec<4,float> color3 = Vec<4,float>(col      , 1.0f - fac , 1.0f-col,1.0f);
                Vec<4,float> color4 = Vec<4,float>(col+0.5f , 1.0f - fac , 1.0f-col,1.0f);

                vparams->drawTool()->drawLines(points[0],1,color2 );
                vparams->drawTool()->drawLines(points[1],1,color3 );
                vparams->drawTool()->drawLines(points[2],1,color4 );

                for(unsigned int i=0 ; i<3 ; i++) points[i].clear();
            }
        }

        if(!heterogeneous)
        {
            vparams->drawTool()->drawLines(points[0], 1, Vec<4,float>(0.0,0.5,1.0,1.0));
            vparams->drawTool()->drawLines(points[1], 1, Vec<4,float>(0.0,1.0,1.0,1.0));
            vparams->drawTool()->drawLines(points[2], 1, Vec<4,float>(0.5,1.0,1.0,1.0));
        }
    }
    else
    {
        std::vector< Vector3 > points[4];
        typename VecTetra::const_iterator it;
        int i;
        for(it = _indexedTetra->begin(), i = 0 ; it != _indexedTetra->end() ; ++it, ++i)
        {
            Index a = (*it)[0];
            Index b = (*it)[1];
            Index c = (*it)[2];
            Index d = (*it)[3];
            //Coord center = (x[a]+x[b]+x[c]+x[d])*0.125;
//            Coord pa = (x[a]+center)*(Real)0.666667;
//            Coord pb = (x[b]+center)*(Real)0.666667;
//            Coord pc = (x[c]+center)*(Real)0.666667;
//            Coord pd = (x[d]+center)*(Real)0.666667;
            Coord pa = x[a];
            Coord pb = x[b];
            Coord pc = x[c];
            Coord pd = x[d];

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

            if(heterogeneous)
            {
                float col = (float)((youngModulus[i]-minYoung) / (maxYoung-minYoung));
                float fac = col * 0.5f;
                Vec<4,float> color1 = Vec<4,float>(col      , 0.0f - fac , 1.0f-col,1.0f);
                Vec<4,float> color2 = Vec<4,float>(col      , 0.5f - fac , 1.0f-col,1.0f);
                Vec<4,float> color3 = Vec<4,float>(col      , 1.0f - fac , 1.0f-col,1.0f);
                Vec<4,float> color4 = Vec<4,float>(col+0.5f , 1.0f - fac , 1.0f-col,1.0f);

                vparams->drawTool()->drawTriangles(points[0],color1 );
                vparams->drawTool()->drawTriangles(points[1],color2 );
                vparams->drawTool()->drawTriangles(points[2],color3 );
                vparams->drawTool()->drawTriangles(points[3],color4 );

                for(unsigned int i=0 ; i<4 ; i++) points[i].clear();
            }

            std::vector< Vector3 > pointsl(2);
            pointsl[0] = x[a];
            pointsl[1] = x[a] - _b[i][0];
            vparams->drawTool()->drawLines( pointsl, 5, Vec<4,float>(1,1,1,1.0f) );
            pointsl[0] = x[b];
            pointsl[1] = x[b] - _b[i][1];
            vparams->drawTool()->drawLines( pointsl, 5, Vec<4,float>(1,1,1,1.0f) );
            pointsl[0] = x[c];
            pointsl[1] = x[c] - _b[i][2];
            vparams->drawTool()->drawLines( pointsl, 5, Vec<4,float>(1,1,1,1.0f) );
            pointsl[0] = x[d];
            pointsl[1] = x[d] + (_b[i][0]+_b[i][1]+_b[i][2]);
            vparams->drawTool()->drawLines( pointsl, 5, Vec<4,float>(1,1,1,1.0f) );


            std::vector< Vector3 > pointsp(1);
            pointsp[0] = x[a];
            vparams->drawTool()->drawPoints( pointsp, 20, Vec<4,float>(1,0,0,1.0f) );
            pointsp[0] = x[b];
            vparams->drawTool()->drawPoints( pointsp, 20, Vec<4,float>(0,1,0,1.0f) );
            pointsp[0] = x[c];
            vparams->drawTool()->drawPoints( pointsp, 20, Vec<4,float>(0,0,1,1.0f) );
            pointsp[0] = x[d];
            vparams->drawTool()->drawPoints( pointsp, 20, Vec<4,float>(1,1,0,1.0f) );



        }

        if(!heterogeneous)
        {
            vparams->drawTool()->drawTriangles(points[0], Vec<4,float>(0.0,0.0,1.0,1.0));
            vparams->drawTool()->drawTriangles(points[1], Vec<4,float>(0.0,0.5,1.0,1.0));
            vparams->drawTool()->drawTriangles(points[2], Vec<4,float>(0.0,1.0,1.0,1.0));
            vparams->drawTool()->drawTriangles(points[3], Vec<4,float>(0.5,1.0,1.0,1.0));
        }

    }

}




template<class DataTypes>
void InvertibleFVMForceField<DataTypes>::addKToMatrix(sofa::defaulttype::BaseMatrix * /*mat*/, SReal /*k*/, unsigned int &/*offset*/)
{
    serr<<"InvertibleFVMForceField::addKToMatrix is not implemented\n";
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_TETRAHEDRONFEMFORCEFIELD_INL
