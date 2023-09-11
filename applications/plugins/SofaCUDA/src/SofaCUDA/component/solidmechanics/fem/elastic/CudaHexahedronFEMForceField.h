/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/solidmechanics/fem/elastic/HexahedronFEMForceField.h>

namespace sofa::gpu::cuda
{

template<class DataTypes>
class CudaKernelsHexahedronFEMForceField;

} // namespace sofa::gpu::cuda

namespace sofa::component::solidmechanics::fem::elastic
{

template <class TCoord, class TDeriv, class TReal>
class HexahedronFEMForceFieldInternalData< gpu::cuda::CudaVectorTypes<TCoord, TDeriv, TReal> > 
{
public:
    typedef gpu::cuda::CudaVectorTypes<TCoord,TDeriv,TReal> DataTypes;
    typedef HexahedronFEMForceField<DataTypes> Main;
    typedef HexahedronFEMForceFieldInternalData<DataTypes> Data;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::Real Real;


    typedef typename Main::Element Element;
    typedef typename Main::VecElement VecElement;
    typedef typename Main::Index Index;
    typedef type::Mat<6, 6, Real> MaterialStiffness;

    typedef type::Mat<24, 24, Real> ElementStiffness;

    typedef type::Mat<3, 3, Real> Transformation;

  	typedef gpu::cuda::CudaKernelsHexahedronFEMForceField<DataTypes> Kernels;	
  
  	struct GPUElement
  	{
          /// @name index of the 8 connected vertices
  		int ia[BSIZE],ib[BSIZE],ic[BSIZE],id[BSIZE],ig[BSIZE],ih[BSIZE],ii[BSIZE],ij[BSIZE];
  
      /// @name initial position of the vertices in the local (rotated) coordinate system
      Real ax[BSIZE],ay[BSIZE],az[BSIZE];
      Real bx[BSIZE],by[BSIZE],bz[BSIZE];
      Real cx[BSIZE],cy[BSIZE],cz[BSIZE];
      Real dx[BSIZE],dy[BSIZE],dz[BSIZE];
      Real gx[BSIZE],gy[BSIZE],gz[BSIZE];
      Real hx[BSIZE],hy[BSIZE],hz[BSIZE];
      Real ix[BSIZE],iy[BSIZE],iz[BSIZE];
      Real jx[BSIZE],jy[BSIZE],jz[BSIZE];

  	};

    gpu::cuda::CudaVector<GPUElement> elems;

     /// Varying data associated with each element
     struct GPURotation
     { 
       /// rotation matrix
       type::Mat<3, 3,Real> Rt;
     };

     struct GPUKMatrix
     {
       /// @name element stifness matrix
       type::Mat<3, 3,Real> K0_0;
       type::Mat<3, 3,Real> K0_1;
       type::Mat<3, 3,Real> K0_2;
       type::Mat<3, 3,Real> K0_3;
       type::Mat<3, 3,Real> K0_4;
       type::Mat<3, 3,Real> K0_5;
       type::Mat<3, 3,Real> K0_6;
       type::Mat<3, 3,Real> K0_7;

       type::Mat<3, 3,Real> K1_0;
       type::Mat<3, 3,Real> K1_1;
       type::Mat<3, 3,Real> K1_2;
       type::Mat<3, 3,Real> K1_3;
       type::Mat<3, 3,Real> K1_4;
       type::Mat<3, 3,Real> K1_5;
       type::Mat<3, 3,Real> K1_6;
       type::Mat<3, 3,Real> K1_7;

       type::Mat<3, 3,Real> K2_0;
       type::Mat<3, 3,Real> K2_1;
       type::Mat<3, 3,Real> K2_2;
       type::Mat<3, 3,Real> K2_3;
       type::Mat<3, 3,Real> K2_4;
       type::Mat<3, 3,Real> K2_5;
       type::Mat<3, 3,Real> K2_6;
       type::Mat<3, 3,Real> K2_7;

       type::Mat<3, 3,Real> K3_0;
       type::Mat<3, 3,Real> K3_1;
       type::Mat<3, 3,Real> K3_2;
       type::Mat<3, 3,Real> K3_3;
       type::Mat<3, 3,Real> K3_4;
       type::Mat<3, 3,Real> K3_5;
       type::Mat<3, 3,Real> K3_6;
       type::Mat<3, 3,Real> K3_7;

       type::Mat<3, 3,Real> K4_0;
       type::Mat<3, 3,Real> K4_1;
       type::Mat<3, 3,Real> K4_2;
       type::Mat<3, 3,Real> K4_3;
       type::Mat<3, 3,Real> K4_4;
       type::Mat<3, 3,Real> K4_5;
       type::Mat<3, 3,Real> K4_6;
       type::Mat<3, 3,Real> K4_7;

       type::Mat<3, 3,Real> K5_0;
       type::Mat<3, 3,Real> K5_1;
       type::Mat<3, 3,Real> K5_2;
       type::Mat<3, 3,Real> K5_3;
       type::Mat<3, 3,Real> K5_4;
       type::Mat<3, 3,Real> K5_5;
       type::Mat<3, 3,Real> K5_6;
       type::Mat<3, 3,Real> K5_7;

       type::Mat<3, 3,Real> K6_0;
       type::Mat<3, 3,Real> K6_1;
       type::Mat<3, 3,Real> K6_2;
       type::Mat<3, 3,Real> K6_3;
       type::Mat<3, 3,Real> K6_4;
       type::Mat<3, 3,Real> K6_5;
       type::Mat<3, 3,Real> K6_6;
       type::Mat<3, 3,Real> K6_7;

       type::Mat<3, 3,Real> K7_0;
       type::Mat<3, 3,Real> K7_1;
       type::Mat<3, 3,Real> K7_2;
       type::Mat<3, 3,Real> K7_3;
       type::Mat<3, 3,Real> K7_4;
       type::Mat<3, 3,Real> K7_5;
       type::Mat<3, 3,Real> K7_6;
       type::Mat<3, 3,Real> K7_7;

     };

    /// Varying data associated with each element
    struct GPUElementForce
    {
        type::Vec<4,Real> fA, fB, fC, fD, fG, fH, fI, fJ;
    };

    gpu::cuda::CudaVector<GPURotation> erotation;
    gpu::cuda::CudaVector<GPURotation> irotation;
    gpu::cuda::CudaVector<GPUKMatrix> ekmatrix;
    gpu::cuda::CudaVector<GPUElementForce> eforce;

    int nbElement; ///< number of elements
    int vertex0; ///< index of the first vertex connected to an element
    int nbVertex; ///< number of vertices to process to compute all elements
    int nbElementPerVertex; ///< max number of elements connected to a vertex
    int GATHER_PT;
    int GATHER_BSIZE;

    /// Index of elements attached to each points (layout per bloc of NBLOC vertices, with first element of each vertex, then second element, etc)
    /// Note that each integer is actually equal to the index of the element * 8 + the index of this vertex inside the hexahedron.
    gpu::cuda::CudaVector<int> velems;

    HexahedronFEMForceFieldInternalData() : nbElement(0), vertex0(0), nbVertex(0), nbElementPerVertex(0) {}

	void init(int nbe, int v0, int nbv, int nbelemperv)
	{
    elems.clear();
    erotation.clear();
    irotation.clear();
    ekmatrix.clear();
    eforce.clear();
    velems.clear();

    nbElement = nbe;
    nbVertex = nbv;
    vertex0 = v0;
    nbElementPerVertex = nbelemperv;
    const int nbloc = (nbVertex+BSIZE-1)/BSIZE;

    elems.resize((nbe+BSIZE-1)/BSIZE);
    erotation.resize(nbe);
    irotation.resize(nbe);
    ekmatrix.resize(nbe);
    eforce.resize(nbe);
    velems.resize(nbloc*nbElementPerVertex*BSIZE);
    for (unsigned int i=0; i<velems.size(); i++)
      velems[i] = 0;

	}

    int size() const { return nbElement; }
    void setV(int vertex, int num, int index)
    {
        vertex -= vertex0;
        const int bloc = vertex/BSIZE;
        const int b_x = vertex%BSIZE;
        velems[ bloc*BSIZE*nbElementPerVertex // start of the bloc
              + num*BSIZE                     // offset to the element
              + b_x                           // offset to the vertex
              ] = index+1;
    }

    void setE(int i, const Element& indices, type::fixed_array<Coord,8> *rotateds)
    {
       GPUElement& e = elems[i/BSIZE]; i = i % BSIZE;
       e.ia[i] = indices[0] - vertex0;
       e.ib[i] = indices[1] - vertex0;
       e.ic[i] = indices[2] - vertex0;
       e.id[i] = indices[3] - vertex0;
       e.ig[i] = indices[4] - vertex0;
       e.ih[i] = indices[5] - vertex0;
       e.ii[i] = indices[6] - vertex0;
       e.ij[i] = indices[7] - vertex0;

       e.ax[i] = rotateds->elems[0][0]; e.ay[i] = rotateds->elems[0][1]; e.az[i] = rotateds->elems[0][2];
       e.bx[i] = rotateds->elems[1][0]; e.by[i] = rotateds->elems[1][1]; e.bz[i] = rotateds->elems[1][2];
       e.cx[i] = rotateds->elems[2][0]; e.cy[i] = rotateds->elems[2][1]; e.cz[i] = rotateds->elems[2][2];
       e.dx[i] = rotateds->elems[3][0]; e.dy[i] = rotateds->elems[3][1]; e.dz[i] = rotateds->elems[3][2];
       e.gx[i] = rotateds->elems[4][0]; e.gy[i] = rotateds->elems[4][1]; e.gz[i] = rotateds->elems[4][2];
       e.hx[i] = rotateds->elems[5][0]; e.hy[i] = rotateds->elems[5][1]; e.hz[i] = rotateds->elems[5][2];
       e.ix[i] = rotateds->elems[6][0]; e.iy[i] = rotateds->elems[6][1]; e.iz[i] = rotateds->elems[6][2];
       e.jx[i] = rotateds->elems[7][0]; e.jy[i] = rotateds->elems[7][1]; e.jz[i] = rotateds->elems[7][2];

    }

    void setS(int i, const Transformation &rotations, const ElementStiffness& K)
    {
      for(unsigned j=0; j<3; j++)
      {
        for(unsigned k=0; k<3; k++)
        {
          erotation[i].Rt[j][k] = rotations[j][k];
          irotation[i].Rt[j][k] = rotations[j][k];
        }
      }

      K.getsub(0, 0, ekmatrix[i].K0_0);
      K.getsub(0, 1*3, ekmatrix[i].K0_1);
      K.getsub(0, 2*3, ekmatrix[i].K0_2);
      K.getsub(0, 3*3, ekmatrix[i].K0_3);
      K.getsub(0, 4*3, ekmatrix[i].K0_4);
      K.getsub(0, 5*3, ekmatrix[i].K0_5);
      K.getsub(0, 6*3, ekmatrix[i].K0_6);
      K.getsub(0, 7*3, ekmatrix[i].K0_7);

      K.getsub(1*3, 0*3, ekmatrix[i].K1_0);
      K.getsub(1*3, 1*3, ekmatrix[i].K1_1);
      K.getsub(1*3, 2*3, ekmatrix[i].K1_2);
      K.getsub(1*3, 3*3, ekmatrix[i].K1_3);
      K.getsub(1*3, 4*3, ekmatrix[i].K1_4);
      K.getsub(1*3, 5*3, ekmatrix[i].K1_5);
      K.getsub(1*3, 6*3, ekmatrix[i].K1_6);
      K.getsub(1*3, 7*3, ekmatrix[i].K1_7);

      K.getsub(2*3, 0*3, ekmatrix[i].K2_0);
      K.getsub(2*3, 1*3, ekmatrix[i].K2_1);
      K.getsub(2*3, 2*3, ekmatrix[i].K2_2);
      K.getsub(2*3, 3*3, ekmatrix[i].K2_3);
      K.getsub(2*3, 4*3, ekmatrix[i].K2_4);
      K.getsub(2*3, 5*3, ekmatrix[i].K2_5);
      K.getsub(2*3, 6*3, ekmatrix[i].K2_6);
      K.getsub(2*3, 7*3, ekmatrix[i].K2_7);

      K.getsub(3*3, 0*3, ekmatrix[i].K3_0);
      K.getsub(3*3, 1*3, ekmatrix[i].K3_1);
      K.getsub(3*3, 2*3, ekmatrix[i].K3_2);
      K.getsub(3*3, 3*3, ekmatrix[i].K3_3);
      K.getsub(3*3, 4*3, ekmatrix[i].K3_4);
      K.getsub(3*3, 5*3, ekmatrix[i].K3_5);
      K.getsub(3*3, 6*3, ekmatrix[i].K3_6);
      K.getsub(3*3, 7*3, ekmatrix[i].K3_7);

      K.getsub(4*3, 0*3, ekmatrix[i].K4_0);
      K.getsub(4*3, 1*3, ekmatrix[i].K4_1);
      K.getsub(4*3, 2*3, ekmatrix[i].K4_2);
      K.getsub(4*3, 3*3, ekmatrix[i].K4_3);
      K.getsub(4*3, 4*3, ekmatrix[i].K4_4);
      K.getsub(4*3, 5*3, ekmatrix[i].K4_5);
      K.getsub(4*3, 6*3, ekmatrix[i].K4_6);
      K.getsub(4*3, 7*3, ekmatrix[i].K4_7);

      K.getsub(5*3, 0*3, ekmatrix[i].K5_0);
      K.getsub(5*3, 1*3, ekmatrix[i].K5_1);
      K.getsub(5*3, 2*3, ekmatrix[i].K5_2);
      K.getsub(5*3, 3*3, ekmatrix[i].K5_3);
      K.getsub(5*3, 4*3, ekmatrix[i].K5_4);
      K.getsub(5*3, 5*3, ekmatrix[i].K5_5);
      K.getsub(5*3, 6*3, ekmatrix[i].K5_6);
      K.getsub(5*3, 7*3, ekmatrix[i].K5_7);

      K.getsub(6*3, 0*3, ekmatrix[i].K6_0);
      K.getsub(6*3, 1*3, ekmatrix[i].K6_1);
      K.getsub(6*3, 2*3, ekmatrix[i].K6_2);
      K.getsub(6*3, 3*3, ekmatrix[i].K6_3);
      K.getsub(6*3, 4*3, ekmatrix[i].K6_4);
      K.getsub(6*3, 5*3, ekmatrix[i].K6_5);
      K.getsub(6*3, 6*3, ekmatrix[i].K6_6);
      K.getsub(6*3, 7*3, ekmatrix[i].K6_7);

      K.getsub(7*3, 0*3, ekmatrix[i].K7_0);
      K.getsub(7*3, 1*3, ekmatrix[i].K7_1);
      K.getsub(7*3, 2*3, ekmatrix[i].K7_2);
      K.getsub(7*3, 3*3, ekmatrix[i].K7_3);
      K.getsub(7*3, 4*3, ekmatrix[i].K7_4);
      K.getsub(7*3, 5*3, ekmatrix[i].K7_5);
      K.getsub(7*3, 6*3, ekmatrix[i].K7_6);
      K.getsub(7*3, 7*3, ekmatrix[i].K7_7);

    }

    void initPtrData(Main* m)
    {
        m->_gatherPt.beginEdit()->setNames({"1","4","8"});
        m->_gatherPt.beginEdit()->setSelectedItem("8");
        m->_gatherPt.endEdit();

        m->_gatherBsize.beginEdit()->setNames({"32","64","128","256"});
        m->_gatherBsize.beginEdit()->setSelectedItem("256");
        m->_gatherBsize.endEdit();
    }

    static void reinit(Main* m);
    static void addForce(Main* m, VecDeriv& f, const VecCoord& x, const VecDeriv& /*v*/);
    static void addDForce(Main* m, VecDeriv& df, const VecDeriv& dx, double kFactor/*, double bFactor*/);
};

#define CudaHexahedronFEMForceField_DeclMethods(T) \
    template<> void HexahedronFEMForceField< T >::reinit(); \
    template<> void HexahedronFEMForceField< T >::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v); \
    template<> void HexahedronFEMForceField< T >::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx); \
    template<> void HexahedronFEMForceField< T >::getRotations(linearalgebra::BaseMatrix* rotations, int offset); \

CudaHexahedronFEMForceField_DeclMethods(gpu::cuda::CudaVec3fTypes);

#undef CudaHexahedronFEMForceField_DeclMethods

} // namespace sofa::component::solidmechanics::fem::elastic
