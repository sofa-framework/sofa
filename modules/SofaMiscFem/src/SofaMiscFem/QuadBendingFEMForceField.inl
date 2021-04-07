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

#include "QuadBendingFEMForceField.h"

#include <sofa/core/topology/BaseMeshTopology.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/ColorMap.h>
#include <sofa/defaulttype/RGBAColor.h>

#include <SofaBaseTopology/TopologyData.inl>

#include <sofa/helper/system/thread/debug.h>
#include <newmat/newmat.h>
#include <newmat/newmatap.h>

#include <fstream> // for reading the file
#include <iostream> //for debugging
#include <vector>
#include <sofa/defaulttype/VecTypes.h>
#include <math.h>
#include <algorithm>
#include <limits>
#include <chrono>
#include <ctime>    
#include <string.h>

#include <iomanip>
#include <sstream>
#include <string>


namespace sofa::component::forcefield

{

using namespace sofa::core::topology;

// --------------------------------------------------------------------------------------
// ---  Topology Creation/Destruction functions
// --------------------------------------------------------------------------------------

template< class DataTypes>
void QuadBendingFEMForceField<DataTypes>::QuadHandler::applyCreateFunction(unsigned int quadIndex, QuadInformation &, 
                                                                           const core::topology::BaseMeshTopology::Quad &t, 
                                                                           const sofa::helper::vector<unsigned int> &, 
                                                                           const sofa::helper::vector<double> &)
{
    if (ff)
    {
        Index idx0 = t[0];
        Index idx1 = t[1];
        Index idx2 = t[2];
        Index idx3 = t[3];
        switch(ff->method)
        {
        case SMALL :
            ff->initSmall(quadIndex,idx0,idx1,idx2,idx3);
            ff->computeBendingMaterialStiffness(quadIndex,idx0,idx1,idx2,idx3);
            ff->computeShearMaterialStiffness(quadIndex,idx0,idx1,idx2,idx3);
            break;

        }
    }
}

// --------------------------------------------------------------------------------------
// --- constructor
// --------------------------------------------------------------------------------------
template <class DataTypes>
QuadBendingFEMForceField<DataTypes>::QuadBendingFEMForceField()
  : quadInfo(initData(&quadInfo,"quadInfo", "Internal quad data"))
  , vertexInfo(initData(&vertexInfo,"vertexInfo", "Internal node data"))
  , edgeInfo(initData(&edgeInfo,"edgeInfo", "Internal edge data"))
  , m_topology(nullptr)
  , method(SMALL)
  , f_method(initData(&f_method,std::string("small"),"method","large: large displacements, small: small displacements"))
  , f_poisson(initData(&f_poisson,helper::vector<Real>(1,static_cast<Real>(0.45)),"poissonRatio","Poisson ratio in Hooke's law (vector)"))
  , f_young(initData(&f_young,helper::vector<Real>(1,static_cast<Real>(1000.0)),"youngModulus","Young modulus in Hooke's law (vector)"))
  , f_thickness(initData(&f_thickness,Real(1.),"thickness","Thickness of the elements"))
  , l_topology(initLink("topology", "link to the topology container"))

{
    quadHandler = new QuadHandler(this, &quadInfo);
}
                
template <class DataTypes>
QuadBendingFEMForceField<DataTypes>::~QuadBendingFEMForceField()
{
    f_poisson.setRequired(true);
    f_young.setRequired(true);
    if(quadHandler) delete quadHandler;
}
    
// --------------------------------------------------------------------------------------
// --- Initialization stage
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::init()
{
    this->Inherited::init();
    if (l_topology.empty())
    {
        msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
        l_topology.set(this->getContext()->getMeshTopologyLink());
    }
  
    m_topology = l_topology.get();
    msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

    if (m_topology == nullptr)
    {
        msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    if (m_topology->getNbQuads() == 0)
    {
        msg_warning() << "No quads found in linked Topology.";
    }
    // Create specific handler for QuadData
    quadInfo.createTopologyHandler(m_topology, quadHandler);
    quadInfo.registerTopologicalData();

    edgeInfo.createTopologyHandler(m_topology);
    edgeInfo.registerTopologicalData();

    vertexInfo.createTopologyHandler(m_topology);
    vertexInfo.registerTopologicalData(); 

    reinit();
}
  
// --------------------------------------------------------------------------------------
// --- Store the initial position of the nodes (small deformation method)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::initSmall(int i, Index&a, Index&b, Index&c, Index&d)
{
  helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());

  QuadInformation *qinfo = &quadInf[i];
  Coord IntlengthElement;
  Coord IntheightElement;
  Coord Intcentroid;
  const  VecCoord& initialPoints = (this->mstate->read(core::ConstVecCoordId::restPosition())->getValue());
  qinfo->IntlengthElement = (initialPoints)[b] - (initialPoints)[a];
  qinfo->IntheightElement = (initialPoints)[d] - (initialPoints)[a];
  qinfo->Intcentroid = ((initialPoints)[a] + (initialPoints)[c])/2.0f;
  qinfo->Inthalflength = (sqrt(IntlengthElement[0]*IntlengthElement[0]+IntlengthElement[1]*IntlengthElement[1]+IntlengthElement[2]*IntlengthElement[2]))/2.0f;
  qinfo->Inthalfheight = (sqrt(IntheightElement[0]*IntheightElement[0]+IntheightElement[1]*IntheightElement[1]+IntheightElement[2]*IntheightElement[2]))/2.0f;
  qinfo->InitialPosElements[0] = (initialPoints)[a] - quadInf[i].Intcentroid; // always (0,0,0)
  qinfo->InitialPosElements[1] = (initialPoints)[b] - quadInf[i].Intcentroid;
  qinfo->InitialPosElements[2] = (initialPoints)[c] - quadInf[i].Intcentroid;
  qinfo->InitialPosElements[3] = (initialPoints)[d] - quadInf[i].Intcentroid;
  
  quadInfo.endEdit();
}
  
  
// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::reinit()
{   

    //---------- check topology data----------------
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
    helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());
    // prepare to store info in the quad array
    quadInf.resize(m_topology->getNbQuads()); 

    // prepare to store info in the edge array
    edgeInf.resize(m_topology->getNbEdges());
  
    unsigned int nbPoints = m_topology->getNbPoints(); 
    helper::vector<VertexInformation>& vi = *(vertexInfo.beginEdit()); 
    vi.resize(nbPoints);
    vertexInfo.endEdit(); 
  
    for (Topology::QuadID i=0; i<m_topology->getNbQuads(); ++i)
    {
        quadHandler->applyCreateFunction(i, quadInf[i],  m_topology->getQuad(i),  
                                         (const sofa::helper::vector< unsigned int > )0, 
                                         (const sofa::helper::vector< double >)0);
    }
    edgeInfo.endEdit();
    quadInfo.endEdit();
}
                
// --------------------------------------------------------------------------------------
// --- Get/Set methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
SReal QuadBendingFEMForceField<DataTypes>::getPotentialEnergy(const core::MechanicalParams* /* mparams */, const DataVecCoord& /* x */) const
{
    msg_error() << "QuadBendingFEMForceField::getPotentialEnergy-not-implemented !!!";
    return 0;
}              
                
// ---------------------------------------------------------------------------------------------------------------
// ---	Compute displacement vector D as the difference between current current position 'p' and initial position in local frame
// ---------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeDisplacementSmall(Displacement &D, Index elementIndex, const VecCoord &p)
{
    Index idx0 = m_topology->getQuad(elementIndex)[0];
    Index idx1 = m_topology->getQuad(elementIndex)[1];
    Index idx2 = m_topology->getQuad(elementIndex)[2];
    Index idx3 = m_topology->getQuad(elementIndex)[3];   

    Coord centroid = (p[idx0]+p[idx2])/2.0f;

    helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());

    Coord deform_t0 = p[idx0]-centroid;  // translation in x direction of node 0
    Coord deform_t1 = p[idx1]-centroid;  // translation in x direction of node 1
    Coord deform_t2 = p[idx2]-centroid;  // translation in x direction of node 2
    Coord deform_t3 = p[idx3]-centroid;  // translation in x direction of node 3  
  
  
    D[0] = quadInf[elementIndex].InitialPosElements[0][0] - deform_t0[0];
    D[1] = quadInf[elementIndex].InitialPosElements[0][1] - deform_t0[1];
    D[2] = quadInf[elementIndex].InitialPosElements[0][2] - deform_t0[2];
    D[3] = 0;  // Assume that the rotational displacements at node are zero
    D[4] = 0;

    D[5] = quadInf[elementIndex].InitialPosElements[1][0] - deform_t1[0];
    D[6] = quadInf[elementIndex].InitialPosElements[1][1] - deform_t1[1];
    D[7] = quadInf[elementIndex].InitialPosElements[1][2] - deform_t1[2];
    D[8] = 0;  
    D[9] = 0;

    D[10] = quadInf[elementIndex].InitialPosElements[2][0] - deform_t2[0];
    D[11] = quadInf[elementIndex].InitialPosElements[2][1] - deform_t2[1];
    D[12] = quadInf[elementIndex].InitialPosElements[2][2] - deform_t2[2];
    D[13] = 0; 
    D[14] = 0;
    D[15] = quadInf[elementIndex].InitialPosElements[3][0] - deform_t3[0];
    D[16] = quadInf[elementIndex].InitialPosElements[3][1] - deform_t3[1];
    D[17] = quadInf[elementIndex].InitialPosElements[3][2] - deform_t3[2];
    D[18] = 0;  
    D[19] = 0;

    quadInfo.endEdit();
}
  
// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix (bending component) where (a, b, c, d) are the coordinates of the 4 nodes of a rectangular
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeBendingStrainDisplacement(StrainDisplacement &Jb, /*Index elementIndex*/ float gauss1, float gauss2, float l, float h/*Coord a, Coord b, Coord c, Coord d*/ )
{

for(int idx=0;idx<4;idx++)
{
  if(idx == 0)
    {const float m = -1.0f;
     const float n = -1.0f;
    // Bmi : membrance stiffness matrices (Gauss integration 2x2)
    Jb[0][0] = Jb[2][1] = m*(1.0f+n*gauss2)/(4.0f*l);  // J[idx0][0][0] = (1*(-1)*(1+(-1)*gauss2))/(4*l)
    Jb[1][1] = Jb[2][0] = n*(1.0f+m*gauss1)/(4.0f*h);  // J[idx0][1][1] = (1*(-1)*(1+(-1)*gauss1))/(4*h)
    }
  else if(idx == 1)
    {const float m = 1.0f;
     const float n = -1.0f;
    Jb[8][5] = Jb[10][6] = m*(1.0f+n*gauss2)/(4.0f*l);  // Ni/x : J[idx0][0][0] = (1*(-1)*(1+(-1)*gauss2))/(4*l)
    Jb[9][6] = Jb[10][5] = n*(1.0f+m*gauss1)/(4.0f*h);  // Ni/y : J[idx0][1][1] = (1*(-1)*(1+(-1)*gauss1))/(4*h) 
    }
  else if(idx == 2)
    {const float m = 1.0f;
     const float n = 1.0f;
    Jb[16][10] = Jb[18][11] = m*(1.0f+n*gauss2)/(4.0f*l);  // Ni/x : J[idx0][0][0] = (1*(-1)*(1+(-1)*gauss2))/(4*l)
    Jb[17][11] = Jb[18][10] = n*(1.0f+m*gauss1)/(4.0f*h);  // Ni/y : J[idx0][1][1] = (1*(-1)*(1+(-1)*gauss1))/(4*h) 
    }
  else if(idx == 3)
    {const float m = -1.0f;
     const float n = 1.0f;
    Jb[24][15] = Jb[26][16] = m*(1.0f+n*gauss2)/(4.0f*l);  // Ni/x : J[idx0][0][0] = (1*(-1)*(1+(-1)*gauss2))/(4*l)
    Jb[25][16] = Jb[26][15] = n*(1.0f+m*gauss1)/(4.0f*h);  // Ni/y : J[idx0][1][1] = (1*(-1)*(1+(-1)*gauss1))/(4*h)
    }


  }
}
  
// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix (Shear componenent) where (a, b, c, d) are the coordinates of the 4 nodes of a rectangular
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeShearStrainDisplacement(StrainDisplacement &Js, /*Index elementIndex*/ float l, float h/*Coord a, Coord b, Coord c, Coord d*/  )
{
  
for(int idx=0;idx<4;idx++)
  {
    if(idx == 0)
      {const float m = -1.0f, n = -1.0f;
      // Bbi : bending stiffness matrices (Gauss integration 1x1)
      Js[3][3] = Js[5][4] = m/(4.0f*l);  //gauss2 = 0
      Js[4][4] = Js[5][3] = n/(4.0f*h);   //gauss1 = 0 
      // Bsi : shear stiffness matrices (Gauss integration 1x1)
      Js[6][2] = m/(4.0f*l);   //gauss2 = 0
      Js[7][2] = n/(4.0f*h);   //gauss1 = 0
      Js[6][3] = Js[7][4] = -1.0f/4.0f; // -Ni : gauss1 = 0 va gauss2 = 0
      }
    else if(idx == 1)
      {const float m = 1.0f, n = -1.0f;
      // Bbi : bending stiffness matrices (Gauss integration 1x1)
      Js[11][8] = Js[13][9] = m/(4.0f*l);  //gauss2 = 0
      Js[12][9] = Js[13][8] = n/(4.0f*h);   //gauss1 = 0 
      // Bsi : shear stiffness matrices (Gauss integration 1x1)
      Js[14][7] = m/(4.0f*l);   //gauss2 = 0
      Js[15][7] = n/(4.0f*h);   //gauss1 = 0
      Js[14][8] = Js[15][9] = -1.0f/4.0f; // -Ni : gauss1 = 0 va gauss2 = 0
      }
    else if(idx == 2)
      {const float m = 1.0f, n = 1.0f;
      // Bbi : bending stiffness matrices (Gauss integration 1x1)
      Js[19][13] = Js[21][14] = m/(4.0f*l);  //gauss2 = 0
      Js[20][14] = Js[21][13] = n/(4.0f*h);   //gauss1 = 0 
      // Bsi : shear stiffness matrices (Gauss integration 1x1)
      Js[22][12] = m/(4.0f*l);   //gauss2 = 0
      Js[23][12] = n/(4.0f*h);   //gauss1 = 0
      Js[22][13] = Js[23][14] = -1.0f/4.0f; // -Ni : gauss1 = 0 va gauss2 = 0
      }
    else if (idx == 3)
      {const float m = -1.0f, n = 1.0f;
      // Bbi : bending stiffness matrices (Gauss integration 1x1)
      Js[27][18] = Js[29][19] = m/(4.0f*l);  //gauss2 = 0
      Js[28][19] = Js[29][18] = n/(4.0f*h);   //gauss1 = 0 
      // Bsi : shear stiffness matrices (Gauss integration 1x1)
      Js[30][17] = m/(4.0f*l);   //gauss2 = 0
      Js[31][17] = n/(4.0f*h);   //gauss1 = 0
      Js[30][18] = Js[31][19] = -1.0f/4.0f; // -Ni : gauss1 = 0 va gauss2 = 0
      }

  }
}

  
// --------------------------------------------------------------------------------------
// ---	Compute material stiffness (bending component)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeBendingMaterialStiffness(int i, Index &/*a*/, Index &/*b*/, Index &/*c*/, Index &/*d*/)
{
  helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());

  const helper::vector<Real> & youngArray = f_young.getValue();
  const helper::vector<Real> & poissonArray = f_poisson.getValue();

  QuadInformation *qinfo = &quadInf[i];

  Real y = ((int)youngArray.size() > i ) ? youngArray[i] : youngArray[0] ;
  Real p = ((int)poissonArray.size() > i ) ? poissonArray[i] : poissonArray[0];
  Real  thickness = f_thickness.getValue();
  MaterialStiffness BendingmaterialMatrix;
  
  // Membrance material stiffness Cm
  qinfo->BendingmaterialMatrix[0][0] = y*thickness/(1.0f-p*p);
  qinfo->BendingmaterialMatrix[0][1] = p*y*thickness/(1.0f-p*p);
  qinfo->BendingmaterialMatrix[1][0] = p*y*thickness/(1.0f-p*p);
  qinfo->BendingmaterialMatrix[1][1] = y*thickness/(1.0f-p*p);
  qinfo->BendingmaterialMatrix[2][2] = y*thickness*(1.0f-p)/(2.0f-2.0f*p*p);

  quadInfo.endEdit();
}
  
// --------------------------------------------------------------------------------------
// ---	Compute material stiffness (shear component)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeShearMaterialStiffness(int i, Index &/*a*/, Index &/*b*/, Index &/*c*/, Index &/*d*/)
{
  helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());

  const helper::vector<Real> & youngArray = f_young.getValue();
  const helper::vector<Real> & poissonArray = f_poisson.getValue();

  QuadInformation *qinfo = &quadInf[i];

  Real y = ((int)youngArray.size() > i ) ? youngArray[i] : youngArray[0] ;
  Real p = ((int)poissonArray.size() > i ) ? poissonArray[i] : poissonArray[0];
  Real thickness = f_thickness.getValue();
  const float k = 5.0f/6.0f;
  MaterialStiffness ShearmaterialMatrix;
  // Bending material stiffness Cb
  qinfo->ShearmaterialMatrix[3][3] = y*thickness*thickness*thickness/(12.0f-12.0f*p*p);
  qinfo->ShearmaterialMatrix[3][4] = p*y*thickness*thickness*thickness/(12.0f-12.0f*p*p);
  qinfo->ShearmaterialMatrix[4][3] = p*y*thickness*thickness*thickness/(12.0f-12.0f*p*p);
  qinfo->ShearmaterialMatrix[4][4] = y*thickness*thickness*thickness/(12.0f-12.0f*p*p);
  qinfo->ShearmaterialMatrix[5][5] = (p*y*thickness*thickness*thickness)*(1.0f-p)/(24.0f-24.0f*p*p);
  // Shear material stiffness Cs
  qinfo->ShearmaterialMatrix[6][6] = k*y*thickness/(2.0f+2.0f*p);
  qinfo->ShearmaterialMatrix[7][7] = k*y*thickness/(2.0f+2.0f*p);
  quadInfo.endEdit();

}
  
// --------------------------------------------------------------------------------------------------------
// --- Stiffness = K = Jt*D*J
// --------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeElementStiffness( Stiffness &K, Index elementIndex)
{  
  helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());
  const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
  //QuadInformation *qinfo = &quadInf[elementIndex];

  Index idx0 = m_topology->getQuad(elementIndex)[0];
  Index idx1 = m_topology->getQuad(elementIndex)[1];
  //Index idx2 = m_topology->getQuad(elementIndex)[2];
  Index idx3 = m_topology->getQuad(elementIndex)[3];
  
  Coord length_vec = p[idx1] - p[idx0];
  Coord height_vec = p[idx3] - p[idx0];
  float length = (sqrt(length_vec[0]*length_vec[0]+length_vec[1]*length_vec[1]+length_vec[2]*length_vec[2]))/2.0f; // length of quad element
  float height = (sqrt(height_vec[0]*height_vec[0]+height_vec[1]*height_vec[1]+height_vec[2]*height_vec[2]))/2.0f; // height of quad element
 
  // Bending component of strain displacement
  defaulttype::Mat<20, 32, Real> Jb0_t;
  StrainDisplacement Jb0;
  computeBendingStrainDisplacement(Jb0, (sqrt(3))/3 , (sqrt(3))/3 , length, height);
  Jb0_t.transpose(Jb0);
  
  defaulttype::Mat<20, 32, Real> Jb1_t;
  StrainDisplacement Jb1;
  computeBendingStrainDisplacement(Jb1,  (-sqrt(3))/3 , (sqrt(3))/3 , length, height);
  Jb1_t.transpose(Jb1);
  
  defaulttype::Mat<20, 32, Real> Jb2_t;
  StrainDisplacement Jb2;
  computeBendingStrainDisplacement(Jb2, (sqrt(3))/3 , (-sqrt(3))/3 , length, height);
  Jb2_t.transpose(Jb2);
  
  defaulttype::Mat<20, 32, Real> Jb3_t;
  StrainDisplacement Jb3;
  computeBendingStrainDisplacement(Jb3, (-sqrt(3))/3 , (-sqrt(3))/3 , length, height);
  Jb3_t.transpose(Jb3);
  // Bending component of material stiffness
  MaterialStiffness Cb;
  Cb = quadInf[elementIndex].BendingmaterialMatrix ;
  
  // Stiffness matrix for bending component
  const float wb = 1.0f; // weight coff of gauss integration 2x2
  Stiffness Kb0;
  Stiffness Kb1;
  Stiffness Kb2;
  Stiffness Kb3;
  Stiffness Kb;
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      Kb0[5*i][5*j]=length*height*(Jb0_t[5*i][8*i]*Cb[0][0]*Jb0[8*j][5*j]+Jb0_t[5*i][8*i+2]*Cb[2][2]*Jb0[8*j+2][5*j])*wb;
      Kb0[5*i][5*j+1]=length*height*(Jb0_t[5*i][8*i]*Cb[0][1]*Jb0[8*j+1][5*j+1]+Jb0_t[5*i][8*i+2]*Cb[2][2]*Jb0[8*j+2][5*j+1])*wb;
      Kb0[5*i+1][5*j]=length*height*(Jb0_t[5*i+1][8*i+1]*Cb[1][0]*Jb0[8*j][5*j]+Jb0_t[5*i+1][8*i+2]*Cb[2][2]*Jb0[8*j+2][5*j])*wb;
      Kb0[5*i+1][5*j+1]=length*height*(Jb0_t[5*i+1][8*i+1]*Cb[1][1]*Jb0[8*j+1][5*j+1]+Jb0_t[5*i+1][8*i+2]*Cb[2][2]*Jb0[8*j+2][5*j+1])*wb;

      Kb1[5*i][5*j]=length*height*(Jb1_t[5*i][8*i]*Cb[0][0]*Jb1[8*j][5*j]+Jb1_t[5*i][8*i+2]*Cb[2][2]*Jb1[8*j+2][5*j])*wb;
      Kb1[5*i][5*j+1]=length*height*(Jb1_t[5*i][8*i]*Cb[0][1]*Jb1[8*j+1][5*j+1]+Jb1_t[5*i][8*i+2]*Cb[2][2]*Jb1[8*j+2][5*j+1])*wb;
      Kb1[5*i+1][5*j]=length*height*(Jb1_t[5*i+1][8*i+1]*Cb[1][0]*Jb1[8*j][5*j]+Jb1_t[5*i+1][8*i+2]*Cb[2][2]*Jb1[8*j+2][5*j])*wb;
      Kb1[5*i+1][5*j+1]=length*height*(Jb1_t[5*i+1][8*i+1]*Cb[1][1]*Jb1[8*j+1][5*j+1]+Jb1_t[5*i+1][8*i+2]*Cb[2][2]*Jb1[8*j+2][5*j+1])*wb;

      Kb2[5*i][5*j]=length*height*(Jb2_t[5*i][8*i]*Cb[0][0]*Jb2[8*j][5*j]+Jb2_t[5*i][8*i+2]*Cb[2][2]*Jb2[8*j+2][5*j])*wb;
      Kb2[5*i][5*j+1]=length*height*(Jb2_t[5*i][8*i]*Cb[0][1]*Jb2[8*j+1][5*j+1]+Jb2_t[5*i][8*i+2]*Cb[2][2]*Jb2[8*j+2][5*j+1])*wb;
      Kb2[5*i+1][5*j]=length*height*(Jb2_t[5*i+1][8*i+1]*Cb[1][0]*Jb2[8*j][5*j]+Jb2_t[5*i+1][8*i+2]*Cb[2][2]*Jb2[8*j+2][5*j])*wb;
      Kb2[5*i+1][5*j+1]=length*height*(Jb2_t[5*i+1][8*i+1]*Cb[1][1]*Jb2[8*j+1][5*j+1]+Jb2_t[5*i+1][8*i+2]*Cb[2][2]*Jb2[8*j+2][5*j+1])*wb;

      Kb3[5*i][5*j]=length*height*(Jb3_t[5*i][8*i]*Cb[0][0]*Jb3[8*j][5*j]+Jb3_t[5*i][8*i+2]*Cb[2][2]*Jb3[8*j+2][5*j])*wb;
      Kb3[5*i][5*j+1]=length*height*(Jb3_t[5*i][8*i]*Cb[0][1]*Jb3[8*j+1][5*j+1]+Jb3_t[5*i][8*i+2]*Cb[2][2]*Jb3[8*j+2][5*j+1])*wb;
      Kb3[5*i+1][5*j]=length*height*(Jb3_t[5*i+1][8*i+1]*Cb[1][0]*Jb3[8*j][5*j]+Jb3_t[5*i+1][8*i+2]*Cb[2][2]*Jb3[8*j+2][5*j])*wb;
      Kb3[5*i+1][5*j+1]=length*height*(Jb3_t[5*i+1][8*i+1]*Cb[1][1]*Jb3[8*j+1][5*j+1]+Jb3_t[5*i+1][8*i+2]*Cb[2][2]*Jb3[8*j+2][5*j+1])*wb;

      Kb[5*i][5*j] = Kb0[5*i][5*j]+Kb1[5*i][5*j]+Kb2[5*i][5*j]+Kb3[5*i][5*j];
      Kb[5*i][5*j+1] = Kb0[5*i][5*j+1]+Kb1[5*i][5*j+1]+Kb2[5*i][5*j+1]+Kb3[5*i][5*j+1];
      Kb[5*i+1][5*j] = Kb0[5*i+1][5*j]+Kb1[5*i+1][5*j]+Kb2[5*i+1][5*j]+Kb3[5*i+1][5*j];
      Kb[5*i+1][5*j+1] = Kb0[5*i+1][5*j+1]+Kb1[5*i+1][5*j+1]+Kb2[5*i+1][5*j+1]+Kb3[5*i+1][5*j+1];
    }
  }

  //Shear Component of strain displacement
  defaulttype::Mat<20, 32, Real> Js_t;
  StrainDisplacement Js;
  computeShearStrainDisplacement(Js, length, height /*p[idx0], p[idx1], p[idx2], p[idx3]*/);
  Js_t.transpose(Js);
  // Shear component of material stiffness
  MaterialStiffness Cs;
  Cs = quadInf[elementIndex].ShearmaterialMatrix ;

  // Stiffness matrix for bending component
  const float ws = 2.0f; // weight coff of gauss integration 1x1
  Stiffness Ks;
  //Ks = length*height*(Js_t*Cs_e*Js)*ws*ws;
  for (int i = 0; i < 4; i++)
  {
    for (int j = 0; j < 4; j++)
    {
      Ks[5*i+2][5*j+2] = length*height*(Js_t[5*i+2][8*i+6]*Cs[6][6]*Js[8*j+6][5*j+2]+Js_t[5*i+2][8*i+7]*Cs[7][7]*Js[8*j+7][5*j+2])*ws*ws;
      Ks[5*i+2][5*j+3] = length*height*(Js_t[5*i+2][8*i+6]*Cs[6][6]*Js[8*j+6][5*j+3])*ws*ws;
      Ks[5*i+2][5*j+4] = length*height*(Js_t[5*i+2][8*i+7]*Cs[7][7]*Js[8*j+7][5*j+4])*ws*ws;
      Ks[5*i+3][5*j+2] = length*height*(Js_t[5*i+3][8*i+6]*Cs[6][6]*Js[8*j+6][5*j+2])*ws*ws;
      Ks[5*i+3][5*j+3] = length*height*(Js_t[5*i+3][8*i+3]*Cs[3][3]*Js[8*j+3][5*j+3]+Js_t[5*i+3][8*i+5]*Cs[5][5]*Js[8*j+5][5*j+3]+Js_t[5*i+3][8*i+6]*Cs[6][6]*Js[8*j+6][5*j+3])*ws*ws;
      Ks[5*i+3][5*j+4] = length*height*(Js_t[5*i+3][8*i+3]*Cs[3][4]*Js[8*j+4][5*j+4]+Js_t[5*i+3][8*i+5]*Cs[5][5]*Js[8*j+5][5*j+4])*ws*ws;
      Ks[5*i+4][5*j+2] = length*height*(Js_t[5*i+4][8*i+7]*Cs[7][7]*Js[8*j+7][5*j+2])*ws*ws;
      Ks[5*i+4][5*j+3] = length*height*(Js_t[5*i+4][8*i+4]*Cs[4][3]*Js[8*j+3][5*j+3]+Js_t[5*i+4][8*i+5]*Cs[5][5]*Js[8*j+5][5*j+3])*ws*ws;
      Ks[5*i+4][5*j+4] = length*height*(Js_t[5*i+4][8*i+4]*Cs[4][4]*Js[8*j+4][5*j+4]+Js_t[5*i+4][8*i+5]*Cs[5][5]*Js[8*j+5][5*j+4]+Js_t[5*i+4][8*i+7]*Cs[7][7]*Js[8*j+7][5*j+4])*ws*ws;
    }
  }
  
  // Stiffness matrix of a element: K = Kb + Ks
  K = Kb + Ks;
  // save stiffness
  quadInf[elementIndex].Bendingstiffness=Kb;
  quadInf[elementIndex].Shearstiffness=Ks;
  quadInfo.endEdit();

}
  
// --------------------------------------------------------------------------------------
// ---	Compute F = K*D
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeForce(Displacement &F, Index elementIndex, Displacement &D)
{
    helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit()); 
    // Tinh stiffness matrix K
    Stiffness K;
    computeElementStiffness(K, elementIndex);
    //F = K * D;
    // Only calculate translational deformation corresponding to Forces F[0](Fx) F[1](Fy) F[2](Fz)
    F[0]=K[0][0]*D[0]+K[0][1]*D[1]+K[0][5]*D[5]+K[0][6]*D[6]+K[0][10]*D[10]+K[0][11]*D[11]+K[0][15]*D[15]+K[0][16]*D[16];
    F[1]=K[1][0]*D[0]+K[1][1]*D[1]+K[1][5]*D[5]+K[1][6]*D[6]+K[1][10]*D[10]+K[1][11]*D[11]+K[1][15]*D[15]+K[1][16]*D[16];
    F[2]=K[2][2]*D[2]+K[2][7]*D[7]+K[2][12]*D[12]+K[2][17]*D[17];   //Assume: D[3]=D[4]=D[8]=D[9]=D[13]=D[14]=D[18]=D[19]=0
    F[3]=0; 
    F[4]=0;
    F[5]=K[5][0]*D[0]+K[5][1]*D[1]+K[5][5]*D[5]+K[5][6]*D[6]+K[5][10]*D[10]+K[5][11]*D[11]+K[5][15]*D[15]+K[5][16]*D[16];
    F[6]=K[6][0]*D[0]+K[6][1]*D[1]+K[6][5]*D[5]+K[6][6]*D[6]+K[6][10]*D[10]+K[6][11]*D[11]+K[6][15]*D[15]+K[6][16]*D[16];
    F[7]=K[7][2]*D[2]+K[7][7]*D[7]+K[7][12]*D[12]+K[7][17]*D[17];   //Assume: D[3]=D[4]=D[8]=D[9]=D[13]=D[14]=D[18]=D[19]=0
    F[8]=0;
    F[9]=0;
    F[10]=K[10][0]*D[0]+K[10][1]*D[1]+K[10][5]*D[5]+K[10][6]*D[6]+K[10][10]*D[10]+K[10][11]*D[11]+K[10][15]*D[15]+K[10][16]*D[16];
    F[11]=K[11][0]*D[0]+K[11][1]*D[1]+K[11][5]*D[5]+K[11][6]*D[6]+K[11][10]*D[10]+K[11][11]*D[11]+K[11][15]*D[15]+K[11][16]*D[16];
    F[12]=K[12][2]*D[2]+K[12][7]*D[7]+K[12][12]*D[12]+K[12][17]*D[17];   //Assume: D[3]=D[4]=D[8]=D[9]=D[13]=D[14]=D[18]=D[19]=0
    F[13]=0;
    F[14]=0;
    F[15]=K[15][0]*D[0]+K[15][1]*D[1]+K[15][5]*D[5]+K[15][6]*D[6]+K[15][10]*D[10]+K[15][11]*D[11]+K[15][15]*D[15]+K[15][16]*D[16];
    F[16]=K[16][0]*D[0]+K[16][1]*D[1]+K[16][5]*D[5]+K[16][6]*D[6]+K[16][10]*D[10]+K[16][11]*D[11]+K[16][15]*D[15]+K[16][16]*D[16];
    F[17]=K[17][2]*D[2]+K[17][7]*D[7]+K[17][12]*D[12]+K[17][17]*D[17];   //Assume: D[3]=D[4]=D[8]=D[9]=D[13]=D[14]=D[18]=D[19]=0
    F[18]=0;
    F[19]=0; //F[19]=K[19][2]*D[2]+K[19][7]*D[7]+K[19][12]*D[12]+K[19][17]*D[17];   //Assume: D[3]=D[4]=D[8]=D[9]=D[13]=D[14]=D[18]=D[19]=0
    quadInf[elementIndex].stiffness = K; 

quadInfo.endEdit();
}

  
// --------------------------------------------------------------------------------------
// --- Apply functions for global frame
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::applyStiffnessSmall(VecCoord &v, Real h, const VecCoord &x, const SReal &kFactor)
{ 
  Displacement D, F;
  
  unsigned int nbQuads = m_topology->getNbQuads();
  
  for (unsigned int i=0;i<nbQuads;i++)
  {
    Index idx0 = m_topology->getQuad(i)[0];
    Index idx1 = m_topology->getQuad(i)[1];
    Index idx2 = m_topology->getQuad(i)[2];
    Index idx3 = m_topology->getQuad(i)[3];
    // Displacement in global frame
    D[0] = x[idx0][0];
    D[1] = x[idx0][1];
    D[2] = x[idx0][2];
    D[3] = 0;
    D[4] = 0;
    
    D[5] = x[idx1][0];
    D[6] = x[idx1][1];
    D[7] = x[idx1][2];
    D[8] = 0;
    D[9] = 0;
    
    D[10] = x[idx2][0];
    D[11] = x[idx2][1];
    D[12] = x[idx2][2];
    D[13] = 0;
    D[14] = 0;
    
    D[15] = x[idx3][0];
    D[16] = x[idx3][1];
    D[17] = x[idx3][2];
    D[18] = 0;
    D[19] = 0;

    computeForce(F, i, D);   

    v[idx0] += (Coord(-h*F[0], -h*F[1],-h*F[2])) * kFactor;
    v[idx1] += (Coord(-h*F[5], -h*F[6], -h*F[7])) * kFactor;
    v[idx2] += (Coord(-h*F[10], -h*F[11], -h*F[12])) * kFactor;
    v[idx3] += (Coord(-h*F[15], -h*F[16], -h*F[17])) * kFactor;

quadInfo.endEdit();
  }
}
  
/*template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::applyStiffnessLarge()
{ 
}*/
  
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::applyStiffness( VecCoord& v, Real h, const VecCoord& x, const SReal &kFactor )
{
  
applyStiffnessSmall( v, h, x, kFactor );

}
  
// --------------------------------------------------------------------------------------
// ---Accumulate function
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::accumulateForceSmall( VecCoord &f, const VecCoord &p, Index elementIndex )
{
  Displacement F, D;
  
  Index idx0 = m_topology->getQuad(elementIndex)[0];
  Index idx1 = m_topology->getQuad(elementIndex)[1];
  Index idx2 = m_topology->getQuad(elementIndex)[2];
  Index idx3 = m_topology->getQuad(elementIndex)[3];
  
  //compute force
  computeDisplacementSmall(D, elementIndex, p);
  computeForce(F,elementIndex,D);
  
  f[idx0] += Coord(F[0],F[1],F[2]);
  f[idx1] += Coord(F[5],F[6],F[7]);
  f[idx2] += Coord(F[10],F[11],F[12]);
  f[idx3] += Coord(F[15],F[16],F[17]);
    
  quadInfo.endEdit();
}
  
/*template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::accumulateForceLarge()
{
  
}*/
// --------------------------------------------------------------------------------------
// --- AddForce and AddDForce methods
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::addForce(const core::MechanicalParams* /* mparams */, DataVecDeriv& f, const DataVecCoord& x, const DataVecDeriv& /* v */)
{
  VecDeriv& f1 = *f.beginEdit();
  const VecCoord& x1 = x.getValue(); 
  int nbQuads=m_topology->getNbQuads();

  f1.resize(x1.size());
  
  for(int i=0;i<nbQuads;i+=1)
  {
    accumulateForceSmall( f1, x1, i );
  }
  f.endEdit();
}
  
// --------------------------------------------------------------------------------------
// --- addDForce
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& df, const DataVecDeriv& dx)  
{
  VecDeriv& df1 = *df.beginEdit();
  const VecDeriv& dx1 = dx.getValue();
  Real kFactor = sofa::core::mechanicalparams::kFactor(mparams);

  Real h=1;
  df1.resize(dx1.size());
  applyStiffnessSmall( df1, h, dx1, kFactor );

  df.endEdit();
}

} // namespace sofa::component::forcefield
