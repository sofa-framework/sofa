/******************************************************************************
*									      *
*			Ho Lab - IoTouch Project			      *
*		       Developer: Nguyen Huu Nhan                             *
*                                  					      *
******************************************************************************/


#ifndef SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_INL

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
#include "config.h"
#include <math.h> 
#include <algorithm>
#include <limits>
#include <chrono>
#include <ctime>    
#include <string.h>
#include <chrono>

#include <iomanip>
#include <sstream>
#include <string>


namespace sofa
{

namespace component
{

namespace forcefield
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
        /*case LARGE :
            ff->initLarge();
            ff->computeBendingMaterialStiffness(quadIndex,idx0,idx1,idx2,idx3);
            ff->computeShearMaterialStiffness(quadIndex,idx0,idx1,idx2,idx3);
            break;*/
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
        sofa::core::objectmodel::BaseObject::d_componentstate.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    if (m_topology->getNbQuads() == 0)
    {
        msg_warning() << "No quads found in linked Topology.";
    }
// Create specific handler for QuadData
    quadInfo.createTopologicalEngine(m_topology, quadHandler);
    quadInfo.registerTopologicalData();

    edgeInfo.createTopologicalEngine(m_topology);
    edgeInfo.registerTopologicalData();

    vertexInfo.createTopologicalEngine(m_topology);
    vertexInfo.registerTopologicalData(); 
  
    /*if (f_method.getValue() == "small")
        method = SMALL;
    else if (f_method.getValue() == "large")
        method = LARGE;*/

    //lastFracturedEdgeIndex = -1; // chua hieu cai nay lam gi

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
  //Real Inthalflength;
  //Real Inthalfheight;
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
// --- Store the initial position of the nodes (large deformation method)
// --------------------------------------------------------------------------------------
/*template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::initLarge()
{
  
}*/
  
// --------------------------------------------------------------------------------------
// --- Re-initialization (called when we change a parameter through the GUI)
// --------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::reinit()
{   
    /*if (f_method.getValue() == "small")
      method = SMALL;
    else if (f_method.getValue() == "large")
      method = LARGE;*/
    //---------- check lai topology data----------------
    helper::vector<EdgeInformation>& edgeInf = *(edgeInfo.beginEdit());
    helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());
    // prepare to store info in the quad array
    quadInf.resize(m_topology->getNbQuads()); //set size cho quadInfo theo so luong quad element

    // prepare to store info in the edge array
    edgeInf.resize(m_topology->getNbEdges());
  
    unsigned int nbPoints = m_topology->getNbPoints();  // check va save lai so luong node
    helper::vector<VertexInformation>& vi = *(vertexInfo.beginEdit()); //save vao trong vertexInformation thong qua bien vi
    vi.resize(nbPoints); // sua lai size cho vi
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

    /*Coord length_vec = p[idx1] - p[idx0];
    Coord height_vec = p[idx3] - p[idx0];
    float length = (length_vec.norm())/2.0f; // do dai` a (chieu dai)
    float height = (height_vec.norm())/2.0f; // do dai` b (chieu cao)*/
    Coord centroid = (p[idx0]+p[idx2])/2.0f;

    helper::vector<QuadInformation>& quadInf = *(quadInfo.beginEdit());

    Coord deform_t0 = p[idx0]-centroid;  // translation in x direction of node 0
    Coord deform_t1 = p[idx1]-centroid;  // translation in x direction of node 1
    Coord deform_t2 = p[idx2]-centroid;  // translation in x direction of node 2
    Coord deform_t3 = p[idx3]-centroid;  // translation in x direction of node 3  
  
  
    D[0] = quadInf[elementIndex].InitialPosElements[0][0] - deform_t0[0];
    D[1] = quadInf[elementIndex].InitialPosElements[0][1] - deform_t0[1];
    D[2] = quadInf[elementIndex].InitialPosElements[0][2] - deform_t0[2];
    D[3] = 0;  // Assume la khong co rotational displacement (i.e., phix va phiy bang 0)
    D[4] = 0;

    D[5] = quadInf[elementIndex].InitialPosElements[1][0] - deform_t1[0];
    D[6] = quadInf[elementIndex].InitialPosElements[1][1] - deform_t1[1];
    D[7] = quadInf[elementIndex].InitialPosElements[1][2] - deform_t1[2];
    D[8] = 0;  // Assume la khong co rotational displacement (i.e., phix va phiy bang 0)
    D[9] = 0;

    D[10] = quadInf[elementIndex].InitialPosElements[2][0] - deform_t2[0];
    D[11] = quadInf[elementIndex].InitialPosElements[2][1] - deform_t2[1];
    D[12] = quadInf[elementIndex].InitialPosElements[2][2] - deform_t2[2];
    D[13] = 0;  // Assume la khong co rotational displacement (i.e., phix va phiy bang 0)
    D[14] = 0;
    D[15] = quadInf[elementIndex].InitialPosElements[3][0] - deform_t3[0];
    D[16] = quadInf[elementIndex].InitialPosElements[3][1] - deform_t3[1];
    D[17] = quadInf[elementIndex].InitialPosElements[3][2] - deform_t3[2];
    D[18] = 0;  // Assume la khong co rotational displacement (i.e., phix va phiy bang 0)
    D[19] = 0;

    quadInfo.endEdit();
}
  
// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix (bending component) where (a, b, c, d) are the coordinates of the 4 nodes of a rectangular
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeBendingStrainDisplacement(StrainDisplacement &Jb, /*Index elementIndex*/ float gauss1, float gauss2, float l, float h/*Coord a, Coord b, Coord c, Coord d*/ )
{
  
  /*Coord lengthElement = b - a;  // vector ab
  Coord heightElement = d - a;  // vector ad
  float l = (lengthElement.norm())/2.0f; // magnitude cua vector ab/2 bang chieu dai a
  float h = (heightElement.norm())/2.0f; // magnitude cua vector ad/2 bang chieu cao b*/
  
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
  //const float w = 1.0f; //weight coeff of gauss intergration method for 2x2 quadrupter
  //Jb *= w;

  }
}
  
// ------------------------------------------------------------------------------------------------------------
// --- Compute the strain-displacement matrix (Shear componenent) where (a, b, c, d) are the coordinates of the 4 nodes of a rectangular
// ------------------------------------------------------------------------------------------------------------
template <class DataTypes>
void QuadBendingFEMForceField<DataTypes>::computeShearStrainDisplacement(StrainDisplacement &Js, /*Index elementIndex*/ float l, float h/*Coord a, Coord b, Coord c, Coord d*/  )
{
  /*Coord lengthElement = b - a;  // vector ab
  Coord heightElement = d - a;  // vector ad
  float l = (lengthElement.norm())/2.0f; // magnitude cua vector ab/2 bang chieu dai a
  float h = (heightElement.norm())/2.0f; // magnitude cua vector ad/2 bang chieu cao b*/
  
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
  //const float w = 4.0f;   //weight2 coeff of gauss intergration method for 1x1 quadrupter w = 2 * 2
  //Js *= w;

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
  float length = (sqrt(length_vec[0]*length_vec[0]+length_vec[1]*length_vec[1]+length_vec[2]*length_vec[2]))/2.0f; // do dai` a (chieu dai)
  float height = (sqrt(height_vec[0]*height_vec[0]+height_vec[1]*height_vec[1]+height_vec[2]*height_vec[2]))/2.0f; // do dai` b (chieu cao)
  //Coord centroid = (p[idx0]+p[idx2])/2.0f;
 
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
  /*// expand Cb from 8x8 to 32x32 diagonal matrix
  defaulttype::Mat<32,32,Real> Cb_e;
  for (unsigned i = 0;i<4;i++)
  {
    for(unsigned j = 0;j<4;j++)
    {
      for(unsigned k = 0;k<8;k++)
      {
        for(unsigned l = 0;l<8;l++)
        {
          Cb_e[8*i+k][8*j+l] = Cb[k][l]; //20x20 matrix
        }
      }
    }
  }*/
  // Stiffness matrix for bending component
  const float wb = 1.0f; // weight coff of gauss integration 2x2
  Stiffness Kb0;
  Kb0[0][0]=length*height*(Jb0_t[0][0]*Cb[0][0]*Jb0[0][0]+Jb0_t[0][2]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[0][1]=length*height*(Jb0_t[0][0]*Cb[0][1]*Jb0[1][1]+Jb0_t[0][2]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[0][5]=length*height*(Jb0_t[0][0]*Cb[0][0]*Jb0[8][5]+Jb0_t[0][2]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[0][6]=length*height*(Jb0_t[0][0]*Cb[0][1]*Jb0[9][6]+Jb0_t[0][2]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[0][10]=length*height*(Jb0_t[0][0]*Cb[0][0]*Jb0[16][10]+Jb0_t[0][2]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[0][11]=length*height*(Jb0_t[0][0]*Cb[0][1]*Jb0[17][11]+Jb0_t[0][2]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[0][15]=length*height*(Jb0_t[0][0]*Cb[0][0]*Jb0[24][15]+Jb0_t[0][2]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[0][16]=length*height*(Jb0_t[0][0]*Cb[0][1]*Jb0[25][16]+Jb0_t[0][2]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[1][0]=length*height*(Jb0_t[1][1]*Cb[1][0]*Jb0[0][0]+Jb0_t[1][2]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[1][1]=length*height*(Jb0_t[1][1]*Cb[1][1]*Jb0[1][1]+Jb0_t[1][2]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[1][5]=length*height*(Jb0_t[1][1]*Cb[1][0]*Jb0[8][5]+Jb0_t[1][2]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[1][6]=length*height*(Jb0_t[1][1]*Cb[1][1]*Jb0[9][6]+Jb0_t[1][2]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[1][10]=length*height*(Jb0_t[1][1]*Cb[1][0]*Jb0[16][10]+Jb0_t[1][2]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[1][11]=length*height*(Jb0_t[1][1]*Cb[1][1]*Jb0[17][11]+Jb0_t[1][2]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[1][15]=length*height*(Jb0_t[1][1]*Cb[1][0]*Jb0[24][15]+Jb0_t[1][2]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[1][16]=length*height*(Jb0_t[1][1]*Cb[1][1]*Jb0[25][16]+Jb0_t[1][2]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[5][0]=length*height*(Jb0_t[5][8]*Cb[0][0]*Jb0[0][0]+Jb0_t[5][10]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[5][1]=length*height*(Jb0_t[5][8]*Cb[0][1]*Jb0[1][1]+Jb0_t[5][10]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[5][5]=length*height*(Jb0_t[5][8]*Cb[0][0]*Jb0[8][5]+Jb0_t[5][10]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[5][6]=length*height*(Jb0_t[5][8]*Cb[0][1]*Jb0[9][6]+Jb0_t[5][10]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[5][10]=length*height*(Jb0_t[5][8]*Cb[0][0]*Jb0[16][10]+Jb0_t[5][10]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[5][11]=length*height*(Jb0_t[5][8]*Cb[0][1]*Jb0[17][11]+Jb0_t[5][10]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[5][15]=length*height*(Jb0_t[5][8]*Cb[0][0]*Jb0[24][15]+Jb0_t[5][10]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[5][16]=length*height*(Jb0_t[5][8]*Cb[0][1]*Jb0[25][16]+Jb0_t[5][10]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[6][0]=length*height*(Jb0_t[6][9]*Cb[1][0]*Jb0[0][0]+Jb0_t[6][10]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[6][1]=length*height*(Jb0_t[6][9]*Cb[1][1]*Jb0[1][1]+Jb0_t[6][10]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[6][5]=length*height*(Jb0_t[6][9]*Cb[1][0]*Jb0[8][5]+Jb0_t[6][10]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[6][6]=length*height*(Jb0_t[6][9]*Cb[1][1]*Jb0[9][6]+Jb0_t[6][10]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[6][10]=length*height*(Jb0_t[6][9]*Cb[1][0]*Jb0[16][10]+Jb0_t[6][10]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[6][11]=length*height*(Jb0_t[6][9]*Cb[1][1]*Jb0[17][11]+Jb0_t[6][10]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[6][15]=length*height*(Jb0_t[6][9]*Cb[1][0]*Jb0[24][15]+Jb0_t[6][10]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[6][16]=length*height*(Jb0_t[6][9]*Cb[1][1]*Jb0[25][16]+Jb0_t[6][10]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[10][0]=length*height*(Jb0_t[10][16]*Cb[0][0]*Jb0[0][0]+Jb0_t[10][18]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[10][1]=length*height*(Jb0_t[10][16]*Cb[0][1]*Jb0[1][1]+Jb0_t[10][18]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[10][5]=length*height*(Jb0_t[10][16]*Cb[0][0]*Jb0[8][5]+Jb0_t[10][18]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[10][6]=length*height*(Jb0_t[10][16]*Cb[0][1]*Jb0[9][6]+Jb0_t[10][18]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[10][10]=length*height*(Jb0_t[10][16]*Cb[0][0]*Jb0[16][10]+Jb0_t[10][18]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[10][11]=length*height*(Jb0_t[10][16]*Cb[0][1]*Jb0[17][11]+Jb0_t[10][18]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[10][15]=length*height*(Jb0_t[10][16]*Cb[0][0]*Jb0[24][15]+Jb0_t[10][18]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[10][16]=length*height*(Jb0_t[10][16]*Cb[0][1]*Jb0[25][16]+Jb0_t[10][18]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[11][0]=length*height*(Jb0_t[11][17]*Cb[1][0]*Jb0[0][0]+Jb0_t[11][18]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[11][1]=length*height*(Jb0_t[11][17]*Cb[1][1]*Jb0[1][1]+Jb0_t[11][18]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[11][5]=length*height*(Jb0_t[11][17]*Cb[1][0]*Jb0[8][5]+Jb0_t[11][18]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[11][6]=length*height*(Jb0_t[11][17]*Cb[1][1]*Jb0[9][6]+Jb0_t[11][18]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[11][10]=length*height*(Jb0_t[11][17]*Cb[1][0]*Jb0[16][10]+Jb0_t[11][18]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[11][11]=length*height*(Jb0_t[11][17]*Cb[1][1]*Jb0[17][11]+Jb0_t[11][18]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[11][15]=length*height*(Jb0_t[11][17]*Cb[1][0]*Jb0[24][15]+Jb0_t[11][18]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[11][16]=length*height*(Jb0_t[11][17]*Cb[1][1]*Jb0[25][16]+Jb0_t[11][18]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[15][0]=length*height*(Jb0_t[15][24]*Cb[0][0]*Jb0[0][0]+Jb0_t[15][26]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[15][1]=length*height*(Jb0_t[15][24]*Cb[0][1]*Jb0[1][1]+Jb0_t[15][26]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[15][5]=length*height*(Jb0_t[15][24]*Cb[0][0]*Jb0[8][5]+Jb0_t[15][26]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[15][6]=length*height*(Jb0_t[15][24]*Cb[0][1]*Jb0[9][6]+Jb0_t[15][26]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[15][10]=length*height*(Jb0_t[15][24]*Cb[0][0]*Jb0[16][10]+Jb0_t[15][26]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[15][11]=length*height*(Jb0_t[15][24]*Cb[0][1]*Jb0[17][11]+Jb0_t[15][26]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[15][15]=length*height*(Jb0_t[15][24]*Cb[0][0]*Jb0[24][15]+Jb0_t[15][26]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[15][16]=length*height*(Jb0_t[15][24]*Cb[0][1]*Jb0[25][16]+Jb0_t[15][26]*Cb[2][2]*Jb0[26][16])*wb;

  Kb0[16][0]=length*height*(Jb0_t[16][25]*Cb[1][0]*Jb0[0][0]+Jb0_t[16][26]*Cb[2][2]*Jb0[2][0])*wb;
  Kb0[16][1]=length*height*(Jb0_t[16][25]*Cb[1][1]*Jb0[1][1]+Jb0_t[16][26]*Cb[2][2]*Jb0[2][1])*wb;
  Kb0[16][5]=length*height*(Jb0_t[16][25]*Cb[1][0]*Jb0[8][5]+Jb0_t[16][26]*Cb[2][2]*Jb0[10][5])*wb;
  Kb0[16][6]=length*height*(Jb0_t[16][25]*Cb[1][1]*Jb0[9][6]+Jb0_t[16][26]*Cb[2][2]*Jb0[10][6])*wb;
  Kb0[16][10]=length*height*(Jb0_t[16][25]*Cb[1][0]*Jb0[16][10]+Jb0_t[16][26]*Cb[2][2]*Jb0[18][10])*wb;
  Kb0[16][11]=length*height*(Jb0_t[16][25]*Cb[1][1]*Jb0[17][11]+Jb0_t[16][26]*Cb[2][2]*Jb0[18][11])*wb;
  Kb0[16][15]=length*height*(Jb0_t[16][25]*Cb[1][0]*Jb0[24][15]+Jb0_t[16][26]*Cb[2][2]*Jb0[26][15])*wb;
  Kb0[16][16]=length*height*(Jb0_t[16][25]*Cb[1][1]*Jb0[25][16]+Jb0_t[16][26]*Cb[2][2]*Jb0[26][16])*wb;

  Stiffness Kb1;
  Kb1[0][0]=length*height*(Jb1_t[0][0]*Cb[0][0]*Jb1[0][0]+Jb1_t[0][2]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[0][1]=length*height*(Jb1_t[0][0]*Cb[0][1]*Jb1[1][1]+Jb1_t[0][2]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[0][5]=length*height*(Jb1_t[0][0]*Cb[0][0]*Jb1[8][5]+Jb1_t[0][2]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[0][6]=length*height*(Jb1_t[0][0]*Cb[0][1]*Jb1[9][6]+Jb1_t[0][2]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[0][10]=length*height*(Jb1_t[0][0]*Cb[0][0]*Jb1[16][10]+Jb1_t[0][2]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[0][11]=length*height*(Jb1_t[0][0]*Cb[0][1]*Jb1[17][11]+Jb1_t[0][2]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[0][15]=length*height*(Jb1_t[0][0]*Cb[0][0]*Jb1[24][15]+Jb1_t[0][2]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[0][16]=length*height*(Jb1_t[0][0]*Cb[0][1]*Jb1[25][16]+Jb1_t[0][2]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[1][0]=length*height*(Jb1_t[1][1]*Cb[1][0]*Jb1[0][0]+Jb1_t[1][2]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[1][1]=length*height*(Jb1_t[1][1]*Cb[1][1]*Jb1[1][1]+Jb1_t[1][2]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[1][5]=length*height*(Jb1_t[1][1]*Cb[1][0]*Jb1[8][5]+Jb1_t[1][2]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[1][6]=length*height*(Jb1_t[1][1]*Cb[1][1]*Jb1[9][6]+Jb1_t[1][2]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[1][10]=length*height*(Jb1_t[1][1]*Cb[1][0]*Jb1[16][10]+Jb1_t[1][2]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[1][11]=length*height*(Jb1_t[1][1]*Cb[1][1]*Jb1[17][11]+Jb1_t[1][2]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[1][15]=length*height*(Jb1_t[1][1]*Cb[1][0]*Jb1[24][15]+Jb1_t[1][2]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[1][16]=length*height*(Jb1_t[1][1]*Cb[1][1]*Jb1[25][16]+Jb1_t[1][2]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[5][0]=length*height*(Jb1_t[5][8]*Cb[0][0]*Jb1[0][0]+Jb1_t[5][10]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[5][1]=length*height*(Jb1_t[5][8]*Cb[0][1]*Jb1[1][1]+Jb1_t[5][10]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[5][5]=length*height*(Jb1_t[5][8]*Cb[0][0]*Jb1[8][5]+Jb1_t[5][10]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[5][6]=length*height*(Jb1_t[5][8]*Cb[0][1]*Jb1[9][6]+Jb1_t[5][10]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[5][10]=length*height*(Jb1_t[5][8]*Cb[0][0]*Jb1[16][10]+Jb1_t[5][10]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[5][11]=length*height*(Jb1_t[5][8]*Cb[0][1]*Jb1[17][11]+Jb1_t[5][10]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[5][15]=length*height*(Jb1_t[5][8]*Cb[0][0]*Jb1[24][15]+Jb1_t[5][10]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[5][16]=length*height*(Jb1_t[5][8]*Cb[0][1]*Jb1[25][16]+Jb1_t[5][10]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[6][0]=length*height*(Jb1_t[6][9]*Cb[1][0]*Jb1[0][0]+Jb1_t[6][10]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[6][1]=length*height*(Jb1_t[6][9]*Cb[1][1]*Jb1[1][1]+Jb1_t[6][10]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[6][5]=length*height*(Jb1_t[6][9]*Cb[1][0]*Jb1[8][5]+Jb1_t[6][10]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[6][6]=length*height*(Jb1_t[6][9]*Cb[1][1]*Jb1[9][6]+Jb1_t[6][10]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[6][10]=length*height*(Jb1_t[6][9]*Cb[1][0]*Jb1[16][10]+Jb1_t[6][10]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[6][11]=length*height*(Jb1_t[6][9]*Cb[1][1]*Jb1[17][11]+Jb1_t[6][10]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[6][15]=length*height*(Jb1_t[6][9]*Cb[1][0]*Jb1[24][15]+Jb1_t[6][10]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[6][16]=length*height*(Jb1_t[6][9]*Cb[1][1]*Jb1[25][16]+Jb1_t[6][10]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[10][0]=length*height*(Jb1_t[10][16]*Cb[0][0]*Jb1[0][0]+Jb1_t[10][18]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[10][1]=length*height*(Jb1_t[10][16]*Cb[0][1]*Jb1[1][1]+Jb1_t[10][18]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[10][5]=length*height*(Jb1_t[10][16]*Cb[0][0]*Jb1[8][5]+Jb1_t[10][18]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[10][6]=length*height*(Jb1_t[10][16]*Cb[0][1]*Jb1[9][6]+Jb1_t[10][18]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[10][10]=length*height*(Jb1_t[10][16]*Cb[0][0]*Jb1[16][10]+Jb1_t[10][18]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[10][11]=length*height*(Jb1_t[10][16]*Cb[0][1]*Jb1[17][11]+Jb1_t[10][18]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[10][15]=length*height*(Jb1_t[10][16]*Cb[0][0]*Jb1[24][15]+Jb1_t[10][18]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[10][16]=length*height*(Jb1_t[10][16]*Cb[0][1]*Jb1[25][16]+Jb1_t[10][18]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[11][0]=length*height*(Jb1_t[11][17]*Cb[1][0]*Jb1[0][0]+Jb1_t[11][18]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[11][1]=length*height*(Jb1_t[11][17]*Cb[1][1]*Jb1[1][1]+Jb1_t[11][18]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[11][5]=length*height*(Jb1_t[11][17]*Cb[1][0]*Jb1[8][5]+Jb1_t[11][18]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[11][6]=length*height*(Jb1_t[11][17]*Cb[1][1]*Jb1[9][6]+Jb1_t[11][18]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[11][10]=length*height*(Jb1_t[11][17]*Cb[1][0]*Jb1[16][10]+Jb1_t[11][18]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[11][11]=length*height*(Jb1_t[11][17]*Cb[1][1]*Jb1[17][11]+Jb1_t[11][18]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[11][15]=length*height*(Jb1_t[11][17]*Cb[1][0]*Jb1[24][15]+Jb1_t[11][18]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[11][16]=length*height*(Jb1_t[11][17]*Cb[1][1]*Jb1[25][16]+Jb1_t[11][18]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[15][0]=length*height*(Jb1_t[15][24]*Cb[0][0]*Jb1[0][0]+Jb1_t[15][26]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[15][1]=length*height*(Jb1_t[15][24]*Cb[0][1]*Jb1[1][1]+Jb1_t[15][26]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[15][5]=length*height*(Jb1_t[15][24]*Cb[0][0]*Jb1[8][5]+Jb1_t[15][26]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[15][6]=length*height*(Jb1_t[15][24]*Cb[0][1]*Jb1[9][6]+Jb1_t[15][26]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[15][10]=length*height*(Jb1_t[15][24]*Cb[0][0]*Jb1[16][10]+Jb1_t[15][26]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[15][11]=length*height*(Jb1_t[15][24]*Cb[0][1]*Jb1[17][11]+Jb1_t[15][26]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[15][15]=length*height*(Jb1_t[15][24]*Cb[0][0]*Jb1[24][15]+Jb1_t[15][26]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[15][16]=length*height*(Jb1_t[15][24]*Cb[0][1]*Jb1[25][16]+Jb1_t[15][26]*Cb[2][2]*Jb1[26][16])*wb;

  Kb1[16][0]=length*height*(Jb1_t[16][25]*Cb[1][0]*Jb1[0][0]+Jb1_t[16][26]*Cb[2][2]*Jb1[2][0])*wb;
  Kb1[16][1]=length*height*(Jb1_t[16][25]*Cb[1][1]*Jb1[1][1]+Jb1_t[16][26]*Cb[2][2]*Jb1[2][1])*wb;
  Kb1[16][5]=length*height*(Jb1_t[16][25]*Cb[1][0]*Jb1[8][5]+Jb1_t[16][26]*Cb[2][2]*Jb1[10][5])*wb;
  Kb1[16][6]=length*height*(Jb1_t[16][25]*Cb[1][1]*Jb1[9][6]+Jb1_t[16][26]*Cb[2][2]*Jb1[10][6])*wb;
  Kb1[16][10]=length*height*(Jb1_t[16][25]*Cb[1][0]*Jb1[16][10]+Jb1_t[16][26]*Cb[2][2]*Jb1[18][10])*wb;
  Kb1[16][11]=length*height*(Jb1_t[16][25]*Cb[1][1]*Jb1[17][11]+Jb1_t[16][26]*Cb[2][2]*Jb1[18][11])*wb;
  Kb1[16][15]=length*height*(Jb1_t[16][25]*Cb[1][0]*Jb1[24][15]+Jb1_t[16][26]*Cb[2][2]*Jb1[26][15])*wb;
  Kb1[16][16]=length*height*(Jb1_t[16][25]*Cb[1][1]*Jb1[25][16]+Jb1_t[16][26]*Cb[2][2]*Jb1[26][16])*wb;

  Stiffness Kb2;
  Kb2[0][0]=length*height*(Jb2_t[0][0]*Cb[0][0]*Jb2[0][0]+Jb2_t[0][2]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[0][1]=length*height*(Jb2_t[0][0]*Cb[0][1]*Jb2[1][1]+Jb2_t[0][2]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[0][5]=length*height*(Jb2_t[0][0]*Cb[0][0]*Jb2[8][5]+Jb2_t[0][2]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[0][6]=length*height*(Jb2_t[0][0]*Cb[0][1]*Jb2[9][6]+Jb2_t[0][2]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[0][10]=length*height*(Jb2_t[0][0]*Cb[0][0]*Jb2[16][10]+Jb2_t[0][2]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[0][11]=length*height*(Jb2_t[0][0]*Cb[0][1]*Jb2[17][11]+Jb2_t[0][2]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[0][15]=length*height*(Jb2_t[0][0]*Cb[0][0]*Jb2[24][15]+Jb2_t[0][2]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[0][16]=length*height*(Jb2_t[0][0]*Cb[0][1]*Jb2[25][16]+Jb2_t[0][2]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[1][0]=length*height*(Jb2_t[1][1]*Cb[1][0]*Jb2[0][0]+Jb2_t[1][2]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[1][1]=length*height*(Jb2_t[1][1]*Cb[1][1]*Jb2[1][1]+Jb2_t[1][2]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[1][5]=length*height*(Jb2_t[1][1]*Cb[1][0]*Jb2[8][5]+Jb2_t[1][2]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[1][6]=length*height*(Jb2_t[1][1]*Cb[1][1]*Jb2[9][6]+Jb2_t[1][2]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[1][10]=length*height*(Jb2_t[1][1]*Cb[1][0]*Jb2[16][10]+Jb2_t[1][2]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[1][11]=length*height*(Jb2_t[1][1]*Cb[1][1]*Jb2[17][11]+Jb2_t[1][2]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[1][15]=length*height*(Jb2_t[1][1]*Cb[1][0]*Jb2[24][15]+Jb2_t[1][2]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[1][16]=length*height*(Jb2_t[1][1]*Cb[1][1]*Jb2[25][16]+Jb2_t[1][2]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[5][0]=length*height*(Jb2_t[5][8]*Cb[0][0]*Jb2[0][0]+Jb2_t[5][10]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[5][1]=length*height*(Jb2_t[5][8]*Cb[0][1]*Jb2[1][1]+Jb2_t[5][10]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[5][5]=length*height*(Jb2_t[5][8]*Cb[0][0]*Jb2[8][5]+Jb2_t[5][10]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[5][6]=length*height*(Jb2_t[5][8]*Cb[0][1]*Jb2[9][6]+Jb2_t[5][10]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[5][10]=length*height*(Jb2_t[5][8]*Cb[0][0]*Jb2[16][10]+Jb2_t[5][10]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[5][11]=length*height*(Jb2_t[5][8]*Cb[0][1]*Jb2[17][11]+Jb2_t[5][10]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[5][15]=length*height*(Jb2_t[5][8]*Cb[0][0]*Jb2[24][15]+Jb2_t[5][10]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[5][16]=length*height*(Jb2_t[5][8]*Cb[0][1]*Jb2[25][16]+Jb2_t[5][10]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[6][0]=length*height*(Jb2_t[6][9]*Cb[1][0]*Jb2[0][0]+Jb2_t[6][10]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[6][1]=length*height*(Jb2_t[6][9]*Cb[1][1]*Jb2[1][1]+Jb2_t[6][10]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[6][5]=length*height*(Jb2_t[6][9]*Cb[1][0]*Jb2[8][5]+Jb2_t[6][10]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[6][6]=length*height*(Jb2_t[6][9]*Cb[1][1]*Jb2[9][6]+Jb2_t[6][10]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[6][10]=length*height*(Jb2_t[6][9]*Cb[1][0]*Jb2[16][10]+Jb2_t[6][10]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[6][11]=length*height*(Jb2_t[6][9]*Cb[1][1]*Jb2[17][11]+Jb2_t[6][10]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[6][15]=length*height*(Jb2_t[6][9]*Cb[1][0]*Jb2[24][15]+Jb2_t[6][10]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[6][16]=length*height*(Jb2_t[6][9]*Cb[1][1]*Jb2[25][16]+Jb2_t[6][10]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[10][0]=length*height*(Jb2_t[10][16]*Cb[0][0]*Jb2[0][0]+Jb2_t[10][18]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[10][1]=length*height*(Jb2_t[10][16]*Cb[0][1]*Jb2[1][1]+Jb2_t[10][18]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[10][5]=length*height*(Jb2_t[10][16]*Cb[0][0]*Jb2[8][5]+Jb2_t[10][18]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[10][6]=length*height*(Jb2_t[10][16]*Cb[0][1]*Jb2[9][6]+Jb2_t[10][18]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[10][10]=length*height*(Jb2_t[10][16]*Cb[0][0]*Jb2[16][10]+Jb2_t[10][18]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[10][11]=length*height*(Jb2_t[10][16]*Cb[0][1]*Jb2[17][11]+Jb2_t[10][18]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[10][15]=length*height*(Jb2_t[10][16]*Cb[0][0]*Jb2[24][15]+Jb2_t[10][18]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[10][16]=length*height*(Jb2_t[10][16]*Cb[0][1]*Jb2[25][16]+Jb2_t[10][18]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[11][0]=length*height*(Jb2_t[11][17]*Cb[1][0]*Jb2[0][0]+Jb2_t[11][18]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[11][1]=length*height*(Jb2_t[11][17]*Cb[1][1]*Jb2[1][1]+Jb2_t[11][18]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[11][5]=length*height*(Jb2_t[11][17]*Cb[1][0]*Jb2[8][5]+Jb2_t[11][18]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[11][6]=length*height*(Jb2_t[11][17]*Cb[1][1]*Jb2[9][6]+Jb2_t[11][18]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[11][10]=length*height*(Jb2_t[11][17]*Cb[1][0]*Jb2[16][10]+Jb2_t[11][18]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[11][11]=length*height*(Jb2_t[11][17]*Cb[1][1]*Jb2[17][11]+Jb2_t[11][18]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[11][15]=length*height*(Jb2_t[11][17]*Cb[1][0]*Jb2[24][15]+Jb2_t[11][18]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[11][16]=length*height*(Jb2_t[11][17]*Cb[1][1]*Jb2[25][16]+Jb2_t[11][18]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[15][0]=length*height*(Jb2_t[15][24]*Cb[0][0]*Jb2[0][0]+Jb2_t[15][26]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[15][1]=length*height*(Jb2_t[15][24]*Cb[0][1]*Jb2[1][1]+Jb2_t[15][26]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[15][5]=length*height*(Jb2_t[15][24]*Cb[0][0]*Jb2[8][5]+Jb2_t[15][26]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[15][6]=length*height*(Jb2_t[15][24]*Cb[0][1]*Jb2[9][6]+Jb2_t[15][26]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[15][10]=length*height*(Jb2_t[15][24]*Cb[0][0]*Jb2[16][10]+Jb2_t[15][26]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[15][11]=length*height*(Jb2_t[15][24]*Cb[0][1]*Jb2[17][11]+Jb2_t[15][26]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[15][15]=length*height*(Jb2_t[15][24]*Cb[0][0]*Jb2[24][15]+Jb2_t[15][26]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[15][16]=length*height*(Jb2_t[15][24]*Cb[0][1]*Jb2[25][16]+Jb2_t[15][26]*Cb[2][2]*Jb2[26][16])*wb;

  Kb2[16][0]=length*height*(Jb2_t[16][25]*Cb[1][0]*Jb2[0][0]+Jb2_t[16][26]*Cb[2][2]*Jb2[2][0])*wb;
  Kb2[16][1]=length*height*(Jb2_t[16][25]*Cb[1][1]*Jb2[1][1]+Jb2_t[16][26]*Cb[2][2]*Jb2[2][1])*wb;
  Kb2[16][5]=length*height*(Jb2_t[16][25]*Cb[1][0]*Jb2[8][5]+Jb2_t[16][26]*Cb[2][2]*Jb2[10][5])*wb;
  Kb2[16][6]=length*height*(Jb2_t[16][25]*Cb[1][1]*Jb2[9][6]+Jb2_t[16][26]*Cb[2][2]*Jb2[10][6])*wb;
  Kb2[16][10]=length*height*(Jb2_t[16][25]*Cb[1][0]*Jb2[16][10]+Jb2_t[16][26]*Cb[2][2]*Jb2[18][10])*wb;
  Kb2[16][11]=length*height*(Jb2_t[16][25]*Cb[1][1]*Jb2[17][11]+Jb2_t[16][26]*Cb[2][2]*Jb2[18][11])*wb;
  Kb2[16][15]=length*height*(Jb2_t[16][25]*Cb[1][0]*Jb2[24][15]+Jb2_t[16][26]*Cb[2][2]*Jb2[26][15])*wb;
  Kb2[16][16]=length*height*(Jb2_t[16][25]*Cb[1][1]*Jb2[25][16]+Jb2_t[16][26]*Cb[2][2]*Jb2[26][16])*wb;

  Stiffness Kb3;
  Kb3[0][0]=length*height*(Jb3_t[0][0]*Cb[0][0]*Jb3[0][0]+Jb3_t[0][2]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[0][1]=length*height*(Jb3_t[0][0]*Cb[0][1]*Jb3[1][1]+Jb3_t[0][2]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[0][5]=length*height*(Jb3_t[0][0]*Cb[0][0]*Jb3[8][5]+Jb3_t[0][2]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[0][6]=length*height*(Jb3_t[0][0]*Cb[0][1]*Jb3[9][6]+Jb3_t[0][2]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[0][10]=length*height*(Jb3_t[0][0]*Cb[0][0]*Jb3[16][10]+Jb3_t[0][2]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[0][11]=length*height*(Jb3_t[0][0]*Cb[0][1]*Jb3[17][11]+Jb3_t[0][2]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[0][15]=length*height*(Jb3_t[0][0]*Cb[0][0]*Jb3[24][15]+Jb3_t[0][2]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[0][16]=length*height*(Jb3_t[0][0]*Cb[0][1]*Jb3[25][16]+Jb3_t[0][2]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[1][0]=length*height*(Jb3_t[1][1]*Cb[1][0]*Jb3[0][0]+Jb3_t[1][2]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[1][1]=length*height*(Jb3_t[1][1]*Cb[1][1]*Jb3[1][1]+Jb3_t[1][2]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[1][5]=length*height*(Jb3_t[1][1]*Cb[1][0]*Jb3[8][5]+Jb3_t[1][2]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[1][6]=length*height*(Jb3_t[1][1]*Cb[1][1]*Jb3[9][6]+Jb3_t[1][2]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[1][10]=length*height*(Jb3_t[1][1]*Cb[1][0]*Jb3[16][10]+Jb3_t[1][2]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[1][11]=length*height*(Jb3_t[1][1]*Cb[1][1]*Jb3[17][11]+Jb3_t[1][2]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[1][15]=length*height*(Jb3_t[1][1]*Cb[1][0]*Jb3[24][15]+Jb3_t[1][2]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[1][16]=length*height*(Jb3_t[1][1]*Cb[1][1]*Jb3[25][16]+Jb3_t[1][2]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[5][0]=length*height*(Jb3_t[5][8]*Cb[0][0]*Jb3[0][0]+Jb3_t[5][10]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[5][1]=length*height*(Jb3_t[5][8]*Cb[0][1]*Jb3[1][1]+Jb3_t[5][10]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[5][5]=length*height*(Jb3_t[5][8]*Cb[0][0]*Jb3[8][5]+Jb3_t[5][10]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[5][6]=length*height*(Jb3_t[5][8]*Cb[0][1]*Jb3[9][6]+Jb3_t[5][10]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[5][10]=length*height*(Jb3_t[5][8]*Cb[0][0]*Jb3[16][10]+Jb3_t[5][10]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[5][11]=length*height*(Jb3_t[5][8]*Cb[0][1]*Jb3[17][11]+Jb3_t[5][10]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[5][15]=length*height*(Jb3_t[5][8]*Cb[0][0]*Jb3[24][15]+Jb3_t[5][10]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[5][16]=length*height*(Jb3_t[5][8]*Cb[0][1]*Jb3[25][16]+Jb3_t[5][10]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[6][0]=length*height*(Jb3_t[6][9]*Cb[1][0]*Jb3[0][0]+Jb3_t[6][10]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[6][1]=length*height*(Jb3_t[6][9]*Cb[1][1]*Jb3[1][1]+Jb3_t[6][10]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[6][5]=length*height*(Jb3_t[6][9]*Cb[1][0]*Jb3[8][5]+Jb3_t[6][10]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[6][6]=length*height*(Jb3_t[6][9]*Cb[1][1]*Jb3[9][6]+Jb3_t[6][10]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[6][10]=length*height*(Jb3_t[6][9]*Cb[1][0]*Jb3[16][10]+Jb3_t[6][10]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[6][11]=length*height*(Jb3_t[6][9]*Cb[1][1]*Jb3[17][11]+Jb3_t[6][10]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[6][15]=length*height*(Jb3_t[6][9]*Cb[1][0]*Jb3[24][15]+Jb3_t[6][10]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[6][16]=length*height*(Jb3_t[6][9]*Cb[1][1]*Jb3[25][16]+Jb3_t[6][10]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[10][0]=length*height*(Jb3_t[10][16]*Cb[0][0]*Jb3[0][0]+Jb3_t[10][18]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[10][1]=length*height*(Jb3_t[10][16]*Cb[0][1]*Jb3[1][1]+Jb3_t[10][18]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[10][5]=length*height*(Jb3_t[10][16]*Cb[0][0]*Jb3[8][5]+Jb3_t[10][18]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[10][6]=length*height*(Jb3_t[10][16]*Cb[0][1]*Jb3[9][6]+Jb3_t[10][18]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[10][10]=length*height*(Jb3_t[10][16]*Cb[0][0]*Jb3[16][10]+Jb3_t[10][18]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[10][11]=length*height*(Jb3_t[10][16]*Cb[0][1]*Jb3[17][11]+Jb3_t[10][18]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[10][15]=length*height*(Jb3_t[10][16]*Cb[0][0]*Jb3[24][15]+Jb3_t[10][18]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[10][16]=length*height*(Jb3_t[10][16]*Cb[0][1]*Jb3[25][16]+Jb3_t[10][18]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[11][0]=length*height*(Jb3_t[11][17]*Cb[1][0]*Jb3[0][0]+Jb3_t[11][18]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[11][1]=length*height*(Jb3_t[11][17]*Cb[1][1]*Jb3[1][1]+Jb3_t[11][18]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[11][5]=length*height*(Jb3_t[11][17]*Cb[1][0]*Jb3[8][5]+Jb3_t[11][18]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[11][6]=length*height*(Jb3_t[11][17]*Cb[1][1]*Jb3[9][6]+Jb3_t[11][18]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[11][10]=length*height*(Jb3_t[11][17]*Cb[1][0]*Jb3[16][10]+Jb3_t[11][18]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[11][11]=length*height*(Jb3_t[11][17]*Cb[1][1]*Jb3[17][11]+Jb3_t[11][18]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[11][15]=length*height*(Jb3_t[11][17]*Cb[1][0]*Jb3[24][15]+Jb3_t[11][18]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[11][16]=length*height*(Jb3_t[11][17]*Cb[1][1]*Jb3[25][16]+Jb3_t[11][18]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[15][0]=length*height*(Jb3_t[15][24]*Cb[0][0]*Jb3[0][0]+Jb3_t[15][26]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[15][1]=length*height*(Jb3_t[15][24]*Cb[0][1]*Jb3[1][1]+Jb3_t[15][26]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[15][5]=length*height*(Jb3_t[15][24]*Cb[0][0]*Jb3[8][5]+Jb3_t[15][26]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[15][6]=length*height*(Jb3_t[15][24]*Cb[0][1]*Jb3[9][6]+Jb3_t[15][26]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[15][10]=length*height*(Jb3_t[15][24]*Cb[0][0]*Jb3[16][10]+Jb3_t[15][26]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[15][11]=length*height*(Jb3_t[15][24]*Cb[0][1]*Jb3[17][11]+Jb3_t[15][26]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[15][15]=length*height*(Jb3_t[15][24]*Cb[0][0]*Jb3[24][15]+Jb3_t[15][26]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[15][16]=length*height*(Jb3_t[15][24]*Cb[0][1]*Jb3[25][16]+Jb3_t[15][26]*Cb[2][2]*Jb3[26][16])*wb;

  Kb3[16][0]=length*height*(Jb3_t[16][25]*Cb[1][0]*Jb3[0][0]+Jb3_t[16][26]*Cb[2][2]*Jb3[2][0])*wb;
  Kb3[16][1]=length*height*(Jb3_t[16][25]*Cb[1][1]*Jb3[1][1]+Jb3_t[16][26]*Cb[2][2]*Jb3[2][1])*wb;
  Kb3[16][5]=length*height*(Jb3_t[16][25]*Cb[1][0]*Jb3[8][5]+Jb3_t[16][26]*Cb[2][2]*Jb3[10][5])*wb;
  Kb3[16][6]=length*height*(Jb3_t[16][25]*Cb[1][1]*Jb3[9][6]+Jb3_t[16][26]*Cb[2][2]*Jb3[10][6])*wb;
  Kb3[16][10]=length*height*(Jb3_t[16][25]*Cb[1][0]*Jb3[16][10]+Jb3_t[16][26]*Cb[2][2]*Jb3[18][10])*wb;
  Kb3[16][11]=length*height*(Jb3_t[16][25]*Cb[1][1]*Jb3[17][11]+Jb3_t[16][26]*Cb[2][2]*Jb3[18][11])*wb;
  Kb3[16][15]=length*height*(Jb3_t[16][25]*Cb[1][0]*Jb3[24][15]+Jb3_t[16][26]*Cb[2][2]*Jb3[26][15])*wb;
  Kb3[16][16]=length*height*(Jb3_t[16][25]*Cb[1][1]*Jb3[25][16]+Jb3_t[16][26]*Cb[2][2]*Jb3[26][16])*wb;

  Stiffness Kb;
  //Kb = length*height*(Jb0_t*Cb_e*Jb0+Jb1_t*Cb_e*Jb1+Jb2_t*Cb_e*Jb2+Jb3_t*Cb_e*Jb3)*wb;  // bending stiffness
  Kb[0][0]=Kb0[0][0]+Kb1[0][0]+Kb2[0][0]+Kb3[0][0];
  Kb[0][1]=Kb0[0][1]+Kb1[0][1]+Kb2[0][1]+Kb3[0][1];
  Kb[0][5]=Kb0[0][5]+Kb1[0][5]+Kb2[0][5]+Kb3[0][5];
  Kb[0][6]=Kb0[0][6]+Kb1[0][6]+Kb2[0][6]+Kb3[0][6];
  Kb[0][10]=Kb0[0][10]+Kb1[0][10]+Kb2[0][10]+Kb3[0][10];
  Kb[0][11]=Kb0[0][11]+Kb1[0][11]+Kb2[0][11]+Kb3[0][11];
  Kb[0][15]=Kb0[0][15]+Kb1[0][15]+Kb2[0][15]+Kb3[0][15];
  Kb[0][16]=Kb0[0][16]+Kb1[0][16]+Kb2[0][16]+Kb3[0][16];

  Kb[1][0]=Kb0[1][0]+Kb1[1][0]+Kb2[1][0]+Kb3[1][0];
  Kb[1][1]=Kb0[1][1]+Kb1[1][1]+Kb2[1][1]+Kb3[1][1];
  Kb[1][5]=Kb0[1][5]+Kb1[1][5]+Kb2[1][5]+Kb3[1][5];
  Kb[1][6]=Kb0[1][6]+Kb1[1][6]+Kb2[1][6]+Kb3[1][6];
  Kb[1][10]=Kb0[1][10]+Kb1[1][10]+Kb2[1][10]+Kb3[1][10];
  Kb[1][11]=Kb0[1][11]+Kb1[1][11]+Kb2[1][11]+Kb3[1][11];
  Kb[1][15]=Kb0[1][15]+Kb1[1][15]+Kb2[1][15]+Kb3[1][15];
  Kb[1][16]=Kb0[1][16]+Kb1[1][16]+Kb2[1][16]+Kb3[1][16];

  Kb[5][0]=Kb0[5][0]+Kb1[5][0]+Kb2[5][0]+Kb3[5][0];
  Kb[5][1]=Kb0[5][1]+Kb1[5][1]+Kb2[5][1]+Kb3[5][1];
  Kb[5][5]=Kb0[5][5]+Kb1[5][5]+Kb2[5][5]+Kb3[5][5];
  Kb[5][6]=Kb0[5][6]+Kb1[5][6]+Kb2[5][6]+Kb3[5][6];
  Kb[5][10]=Kb0[5][10]+Kb1[5][10]+Kb2[5][10]+Kb3[5][10];
  Kb[5][11]=Kb0[5][11]+Kb1[5][11]+Kb2[5][11]+Kb3[5][11];
  Kb[5][15]=Kb0[5][15]+Kb1[5][15]+Kb2[5][15]+Kb3[5][15];
  Kb[5][16]=Kb0[5][16]+Kb1[5][16]+Kb2[5][16]+Kb3[5][16];

  Kb[6][0]=Kb0[6][0]+Kb1[6][0]+Kb2[6][0]+Kb3[6][0];
  Kb[6][1]=Kb0[6][1]+Kb1[6][1]+Kb2[6][1]+Kb3[6][1];
  Kb[6][5]=Kb0[6][5]+Kb1[6][5]+Kb2[6][5]+Kb3[6][5];
  Kb[6][6]=Kb0[6][6]+Kb1[6][6]+Kb2[6][6]+Kb3[6][6];
  Kb[6][10]=Kb0[6][10]+Kb1[6][10]+Kb2[6][10]+Kb3[6][10];
  Kb[6][11]=Kb0[6][11]+Kb1[6][11]+Kb2[6][11]+Kb3[6][11];
  Kb[6][15]=Kb0[6][15]+Kb1[6][15]+Kb2[6][15]+Kb3[6][15];
  Kb[6][16]=Kb0[6][16]+Kb1[6][16]+Kb2[6][16]+Kb3[6][16];

  Kb[10][0]=Kb0[10][0]+Kb1[10][0]+Kb2[10][0]+Kb3[10][0];
  Kb[10][1]=Kb0[10][1]+Kb1[10][1]+Kb2[10][1]+Kb3[10][1];
  Kb[10][5]=Kb0[10][5]+Kb1[10][5]+Kb2[10][5]+Kb3[10][5];
  Kb[10][6]=Kb0[10][6]+Kb1[10][6]+Kb2[10][6]+Kb3[10][6];
  Kb[10][10]=Kb0[10][10]+Kb1[10][10]+Kb2[10][10]+Kb3[10][10];
  Kb[10][11]=Kb0[10][11]+Kb1[10][11]+Kb2[10][11]+Kb3[10][11];
  Kb[10][15]=Kb0[10][15]+Kb1[10][15]+Kb2[10][15]+Kb3[10][15];
  Kb[10][16]=Kb0[10][16]+Kb1[10][16]+Kb2[10][16]+Kb3[10][16];

  Kb[11][0]=Kb0[11][0]+Kb1[11][0]+Kb2[11][0]+Kb3[11][0];
  Kb[11][1]=Kb0[11][1]+Kb1[11][1]+Kb2[11][1]+Kb3[11][1];
  Kb[11][5]=Kb0[11][5]+Kb1[11][5]+Kb2[11][5]+Kb3[11][5];
  Kb[11][6]=Kb0[11][6]+Kb1[11][6]+Kb2[11][6]+Kb3[11][6];
  Kb[11][10]=Kb0[11][10]+Kb1[11][10]+Kb2[11][10]+Kb3[11][10];
  Kb[11][11]=Kb0[11][11]+Kb1[11][11]+Kb2[11][11]+Kb3[11][11];
  Kb[11][15]=Kb0[11][15]+Kb1[11][15]+Kb2[11][15]+Kb3[11][15];
  Kb[11][16]=Kb0[11][16]+Kb1[11][16]+Kb2[11][16]+Kb3[11][16];

  Kb[15][0]=Kb0[15][0]+Kb1[15][0]+Kb2[15][0]+Kb3[15][0];
  Kb[15][1]=Kb0[15][1]+Kb1[15][1]+Kb2[15][1]+Kb3[15][1];
  Kb[15][5]=Kb0[15][5]+Kb1[15][5]+Kb2[15][5]+Kb3[15][5];
  Kb[15][6]=Kb0[15][6]+Kb1[15][6]+Kb2[15][6]+Kb3[15][6];
  Kb[15][10]=Kb0[15][10]+Kb1[15][10]+Kb2[15][10]+Kb3[15][10];
  Kb[15][11]=Kb0[15][11]+Kb1[15][11]+Kb2[15][11]+Kb3[15][11];
  Kb[15][15]=Kb0[15][15]+Kb1[15][15]+Kb2[15][15]+Kb3[15][15];
  Kb[15][16]=Kb0[15][16]+Kb1[15][16]+Kb2[15][16]+Kb3[15][16];

  Kb[16][0]=Kb0[16][0]+Kb1[16][0]+Kb2[16][0]+Kb3[16][0];
  Kb[16][1]=Kb0[16][1]+Kb1[16][1]+Kb2[16][1]+Kb3[16][1];
  Kb[16][5]=Kb0[16][5]+Kb1[16][5]+Kb2[16][5]+Kb3[16][5];
  Kb[16][6]=Kb0[16][6]+Kb1[16][6]+Kb2[16][6]+Kb3[16][6];
  Kb[16][10]=Kb0[16][10]+Kb1[16][10]+Kb2[16][10]+Kb3[16][10];
  Kb[16][11]=Kb0[16][11]+Kb1[16][11]+Kb2[16][11]+Kb3[16][11];
  Kb[16][15]=Kb0[16][15]+Kb1[16][15]+Kb2[16][15]+Kb3[16][15];
  Kb[16][16]=Kb0[16][16]+Kb1[16][16]+Kb2[16][16]+Kb3[16][16];

  //Shear Component of strain displacement
  defaulttype::Mat<20, 32, Real> Js_t;
  StrainDisplacement Js;
  computeShearStrainDisplacement(Js, length, height /*p[idx0], p[idx1], p[idx2], p[idx3]*/);
  Js_t.transpose(Js);
  // Shear component of material stiffness
  MaterialStiffness Cs;
  Cs = quadInf[elementIndex].ShearmaterialMatrix ;
  /*// expand Cs from 8x8 to 32x32 diagonal matrix
  defaulttype::Mat<32,32,Real> Cs_e;
  for (unsigned i = 0;i<4;i++)
  {
    for(unsigned j = 0;j<4;j++)
    {
      for(unsigned k = 0;k<8;k++)
      {
        for(unsigned l = 0;l<8;l++)
        {
          Cs_e[8*i+k][8*j+l] = Cs[k][l];
        }
      }
    }
  }*/
  // Stiffness matrix for bending component
  const float ws = 2.0f; // weight coff of gauss integration 1x1
  Stiffness Ks;
  //Ks = length*height*(Js_t*Cs_e*Js)*ws*ws;
  Ks[2][2]=length*height*(Js_t[2][6]*Cs[6][6]*Js[6][2]+Js_t[2][7]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[2][7]=length*height*(Js_t[2][6]*Cs[6][6]*Js[14][7]+Js_t[2][7]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[2][12]=length*height*(Js_t[2][6]*Cs[6][6]*Js[22][12]+Js_t[2][7]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[2][17]=length*height*(Js_t[2][6]*Cs[6][6]*Js[30][17]+Js_t[2][7]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[2][3]=length*height*(Js_t[2][6]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[2][8]=length*height*(Js_t[2][6]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[2][13]=length*height*(Js_t[2][6]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[2][18]=length*height*(Js_t[2][6]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[2][4]=length*height*(Js_t[2][7]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[2][9]=length*height*(Js_t[2][7]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[2][14]=length*height*(Js_t[2][7]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[2][19]=length*height*(Js_t[2][7]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[3][2]=length*height*(Js_t[3][6]*Cs[6][6]*Js[6][2])*ws*ws;
  Ks[3][7]=length*height*(Js_t[3][6]*Cs[6][6]*Js[14][7])*ws*ws;
  Ks[3][12]=length*height*(Js_t[3][6]*Cs[6][6]*Js[22][12])*ws*ws;
  Ks[3][17]=length*height*(Js_t[3][6]*Cs[6][6]*Js[30][17])*ws*ws;
  
  Ks[3][3]=length*height*(Js_t[3][3]*Cs[3][3]*Js[3][3]+Js_t[3][5]*Cs[5][5]*Js[5][3]+Js_t[3][6]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[3][8]=length*height*(Js_t[3][3]*Cs[3][3]*Js[11][8]+Js_t[3][5]*Cs[5][5]*Js[13][8]+Js_t[3][6]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[3][13]=length*height*(Js_t[3][3]*Cs[3][3]*Js[19][13]+Js_t[3][5]*Cs[5][5]*Js[21][13]+Js_t[3][6]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[3][18]=length*height*(Js_t[3][3]*Cs[3][3]*Js[27][18]+Js_t[3][5]*Cs[5][5]*Js[29][18]+Js_t[3][6]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[3][4]=length*height*(Js_t[3][3]*Cs[3][4]*Js[4][4]+Js_t[3][5]*Cs[5][5]*Js[5][4])*ws*ws;
  Ks[3][9]=length*height*(Js_t[3][3]*Cs[3][4]*Js[12][9]+Js_t[3][5]*Cs[5][5]*Js[13][9])*ws*ws;
  Ks[3][14]=length*height*(Js_t[3][3]*Cs[3][4]*Js[20][14]+Js_t[3][5]*Cs[5][5]*Js[21][14])*ws*ws;
  Ks[3][19]=length*height*(Js_t[3][3]*Cs[3][4]*Js[28][19]+Js_t[3][5]*Cs[5][5]*Js[29][19])*ws*ws;

  Ks[4][2]=length*height*(Js_t[4][7]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[4][7]=length*height*(Js_t[4][7]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[4][12]=length*height*(Js_t[4][7]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[4][17]=length*height*(Js_t[4][7]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[4][3]=length*height*(Js_t[4][4]*Cs[4][3]*Js[3][3]+Js_t[4][5]*Cs[5][5]*Js[5][3])*ws*ws;
  Ks[4][8]=length*height*(Js_t[4][4]*Cs[4][3]*Js[11][8]+Js_t[4][5]*Cs[5][5]*Js[13][8])*ws*ws;
  Ks[4][13]=length*height*(Js_t[4][4]*Cs[4][3]*Js[19][13]+Js_t[4][5]*Cs[5][5]*Js[21][13])*ws*ws;
  Ks[4][18]=length*height*(Js_t[4][4]*Cs[4][3]*Js[27][18]+Js_t[4][5]*Cs[5][5]*Js[29][18])*ws*ws;

  Ks[4][4]=length*height*(Js_t[4][4]*Cs[4][4]*Js[4][4]+Js_t[4][5]*Cs[5][5]*Js[5][4]+Js_t[4][7]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[4][9]=length*height*(Js_t[4][4]*Cs[4][4]*Js[12][9]+Js_t[4][5]*Cs[5][5]*Js[13][9]+Js_t[4][7]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[4][14]=length*height*(Js_t[4][4]*Cs[4][4]*Js[20][14]+Js_t[4][5]*Cs[5][5]*Js[21][14]+Js_t[4][7]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[4][19]=length*height*(Js_t[4][4]*Cs[4][4]*Js[28][19]+Js_t[4][5]*Cs[5][5]*Js[29][19]+Js_t[4][7]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[7][2]=length*height*(Js_t[7][14]*Cs[6][6]*Js[6][2]+Js_t[7][15]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[7][7]=length*height*(Js_t[7][14]*Cs[6][6]*Js[14][7]+Js_t[7][15]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[7][12]=length*height*(Js_t[7][14]*Cs[6][6]*Js[22][12]+Js_t[7][15]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[7][17]=length*height*(Js_t[7][14]*Cs[6][6]*Js[30][17]+Js_t[7][15]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[7][3]=length*height*(Js_t[7][14]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[7][8]=length*height*(Js_t[7][14]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[7][13]=length*height*(Js_t[7][14]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[7][18]=length*height*(Js_t[7][14]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[7][4]=length*height*(Js_t[7][15]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[7][9]=length*height*(Js_t[7][15]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[7][14]=length*height*(Js_t[7][15]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[7][19]=length*height*(Js_t[7][15]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[8][2]=length*height*(Js_t[8][14]*Cs[6][6]*Js[6][2])*ws*ws;
  Ks[8][7]=length*height*(Js_t[8][14]*Cs[6][6]*Js[14][7])*ws*ws;
  Ks[8][12]=length*height*(Js_t[8][14]*Cs[6][6]*Js[22][12])*ws*ws;
  Ks[8][17]=length*height*(Js_t[8][14]*Cs[6][6]*Js[30][17])*ws*ws;
  
  Ks[8][3]=length*height*(Js_t[8][11]*Cs[3][3]*Js[3][3]+Js_t[8][13]*Cs[5][5]*Js[5][3]+Js_t[8][14]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[8][8]=length*height*(Js_t[8][11]*Cs[3][3]*Js[11][8]+Js_t[8][13]*Cs[5][5]*Js[13][8]+Js_t[8][14]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[8][13]=length*height*(Js_t[8][11]*Cs[3][3]*Js[19][13]+Js_t[8][13]*Cs[5][5]*Js[21][13]+Js_t[8][14]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[8][18]=length*height*(Js_t[8][11]*Cs[3][3]*Js[27][18]+Js_t[8][13]*Cs[5][5]*Js[29][18]+Js_t[8][14]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[8][4]=length*height*(Js_t[8][11]*Cs[3][4]*Js[4][4]+Js_t[8][13]*Cs[5][5]*Js[5][4])*ws*ws;
  Ks[8][9]=length*height*(Js_t[8][11]*Cs[3][4]*Js[12][9]+Js_t[8][13]*Cs[5][5]*Js[13][9])*ws*ws;
  Ks[8][14]=length*height*(Js_t[8][11]*Cs[3][4]*Js[20][14]+Js_t[8][13]*Cs[5][5]*Js[21][14])*ws*ws;
  Ks[8][19]=length*height*(Js_t[8][11]*Cs[3][4]*Js[28][19]+Js_t[8][13]*Cs[5][5]*Js[29][19])*ws*ws;

  Ks[9][2]=length*height*(Js_t[9][15]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[9][7]=length*height*(Js_t[9][15]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[9][12]=length*height*(Js_t[9][15]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[9][17]=length*height*(Js_t[9][15]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[9][3]=length*height*(Js_t[9][12]*Cs[4][3]*Js[3][3]+Js_t[9][13]*Cs[5][5]*Js[5][3])*ws*ws;
  Ks[9][8]=length*height*(Js_t[9][12]*Cs[4][3]*Js[11][8]+Js_t[9][13]*Cs[5][5]*Js[13][8])*ws*ws;
  Ks[9][13]=length*height*(Js_t[9][12]*Cs[4][3]*Js[19][13]+Js_t[9][13]*Cs[5][5]*Js[21][13])*ws*ws;
  Ks[9][18]=length*height*(Js_t[9][12]*Cs[4][3]*Js[27][18]+Js_t[9][13]*Cs[5][5]*Js[29][18])*ws*ws;

  Ks[9][4]=length*height*(Js_t[9][12]*Cs[4][4]*Js[4][4]+Js_t[9][13]*Cs[5][5]*Js[5][4]+Js_t[9][15]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[9][9]=length*height*(Js_t[9][12]*Cs[4][4]*Js[12][9]+Js_t[9][13]*Cs[5][5]*Js[13][9]+Js_t[9][15]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[9][14]=length*height*(Js_t[9][12]*Cs[4][4]*Js[20][14]+Js_t[9][13]*Cs[5][5]*Js[21][14]+Js_t[9][15]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[9][19]=length*height*(Js_t[9][12]*Cs[4][4]*Js[28][19]+Js_t[9][13]*Cs[5][5]*Js[29][19]+Js_t[9][15]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[12][2]=length*height*(Js_t[12][22]*Cs[6][6]*Js[6][2]+Js_t[12][23]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[12][7]=length*height*(Js_t[12][22]*Cs[6][6]*Js[14][7]+Js_t[12][23]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[12][12]=length*height*(Js_t[12][22]*Cs[6][6]*Js[22][12]+Js_t[12][23]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[12][17]=length*height*(Js_t[12][22]*Cs[6][6]*Js[30][17]+Js_t[12][23]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[12][3]=length*height*(Js_t[12][22]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[12][8]=length*height*(Js_t[12][22]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[12][13]=length*height*(Js_t[12][22]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[12][18]=length*height*(Js_t[12][22]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[12][4]=length*height*(Js_t[12][23]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[12][9]=length*height*(Js_t[12][23]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[12][14]=length*height*(Js_t[12][23]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[12][19]=length*height*(Js_t[12][23]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[13][2]=length*height*(Js_t[13][22]*Cs[6][6]*Js[6][2])*ws*ws;
  Ks[13][7]=length*height*(Js_t[13][22]*Cs[6][6]*Js[14][7])*ws*ws;
  Ks[13][12]=length*height*(Js_t[13][22]*Cs[6][6]*Js[22][12])*ws*ws;
  Ks[13][17]=length*height*(Js_t[13][22]*Cs[6][6]*Js[30][17])*ws*ws;
 
  Ks[13][3]=length*height*(Js_t[13][19]*Cs[3][3]*Js[3][3]+Js_t[13][21]*Cs[5][5]*Js[5][3]+Js_t[13][22]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[13][8]=length*height*(Js_t[13][19]*Cs[3][3]*Js[11][8]+Js_t[13][21]*Cs[5][5]*Js[13][8]+Js_t[13][22]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[13][13]=length*height*(Js_t[13][19]*Cs[3][3]*Js[19][13]+Js_t[13][21]*Cs[5][5]*Js[21][13]+Js_t[13][22]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[13][18]=length*height*(Js_t[13][19]*Cs[3][3]*Js[27][18]+Js_t[13][21]*Cs[5][5]*Js[29][18]+Js_t[13][22]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[13][4]=length*height*(Js_t[13][19]*Cs[3][4]*Js[4][4]+Js_t[13][21]*Cs[5][5]*Js[5][4])*ws*ws;
  Ks[13][9]=length*height*(Js_t[13][19]*Cs[3][4]*Js[12][9]+Js_t[13][21]*Cs[5][5]*Js[13][9])*ws*ws;
  Ks[13][14]=length*height*(Js_t[13][19]*Cs[3][4]*Js[20][14]+Js_t[13][21]*Cs[5][5]*Js[21][14])*ws*ws;
  Ks[13][19]=length*height*(Js_t[13][19]*Cs[3][4]*Js[28][19]+Js_t[13][21]*Cs[5][5]*Js[29][19])*ws*ws;

  Ks[14][2]=length*height*(Js_t[14][23]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[14][7]=length*height*(Js_t[14][23]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[14][12]=length*height*(Js_t[14][23]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[14][17]=length*height*(Js_t[14][23]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[14][3]=length*height*(Js_t[14][20]*Cs[4][3]*Js[3][3]+Js_t[14][21]*Cs[5][5]*Js[5][3])*ws*ws;
  Ks[14][8]=length*height*(Js_t[14][20]*Cs[4][3]*Js[11][8]+Js_t[14][21]*Cs[5][5]*Js[13][8])*ws*ws;
  Ks[14][13]=length*height*(Js_t[14][20]*Cs[4][3]*Js[19][13]+Js_t[14][21]*Cs[5][5]*Js[21][13])*ws*ws;
  Ks[14][18]=length*height*(Js_t[14][20]*Cs[4][3]*Js[27][18]+Js_t[14][21]*Cs[5][5]*Js[29][18])*ws*ws;

  Ks[14][4]=length*height*(Js_t[14][20]*Cs[4][4]*Js[4][4]+Js_t[14][21]*Cs[5][5]*Js[5][4]+Js_t[14][23]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[14][9]=length*height*(Js_t[14][20]*Cs[4][4]*Js[12][9]+Js_t[14][21]*Cs[5][5]*Js[13][9]+Js_t[14][23]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[14][14]=length*height*(Js_t[14][20]*Cs[4][4]*Js[20][14]+Js_t[14][21]*Cs[5][5]*Js[21][14]+Js_t[14][23]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[14][19]=length*height*(Js_t[14][20]*Cs[4][4]*Js[28][19]+Js_t[14][21]*Cs[5][5]*Js[29][19]+Js_t[14][23]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[17][2]=length*height*(Js_t[17][30]*Cs[6][6]*Js[6][2]+Js_t[17][31]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[17][7]=length*height*(Js_t[17][30]*Cs[6][6]*Js[14][7]+Js_t[17][31]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[17][12]=length*height*(Js_t[17][30]*Cs[6][6]*Js[22][12]+Js_t[17][31]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[17][17]=length*height*(Js_t[17][30]*Cs[6][6]*Js[30][17]+Js_t[17][31]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[17][3]=length*height*(Js_t[17][30]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[17][8]=length*height*(Js_t[17][30]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[17][13]=length*height*(Js_t[17][30]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[17][18]=length*height*(Js_t[17][30]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[17][4]=length*height*(Js_t[17][31]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[17][9]=length*height*(Js_t[17][31]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[17][14]=length*height*(Js_t[17][31]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[17][19]=length*height*(Js_t[17][31]*Cs[7][7]*Js[31][19])*ws*ws;

  Ks[18][2]=length*height*(Js_t[18][30]*Cs[6][6]*Js[6][2])*ws*ws;
  Ks[18][7]=length*height*(Js_t[18][30]*Cs[6][6]*Js[14][7])*ws*ws;
  Ks[18][12]=length*height*(Js_t[18][30]*Cs[6][6]*Js[22][12])*ws*ws;
  Ks[18][17]=length*height*(Js_t[18][30]*Cs[6][6]*Js[30][17])*ws*ws;
 
  Ks[18][3]=length*height*(Js_t[18][27]*Cs[3][3]*Js[3][3]+Js_t[18][29]*Cs[5][5]*Js[5][3]+Js_t[18][30]*Cs[6][6]*Js[6][3])*ws*ws;
  Ks[18][8]=length*height*(Js_t[18][27]*Cs[3][3]*Js[11][8]+Js_t[18][29]*Cs[5][5]*Js[13][8]+Js_t[18][30]*Cs[6][6]*Js[14][8])*ws*ws;
  Ks[18][13]=length*height*(Js_t[18][27]*Cs[3][3]*Js[19][13]+Js_t[18][29]*Cs[5][5]*Js[21][13]+Js_t[18][30]*Cs[6][6]*Js[22][13])*ws*ws;
  Ks[18][18]=length*height*(Js_t[18][27]*Cs[3][3]*Js[27][18]+Js_t[18][29]*Cs[5][5]*Js[29][18]+Js_t[18][30]*Cs[6][6]*Js[30][18])*ws*ws;

  Ks[18][4]=length*height*(Js_t[18][27]*Cs[3][4]*Js[4][4]+Js_t[18][29]*Cs[5][5]*Js[5][4])*ws*ws;
  Ks[18][9]=length*height*(Js_t[18][27]*Cs[3][4]*Js[12][9]+Js_t[18][29]*Cs[5][5]*Js[13][9])*ws*ws;
  Ks[18][14]=length*height*(Js_t[18][27]*Cs[3][4]*Js[20][14]+Js_t[18][29]*Cs[5][5]*Js[21][14])*ws*ws;
  Ks[18][19]=length*height*(Js_t[18][27]*Cs[3][4]*Js[28][19]+Js_t[18][29]*Cs[5][5]*Js[29][19])*ws*ws;

  Ks[19][2]=length*height*(Js_t[19][31]*Cs[7][7]*Js[7][2])*ws*ws;
  Ks[19][7]=length*height*(Js_t[19][31]*Cs[7][7]*Js[15][7])*ws*ws;
  Ks[19][12]=length*height*(Js_t[19][31]*Cs[7][7]*Js[23][12])*ws*ws;
  Ks[19][17]=length*height*(Js_t[19][31]*Cs[7][7]*Js[31][17])*ws*ws;

  Ks[19][3]=length*height*(Js_t[19][28]*Cs[4][3]*Js[3][3]+Js_t[19][29]*Cs[5][5]*Js[5][3])*ws*ws;
  Ks[19][8]=length*height*(Js_t[19][28]*Cs[4][3]*Js[11][8]+Js_t[19][29]*Cs[5][5]*Js[13][8])*ws*ws;
  Ks[19][13]=length*height*(Js_t[19][28]*Cs[4][3]*Js[19][13]+Js_t[19][29]*Cs[5][5]*Js[21][13])*ws*ws;
  Ks[19][18]=length*height*(Js_t[19][28]*Cs[4][3]*Js[27][18]+Js_t[19][29]*Cs[5][5]*Js[29][18])*ws*ws;

  Ks[19][4]=length*height*(Js_t[19][28]*Cs[4][4]*Js[4][4]+Js_t[19][29]*Cs[5][5]*Js[5][4]+Js_t[19][31]*Cs[7][7]*Js[7][4])*ws*ws;
  Ks[19][9]=length*height*(Js_t[19][28]*Cs[4][4]*Js[12][9]+Js_t[19][29]*Cs[5][5]*Js[13][9]+Js_t[19][31]*Cs[7][7]*Js[15][9])*ws*ws;
  Ks[19][14]=length*height*(Js_t[19][28]*Cs[4][4]*Js[20][14]+Js_t[19][29]*Cs[5][5]*Js[21][14]+Js_t[19][31]*Cs[7][7]*Js[23][14])*ws*ws;
  Ks[19][19]=length*height*(Js_t[19][28]*Cs[4][4]*Js[28][19]+Js_t[19][29]*Cs[5][5]*Js[29][19]+Js_t[19][31]*Cs[7][7]*Js[31][19])*ws*ws;



  // Stiffness matrix of a element: K = Kb + Ks
  K = Kb + Ks;
  // luu cac gia tri stiffness
  quadInf[elementIndex].Bendingstiffness=Kb;
  quadInf[elementIndex].Shearstiffness=Ks;
  quadInfo.endEdit();

 int nbQuads = m_topology->getNbQuads();
    //count number line of file 
    int numLines = 0;
    std::ifstream in("/home/nhnhanbk/Desktop/Sofa/sofa/nhnhan/IoTouch/K.txt");
    std::string unused;
    while (std::getline(in, unused))
        ++numLines;
    in.close();

    if (numLines == nbQuads) {std::ofstream myfile("/home/nhnhanbk/Desktop/Sofa/sofa/nhnhan/IoTouch/K.txt", std::ios_base::app);
				myfile.close();
				}
	else
    {
        std::ofstream myfile("/home/nhnhanbk/Desktop/Sofa/sofa/nhnhan/IoTouch/K.txt", std::ios_base::app);
        myfile.flush();
        myfile <<K<<"\n";
        myfile.close();
    }

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
    quadInf[elementIndex].stiffness = K; // co the la da duoc luu roi` nen ko can`

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
    // Tinh displacement cua cac node trong global frame
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
  Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue()); 

  Real h=1;
  df1.resize(dx1.size());
  applyStiffnessSmall( df1, h, dx1, kFactor );

  df.endEdit();
}




  
} // namespace forcefield

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_FORCEFIELD_QUADBENDINGFEMFORCEFIELD_H
  
  
  
  
  
  

