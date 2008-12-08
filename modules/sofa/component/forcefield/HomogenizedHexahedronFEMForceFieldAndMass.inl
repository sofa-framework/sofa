/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_FORCEFIELD_HomogenizedHEXAHEDRONFEMFORCEFIELD_INL
#define SOFA_COMPONENT_FORCEFIELD_HomogenizedHEXAHEDRONFEMFORCEFIELD_INL

#include <sofa/component/forcefield/HomogenizedHexahedronFEMForceFieldAndMass.h>


#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/topology/SparseGridMultipleTopology.h>
// #include <sofa/simulation/tree/GNode.h>

// #include <sofa/simulation/common/InitVisitor.h>
// #include <sofa/simulation/common/UpdateContextVisitor.h>

// #include <sofa/core/Mapping.h>
// #include <sofa/core/componentmodel/behavior/MappedModel.h>
// #include <sofa/core/componentmodel/behavior/State.h>
// #include <sofa/component/visualmodel/VisualModelImpl.h>

#include <iomanip>

#include <sofa/helper/gl/BasicShapes.h>


using std::cerr;
using std::endl;
using std::set;


// using sofa::simulation::tree::GNode;


namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

using topology::SparseGridTopology;
using namespace linearsolver;

// 	  template <class DataTypes>
// 	  const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::IndicesOf_FineHexa_FineNode_forXYZAssemblingOrder[27][2]=
// 	  {
// 		  { 0  , 0 }, // tranche z=0
// 		  { 0 , 1 },
// 		  { 1 , 1 },
// 		  {  2,0  },
// 		  {  2, 1 },
// 		  {  3, 1 },
// 		  {  2, 3 },
// 		  {  2, 2 },
// 		  {  3, 2 },
//
// 		  { 0  , 4 }, // z=1
// 		  { 0 , 5 },
// 		  { 1 , 5 },
// 		  {  2,4  },
// 		  {  2, 5 },
// 		  {  3, 5 },
// 		  {  2, 7 },
// 		  {  2, 6 },
// 		  {  3, 6 },
//
// 		  { 4  , 4 }, // z=2
// 		  { 4 , 5 },
// 		  { 5 , 5 },
// 		  {  6,4  },
// 		  {  6, 5 },
// 		  {  7, 5 },
// 		  {  6, 7 },
// 		  {  6, 6 },
// 		  {  7, 6 }
// 	  };



template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::FineHexa_FineNode_IndiceForAssembling[8][8]=
{
    // for an (fine elem, fine vertex) given -> what position in the assembled matrix
    {  0,  1,  4, 3,  9, 10,  13, 12},
    {  1, 2, 5, 4, 10, 11, 14, 13},
    {  3, 4,  7, 6,  12, 13,  16, 15},
    { 4, 5, 8, 7, 13, 14, 17, 16},
    {  9, 10,  13, 12,  18, 19,  22, 21},
    { 10, 11, 14, 13, 19, 20, 23, 22},
    {  12, 13,  16, 15,  21, 22,  25, 24},
    { 13, 14, 17, 16, 22, 23, 26, 25}
};


/*
	template <class DataTypes>
	const bool HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::IS_CONSTRAINED[8][8]=
	{
      // for an (fine elem, fine vertex) given -> what position in the assembled matrix
		{  1, 0, 0, 0, 0, 0, 0, 0},
		{  0, 1, 0, 0, 0, 0, 0, 0},
		{  0, 0, 0, 1, 0, 0, 0, 0},
		{  0, 0, 1, 0, 0, 0, 0, 0},
		{  0, 0, 0, 0, 1, 0, 0, 0},
		{  0, 0, 0, 0, 0, 1, 0, 0},
		{  0, 0, 0, 0, 0, 0, 0, 1},
		{  0, 0, 0, 0, 0, 0, 1, 0},
	};





// 		{0,	1,	2,	3,	4,	5,	6,	7,	8,	9,	10,	11,	 12,13,	14,	15,	16,	17,	18,	 19,	20,	21,	22,	23,	24,	25,	26},
// 		{0,	0,	1,	1,	2,	3,	3,	4,	2,	5,	6,	7	,8,	9,	10,	11,	12,	13,	4	,14,	5,	15,	16,	17,	7,	18,	6}

	template <class DataTypes>
	const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::FineHexa_FineNode_IndiceForCutAssembling[8][8]=
	{
      // for an (fine elem, fine vertex) given -> what position in the assembled matrix
		{  0,  0,  2, 1,  5, 6,  9, 8},
		{  0, 1, 3, 2, 6, 7, 10, 9},
		{  1, 2,  4, 3,  8, 9,  12, 11},
		{  2, 3, 2, 4, 9, 10, 13, 12},
		{  5, 6,  9, 8 ,  4, 14,  16, 15},
		{  6, 7, 10, 9, 14, 5, 17, 16},
		{  8, 9,  12, 11,  15, 16,  18, 7},
		{  9, 10, 13, 12, 16, 17, 6, 18}
	};
	*/



template <class DataTypes>
const bool HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::IS_CONSTRAINED_27[27] =
{
    1,0,1, 0,0,0, 1,0,1, //tranche devant
    0,0,0, 0,0,0, 0,0,0,  //milieu
    1,0,1, 0,0,0, 1,0,1
};


template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::FineHexa_FineNode_IndiceForCutAssembling_27[27]=
{0,	0,	1,	1,	2,	3,	3,	4,	2,	5,	6,	7	,8,	9,	10,	11,	12,	13,	4	,14,	5,	15,	16,	17,	7,	18,	6};



template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::CoarseToFine[8]=
{ 0, 2, 8, 6, 18, 20, 26, 24 };



template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK[27*3][8*3]=
{
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0},
    {0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0},
    {0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0},
    {0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0},
    {0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1},
    {1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0},
    {0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0},
    {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1},
    {0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1},
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}
};

template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK_CROSSED[27*3][8*3]=
{
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-1,1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-1,0,1,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,-1,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,-1,1,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,-1,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,-1,1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,-1,1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,-1,0,1,-1,0,1,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,-1,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0},
    {0,1,-1,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0},
    {0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
    {1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0},
    {-1,1,-1,-1,1,-1,0,0,0,0,0,0,-1,1,-1,-1,1,-1,0,0,0,0,0,0},
    {0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0},
    {0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0},
    {0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0},
    {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {1,-1,-1,0,0,0,0,0,0,1,-1,-1,1,-1,-1,0,0,0,0,0,0,1,-1,-1},
    {0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0},
    {0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1},
    {1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0},
    {0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0},
    {0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1},
    {0,0,0,1,-1,-1,1,-1,-1,0,0,0,0,0,0,1,-1,-1,1,-1,-1,0,0,0},
    {0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,1,0,-1},
    {0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,1,-1},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,-1,1,-1,-1,1,-1,0,0,0,0,0,0,-1,1,-1,-1,1,-1},
    {0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1},
    {0,0,0,0,0,0,1,0,-1,0,0,0,0,0,0,0,0,0,1,0,-1,0,0,0},
    {0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0},
    {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,-1,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-1,0,1,-1,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,1,-1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,-1,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,1,-1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,-1,1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,-1,1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,1,-1,0,1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}
};

template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK_CROSSED_DIFF[27*3][8*3]=
{
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,-2,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,-2,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-1,-1,0,-1,-1,0,-1,-1,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0},
    {0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {-1,0,-1,-1,0,-1,0,0,0,0,0,0,-1,0,-1,-1,0,-1,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0},
    {0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,-1,-1,0,0,0,0,0,0,0,-1,-1,0,-1,-1,0,0,0,0,0,0,0,-1,-1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,-1,-1,0,-1,-1,0,0,0,0,0,0,0,-1,-1,0,-1,-1,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2},
    {0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,-1,0,-1,-1,0,-1,0,0,0,0,0,0,-1,0,-1,-1,0,-1},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0},
    {0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,-2,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,0,0,0,0,0,0,-2,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,0,-1,-1,0,-1,-1,0,-1,-1,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-2,0,0,-2,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
};



template <class DataTypes>
const float HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::MIDDLE_INTERPOLATION[27][8]=
{
    {1,0,0,0,0,0,0,0},
    {0.5,0.5,0,0,0,0,0,0},
    {0,1,0,0,0,0,0,0},
    {0.5,0,0,0.5,0,0,0,0},
    {0.25,0.25,0.25,0.25,0,0,0,0},
    {0,0.5,0.5,0,0,0,0,0},
    {0,0,0,1,0,0,0,0},
    {0,0,0.5,0.5,0,0,0,0},
    {0,0,1,0,0,0,0,0},
    {0.5,0,0,0,0.5,0,0,0},
    {0.25,0.25,0,0,0.25,0.25,0,0},
    {0,0.5,0,0,0,0.5,0,0},
    {0.25,0,0,0.25,0.25,0,0,0.25},
    {0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125},
    {0,0.25,0.25,0,0,0.25,0.25,0},
    {0,0,0,0.5,0,0,0,0.5},
    {0,0,0.25,0.25,0,0,0.25,0.25},
    {0,0,0.5,0,0,0,0.5,0},
    {0,0,0,0,1,0,0,0},
    {0,0,0,0,0.5,0.5,0,0},
    {0,0,0,0,0,1,0,0},
    {0,0,0,0,0.5,0,0,0.5},
    {0,0,0,0,0.25,0.25,0.25,0.25},
    {0,0,0,0,0,0.5,0.5,0},
    {0,0,0,0,0,0,0,1},
    {0,0,0,0,0,0,0.5,0.5},
    {0,0,0,0,0,0,1,0}
};



// linked with MIDDLE_INTERPOLATION: in which axes do we want the interpolatio? (0->all, 1->x, 2->y, 3->z)
template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::MIDDLE_AXES[27]=
{
    0,   	1,          	0,
    2,  		3,		        2,
    0,		1,				0,

    3,			2,		3,
    1,	0,			 1,
    3,			2,		3,

    0,   	1,          	0,
    2,  		3,		2,
    0,   	1,          	0
};



template <class DataTypes>
const int HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::FINE_ELEM_IN_COARSE_IN_ASS_FRAME[8][8]=
{
    {0,1,4,3,9,10,13,12},
    {1,2,5,4,10,11,14,13},
    {3,4,7,6,12,13,16,15},
    {4,5,8,7,13,14,17,16},
    {9,10,13,12,18,19,22,23},
    {10,11,14,13,19,20,23,22},
    {12,13,16,15,21,22,25,24},
    {13,14,17,16,22,23,26,25}
};


template <class DataTypes>
const float HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::RIGID_STIFFNESS[8*3][8*3]=
{{2.26667e+11,4.25e+10,4.25e+10,-5.66667e+10,-4.25e+10,-4.25e+10,-7.08333e+10,-4.25e+10,-2.125e+10,2.83333e+10,4.25e+10,2.125e+10,2.83333e+10,2.125e+10,4.25e+10,-7.08333e+10,-2.125e+10,-4.25e+10,-5.66667e+10,-2.125e+10,-2.125e+10,-2.83333e+10,2.125e+10,2.125e+10},{4.25e+10,2.26667e+11,4.25e+10,4.25e+10,2.83333e+10,2.125e+10,-4.25e+10,-7.08333e+10,-2.125e+10,-4.25e+10,-5.66667e+10,-4.25e+10,2.125e+10,2.83333e+10,4.25e+10,2.125e+10,-2.83333e+10,2.125e+10,-2.125e+10,-5.66667e+10,-2.125e+10,-2.125e+10,-7.08333e+10,-4.25e+10},{4.25e+10,4.25e+10,2.26667e+11,4.25e+10,2.125e+10,2.83333e+10,2.125e+10,2.125e+10,-2.83333e+10,2.125e+10,4.25e+10,2.83333e+10,-4.25e+10,-4.25e+10,-5.66667e+10,-4.25e+10,-2.125e+10,-7.08333e+10,-2.125e+10,-2.125e+10,-5.66667e+10,-2.125e+10,-4.25e+10,-7.08333e+10},{-5.66667e+10,4.25e+10,4.25e+10,2.26667e+11,-4.25e+10,-4.25e+10,2.83333e+10,-4.25e+10,-2.125e+10,-7.08333e+10,4.25e+10,2.125e+10,-7.08333e+10,2.125e+10,4.25e+10,2.83333e+10,-2.125e+10,-4.25e+10,-2.83333e+10,-2.125e+10,-2.125e+10,-5.66667e+10,2.125e+10,2.125e+10},{-4.25e+10,2.83333e+10,2.125e+10,-4.25e+10,2.26667e+11,4.25e+10,4.25e+10,-5.66667e+10,-4.25e+10,4.25e+10,-7.08333e+10,-2.125e+10,-2.125e+10,-2.83333e+10,2.125e+10,-2.125e+10,2.83333e+10,4.25e+10,2.125e+10,-7.08333e+10,-4.25e+10,2.125e+10,-5.66667e+10,-2.125e+10},{-4.25e+10,2.125e+10,2.83333e+10,-4.25e+10,4.25e+10,2.26667e+11,-2.125e+10,4.25e+10,2.83333e+10,-2.125e+10,2.125e+10,-2.83333e+10,4.25e+10,-2.125e+10,-7.08333e+10,4.25e+10,-4.25e+10,-5.66667e+10,2.125e+10,-4.25e+10,-7.08333e+10,2.125e+10,-2.125e+10,-5.66667e+10},{-7.08333e+10,-4.25e+10,2.125e+10,2.83333e+10,4.25e+10,-2.125e+10,2.26667e+11,4.25e+10,-4.25e+10,-5.66667e+10,-4.25e+10,4.25e+10,-5.66667e+10,-2.125e+10,2.125e+10,-2.83333e+10,2.125e+10,-2.125e+10,2.83333e+10,2.125e+10,-4.25e+10,-7.08333e+10,-2.125e+10,4.25e+10},{-4.25e+10,-7.08333e+10,2.125e+10,-4.25e+10,-5.66667e+10,4.25e+10,4.25e+10,2.26667e+11,-4.25e+10,4.25e+10,2.83333e+10,-2.125e+10,-2.125e+10,-5.66667e+10,2.125e+10,-2.125e+10,-7.08333e+10,4.25e+10,2.125e+10,2.83333e+10,-4.25e+10,2.125e+10,-2.83333e+10,-2.125e+10},{-2.125e+10,-2.125e+10,-2.83333e+10,-2.125e+10,-4.25e+10,2.83333e+10,-4.25e+10,-4.25e+10,2.26667e+11,-4.25e+10,-2.125e+10,2.83333e+10,2.125e+10,2.125e+10,-5.66667e+10,2.125e+10,4.25e+10,-7.08333e+10,4.25e+10,4.25e+10,-5.66667e+10,4.25e+10,2.125e+10,-7.08333e+10},{2.83333e+10,-4.25e+10,2.125e+10,-7.08333e+10,4.25e+10,-2.125e+10,-5.66667e+10,4.25e+10,-4.25e+10,2.26667e+11,-4.25e+10,4.25e+10,-2.83333e+10,-2.125e+10,2.125e+10,-5.66667e+10,2.125e+10,-2.125e+10,-7.08333e+10,2.125e+10,-4.25e+10,2.83333e+10,-2.125e+10,4.25e+10},{4.25e+10,-5.66667e+10,4.25e+10,4.25e+10,-7.08333e+10,2.125e+10,-4.25e+10,2.83333e+10,-2.125e+10,-4.25e+10,2.26667e+11,-4.25e+10,2.125e+10,-7.08333e+10,4.25e+10,2.125e+10,-5.66667e+10,2.125e+10,-2.125e+10,-2.83333e+10,-2.125e+10,-2.125e+10,2.83333e+10,-4.25e+10},{2.125e+10,-4.25e+10,2.83333e+10,2.125e+10,-2.125e+10,-2.83333e+10,4.25e+10,-2.125e+10,2.83333e+10,4.25e+10,-4.25e+10,2.26667e+11,-2.125e+10,4.25e+10,-7.08333e+10,-2.125e+10,2.125e+10,-5.66667e+10,-4.25e+10,2.125e+10,-7.08333e+10,-4.25e+10,4.25e+10,-5.66667e+10},{2.83333e+10,2.125e+10,-4.25e+10,-7.08333e+10,-2.125e+10,4.25e+10,-5.66667e+10,-2.125e+10,2.125e+10,-2.83333e+10,2.125e+10,-2.125e+10,2.26667e+11,4.25e+10,-4.25e+10,-5.66667e+10,-4.25e+10,4.25e+10,-7.08333e+10,-4.25e+10,2.125e+10,2.83333e+10,4.25e+10,-2.125e+10},{2.125e+10,2.83333e+10,-4.25e+10,2.125e+10,-2.83333e+10,-2.125e+10,-2.125e+10,-5.66667e+10,2.125e+10,-2.125e+10,-7.08333e+10,4.25e+10,4.25e+10,2.26667e+11,-4.25e+10,4.25e+10,2.83333e+10,-2.125e+10,-4.25e+10,-7.08333e+10,2.125e+10,-4.25e+10,-5.66667e+10,4.25e+10},{4.25e+10,4.25e+10,-5.66667e+10,4.25e+10,2.125e+10,-7.08333e+10,2.125e+10,2.125e+10,-5.66667e+10,2.125e+10,4.25e+10,-7.08333e+10,-4.25e+10,-4.25e+10,2.26667e+11,-4.25e+10,-2.125e+10,2.83333e+10,-2.125e+10,-2.125e+10,-2.83333e+10,-2.125e+10,-4.25e+10,2.83333e+10},{-7.08333e+10,2.125e+10,-4.25e+10,2.83333e+10,-2.125e+10,4.25e+10,-2.83333e+10,-2.125e+10,2.125e+10,-5.66667e+10,2.125e+10,-2.125e+10,-5.66667e+10,4.25e+10,-4.25e+10,2.26667e+11,-4.25e+10,4.25e+10,2.83333e+10,-4.25e+10,2.125e+10,-7.08333e+10,4.25e+10,-2.125e+10},{-2.125e+10,-2.83333e+10,-2.125e+10,-2.125e+10,2.83333e+10,-4.25e+10,2.125e+10,-7.08333e+10,4.25e+10,2.125e+10,-5.66667e+10,2.125e+10,-4.25e+10,2.83333e+10,-2.125e+10,-4.25e+10,2.26667e+11,-4.25e+10,4.25e+10,-5.66667e+10,4.25e+10,4.25e+10,-7.08333e+10,2.125e+10},{-4.25e+10,2.125e+10,-7.08333e+10,-4.25e+10,4.25e+10,-5.66667e+10,-2.125e+10,4.25e+10,-7.08333e+10,-2.125e+10,2.125e+10,-5.66667e+10,4.25e+10,-2.125e+10,2.83333e+10,4.25e+10,-4.25e+10,2.26667e+11,2.125e+10,-4.25e+10,2.83333e+10,2.125e+10,-2.125e+10,-2.83333e+10},{-5.66667e+10,-2.125e+10,-2.125e+10,-2.83333e+10,2.125e+10,2.125e+10,2.83333e+10,2.125e+10,4.25e+10,-7.08333e+10,-2.125e+10,-4.25e+10,-7.08333e+10,-4.25e+10,-2.125e+10,2.83333e+10,4.25e+10,2.125e+10,2.26667e+11,4.25e+10,4.25e+10,-5.66667e+10,-4.25e+10,-4.25e+10},{-2.125e+10,-5.66667e+10,-2.125e+10,-2.125e+10,-7.08333e+10,-4.25e+10,2.125e+10,2.83333e+10,4.25e+10,2.125e+10,-2.83333e+10,2.125e+10,-4.25e+10,-7.08333e+10,-2.125e+10,-4.25e+10,-5.66667e+10,-4.25e+10,4.25e+10,2.26667e+11,4.25e+10,4.25e+10,2.83333e+10,2.125e+10},{-2.125e+10,-2.125e+10,-5.66667e+10,-2.125e+10,-4.25e+10,-7.08333e+10,-4.25e+10,-4.25e+10,-5.66667e+10,-4.25e+10,-2.125e+10,-7.08333e+10,2.125e+10,2.125e+10,-2.83333e+10,2.125e+10,4.25e+10,2.83333e+10,4.25e+10,4.25e+10,2.26667e+11,4.25e+10,2.125e+10,2.83333e+10},{-2.83333e+10,-2.125e+10,-2.125e+10,-5.66667e+10,2.125e+10,2.125e+10,-7.08333e+10,2.125e+10,4.25e+10,2.83333e+10,-2.125e+10,-4.25e+10,2.83333e+10,-4.25e+10,-2.125e+10,-7.08333e+10,4.25e+10,2.125e+10,-5.66667e+10,4.25e+10,4.25e+10,2.26667e+11,-4.25e+10,-4.25e+10},{2.125e+10,-7.08333e+10,-4.25e+10,2.125e+10,-5.66667e+10,-2.125e+10,-2.125e+10,-2.83333e+10,2.125e+10,-2.125e+10,2.83333e+10,4.25e+10,4.25e+10,-5.66667e+10,-4.25e+10,4.25e+10,-7.08333e+10,-2.125e+10,-4.25e+10,2.83333e+10,2.125e+10,-4.25e+10,2.26667e+11,4.25e+10},{2.125e+10,-4.25e+10,-7.08333e+10,2.125e+10,-2.125e+10,-5.66667e+10,4.25e+10,-2.125e+10,-7.08333e+10,4.25e+10,-4.25e+10,-5.66667e+10,-2.125e+10,4.25e+10,2.83333e+10,-2.125e+10,2.125e+10,-2.83333e+10,-4.25e+10,2.125e+10,2.83333e+10,-4.25e+10,4.25e+10,2.26667e+11}};


template <class DataTypes>
void HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::init()
{

// 		  cerr<<"HomogenizedHexahedronFEMForceFieldAndMass<DataTypes>::init()\n";
    // init topology, virtual levels, calls computeMechanicalMatricesByCondensation, handles masses
    NonUniformHexahedronFEMForceFieldAndMassT::init();



//         // create dynamically a sub-graph in order to have a better homogenized mapping
//         // the idea is to non-uniformly maps the finest dofs into the coarse level
//         // and then really maps others dofs (visual, collision) into the finest dofs (by a classic barycentricMapping for example)
//         if( this->_nbVirtualFinerLevels.getValue()!= 0)
//         {
//
//
//
//           // TODO: create new gnode with mechanicalstate containing the finest nodes and a mapping contening the non linear mapping
//           GNode* parentGNode = dynamic_cast<GNode*>( this->getContext()); // the actual node
//
//           GNode* newGNode = new GNode(); // the new sub-graph
//
//           newGNode->setName("automatically created for homogenized mapping");
//           newGNode->setShowBehaviorModels(true);
//           newGNode->setShowVisualModels(true);
//           newGNode->setShowMappings(true);
//           newGNode->setShowMechanicalMappings(true);
//           newGNode->setShowForceFields(true);
//           newGNode->setShowInteractionForceFields(true);
//
//
//           _finestDOF = new MechanicalObjectT();
//           _finestDOF->setName("finest dofs");
//           newGNode->addObject( _finestDOF );
//
//           _mapping = new MappingT( this->mstate,_finestDOF);
//           _mapping->setName("homogenized mapping for the finest dofs");
//           newGNode->addObject( _mapping );
//
//           newGNode->addObject( this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-this->_nbVirtualFinerLevels.getValue()] );
//
//
//           // move all children gnodes downstair this new gnode
//           vector< GNode*> gNodes;
//           for(GNode::ChildIterator it = parentGNode->child.begin(); it != parentGNode->child.end(); ++it)
//           {
//             gNodes.push_back( *it );
//           }
//           for(unsigned i=0;i<gNodes.size();++i)
//           {
//             newGNode->moveChild( gNodes[i] );
//
// 			for(GNode::ObjectIterator it = gNodes[i]->object.begin(); it != gNodes[i]->object.end(); ++it)
// 			{
// // 				cerr<<(*it)->getTypeName()<<endl;
//
// 				//since previous mapping are moved (but already initialize bu create(...) fonction, we have to manually map the fromModel with the new created dofs
//
// 				if( core::Mapping< core::componentmodel::behavior::State<DataTypes>,core::componentmodel::behavior::MappedModel<ExtVec3fTypes> >* map = dynamic_cast<core::Mapping<core::componentmodel::behavior::State<DataTypes>,core::componentmodel::behavior::MappedModel<ExtVec3fTypes> > *>(*it) )
// 				{
// 					// VISUAL MAPPING
// 					map->setModels( _finestDOF, dynamic_cast<core::componentmodel::behavior::MappedModel<ExtVec3fTypes>*>(map->getTo()) );
// // 					cerr<<"core::Mapping<DataTypes,MappedModel* "<<map->getName()<<endl;
// 				}
// 				else if( MechanicalMappingT* map = dynamic_cast<MechanicalMappingT*>(*it) )
// 				{
// 					//MECHANICAL MAPPING
// 					map->setModels( _finestDOF, dynamic_cast<MechanicalStateT*>(map->getTo()) );
// // 					cerr<<"MechanicalMappingT* "<<map->getName()<<endl;
// 				}
// 			}
//           }
//
//
//
//           // really add the new gnode in the graph
//           parentGNode->addChild(newGNode);
//
//
//         }



    _drawSize = (this->_sparseGrid->getMax()[0]-this->_sparseGrid->getMin()[0]) * .004;


}




/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////




template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( )
{
    if( this->_nbVirtualFinerLevels.getValue() == 0 )
    {
        for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
        {
            //Get the 8 indices of the coarser Hexa
            const helper::fixed_array<unsigned int,8>& points = this->_sparseGrid->getHexas()[i];
            //Get the 8 points of the coarser Hexa
            helper::fixed_array<Coord,8> nodes;

            for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->getPointPos(points[k]);


            //       //given an elementIndice, find the 8 others from the sparse grid
            //       //compute MaterialStiffness
            MaterialStiffness material;
            computeMaterialStiffness(material, this->f_youngModulus.getValue(),this->f_poissonRatio.getValue());

            //Nodes are found using Sparse Grid
            Real stiffnessCoef = 1.0;
            if (topology::SparseGridMultipleTopology* sgmt =  dynamic_cast<topology::SparseGridMultipleTopology*>( this->_sparseGrid ) )
                stiffnessCoef = sgmt->getStiffnessCoef( i );
            else if( this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY )
                stiffnessCoef = .5;


            HexahedronFEMForceFieldAndMassT::computeElementStiffness((*this->_elementStiffnesses.beginEdit())[i],material,nodes,i, stiffnessCoef); // classical stiffness

            HexahedronFEMForceFieldAndMassT::computeElementMass((*this->_elementMasses.beginEdit())[i],nodes,i,this->_sparseGrid->getType(i)==topology::SparseGridTopology::BOUNDARY?.5:1.0);
        }
        return;
    }





    _weights.resize( this->_nbVirtualFinerLevels.getValue() );
    int finestLevel = this->_sparseGrid->getNbVirtualFinerLevels()-this->_nbVirtualFinerLevels.getValue();

    for(int i=0; i<this->_nbVirtualFinerLevels.getValue(); ++i)
    {
        _weights[i].resize( this->_sparseGrid->_virtualFinerLevels[finestLevel+i]->getNbHexas() );
    }

    _finalWeights.resize( _weights[0].size() );



    if( _finestToCoarse.getValue() )
        for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
            computeMechanicalMatricesDirectlyFromTheFinestToCoarse( (*this->_elementStiffnesses.beginEdit())[i], (*this->_elementMasses.beginEdit())[i], i );
    else
    {
        topology::SparseGridRamificationTopology* sparseGridRamification = dynamic_cast<topology::SparseGridRamificationTopology*>( this->_sparseGrid );
        if( _useRamification.getValue() && sparseGridRamification )
        {
            for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
                computeMechanicalMatricesIterativlyWithRamifications( (*this->_elementStiffnesses.beginEdit())[i], (*this->_elementMasses.beginEdit())[i], i, 0 );


            for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
            {
                Weight A; A.identity();

                helper::fixed_array<helper::vector<int>,8 >& finerChildrenRamification = sparseGridRamification->_hierarchicalCubeMapRamification[ i ];

                for(int w=0; w<8; ++w)
                    for(unsigned v=0; v<finerChildrenRamification[w].size(); ++v)
                        computeFinalWeightsRamification( A, i, finerChildrenRamification[w][v], 1 );
            }

        }
        else
        {
            for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
                computeMechanicalMatricesIterativly( (*this->_elementStiffnesses.beginEdit())[i], (*this->_elementMasses.beginEdit())[i], i, 0 );

            for (unsigned int i=0; i<this->_indexedElements->size(); ++i)
            {
                Weight A; A.identity();

                helper::fixed_array<int,8> finerChildren = this->_sparseGrid->_hierarchicalCubeMap[i];

                for(int w=0; w<8; ++w)
                    computeFinalWeights( A, i, finerChildren[w], 1 );
            }
        }
    }

// 			  	for( unsigned i=0;i<_weights.size();++i)
// 				{
// 					printMatlab(cerr,_weights[i][0]);
// 					cerr<<"\n-----------------\n";
// 				}
// 				printMatlab(cerr,_finalWeights[0].second);
// 				cerr<<endl;


// 				//VERIF
// 				for(int i=0;i<8*3;++i)
// 				{
// 					Real sum = 0.0;
// 					for(int j=0;j<8*3;++j)
// 					{
// 						sum += _finalWeights[0].second[i][j];
// 					}
// 					if( fabs(sum-1.0)>1.0e-3 )
// 						cerr<<"WARNING HomogenizedHexahedronFEMForceFieldAndMass _finalWeights sum != 1  "<<sum<<endl;
// 				}



    _weights.resize(0);

// 			  for(unsigned i=0;i<this->_elementStiffnesses.getValue().size();++i)
// 			  {
// 				  cerr<<"K"<<i<<"=";
// 				  printMatlab(cerr,this->_elementStiffnesses.getValue()[i]);
// 			  }

// 			  printMatlab( cerr,this->_elementStiffnesses.getValue()[0] );
}



template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesDirectlyFromTheFinestToCoarse( ElementStiffness &K, ElementMass &M, const int elementIndice)
{
    helper::vector<int> finestChildren;

    //find them
    findFinestChildren( finestChildren, elementIndice );


    SparseGridTopology*finestSparseGrid = this->_sparseGrid->_virtualFinerLevels[ this->_sparseGrid->getNbVirtualFinerLevels()-this->_nbVirtualFinerLevels.getValue() ];

    cerr<<"finestChildren.size() : "<<finestChildren.size()<<endl;
    cerr<<"finestSparseGrid->getNbHexas() : "<<finestSparseGrid->getNbHexas()<<endl;

    int sizeass=2;
    for(int i=0; i<this->_nbVirtualFinerLevels.getValue(); ++i)
        sizeass = (sizeass-1)*2+1;
    sizeass = sizeass*sizeass*sizeass;

    NewMatMatrix assembledStiffness(sizeass*3),assembledMass(sizeass*3);
    assembledStiffness.resize(sizeass*3,sizeass*3);
    assembledMass.resize(sizeass*3,sizeass*3);
    cerr<<assembledStiffness.rowSize()<<"x"<<assembledStiffness.colSize()<<endl;



    helper::vector<ElementStiffness> finestStiffnesses(finestChildren.size());
    helper::vector<ElementMass> finestMasses(finestChildren.size());


    std::map<int,int> map_idxq_idxass; // map a fine point idx to a assembly (local) idx
    int idxass = 0;



    // compute the classical mechanical matrices at the finest level
    for(unsigned i=0 ; i < finestChildren.size() ; ++i )
    {
        computeClassicalMechanicalMatrices(finestStiffnesses[i],finestMasses[i],finestChildren[i],this->_sparseGrid->getNbVirtualFinerLevels()-this->_nbVirtualFinerLevels.getValue());

        const SparseGridTopology::Hexa& hexa = finestSparseGrid->getHexa( finestChildren[i] );


        for(int w=0; w<8; ++w) // sommets
        {
            // idx for assembly
            if( !map_idxq_idxass[ hexa[w] ] )
            {
                map_idxq_idxass[ hexa[w] ] = /*FineHexa_FineNode_IndiceForAssembling[i][w];*/idxass;
                idxass++;
            }
        }

        // assembly
        for(int j=0; j<8; ++j) // vertices1
        {
            int v1 = map_idxq_idxass[hexa[j]];

            for(int k=0; k<8; ++k) // vertices2
            {
                int v2 = map_idxq_idxass[hexa[k]];



                for(int m=0; m<3; ++m)
                    for(int n=0; n<3; ++n)
                    {
                        assembledStiffness.add( v1*3+m, v2*3+n, finestStiffnesses[i][j*3+m][k*3+n] );
                        assembledMass.add( v1*3+m, v2*3+n, finestMasses[i][j*3+m][k*3+n] );
                    }
            }
        }
    }




    std::map<int,int> map_idxq_idxcutass; // map a fine point idx to a the cut assembly (local) idx
    int idxcutass = 0;
    std::map<int,bool> map_idxq_coarse;
    helper::fixed_array<int,8> map_idxcoarse_idxfine;
    const SparseGridTopology::Hexa& coarsehexa = this->_sparseGrid->getHexa( elementIndice );

// 		cerr<<"BUILT\n";

    for(int i=0; i<sizeass; ++i)
    {
        for( std::map<int,int>::iterator it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end(); ++it)
            if( (*it).second==i)
            {
// 					cerr<<(*it).first<<" "<<(*it).second<<endl;
                bool ok=false;
                Coord finesommet = finestSparseGrid->getPointPos( (*it).first );
                for( unsigned sc=0; sc<8; ++sc)
                {
                    Coord coarsesommet = this->_sparseGrid->getPointPos( coarsehexa[sc] );
                    if( coarsesommet == finesommet )
                    {
                        map_idxq_idxcutass[(*it).second] = sc;
                        map_idxq_coarse[  (*it).second] = true;
                        map_idxcoarse_idxfine[ sc ] = (*it).second;
                        ok=true;
                        break;
                    }
                }
                if( !ok )
                {
                    map_idxq_idxcutass[ (*it).second] = idxcutass;
                    map_idxq_coarse[(*it).second] = false;
                    idxcutass++;
                }
            }
    }





// 		for( std::map<int,int>::iterator it = map_idxq_idxass.begin();it!=map_idxq_idxass.end();++it)
// 		{
// 			bool ok=false;
// 			Coord finesommet = finestSparseGrid->getPointPos( (*it).first );
// 			for( unsigned sc=0;sc<8;++sc)
// 			{
// 				Coord coarsesommet = this->_sparseGrid->getPointPos( coarsehexa[sc] );
// 				if( coarsesommet == finesommet )
// 				{
// 					map_idxq_idxcutass[(*it).second] = sc;
// 					map_idxq_coarse[  (*it).second] = true;
// 					map_idxcoarse_idxfine[ sc ] = (*it).first;
// 					ok=true;
// 					break;
// 				}
// 			}
// 			if( !ok )
// 			{
// 				map_idxq_idxcutass[ (*it).second] = idxcutass;
// 				map_idxq_coarse[(*it).second] = false;
// 				idxcutass++;
// 			}
// 		}





// 		for(unsigned i=0 ; i < finestChildren.size() ; ++i )
// 		{
// 			computeClassicalMechanicalMatrices(finestStiffnesses[i],finestMasses[i],finestChildren[i],this->_sparseGrid->getNbVirtualFinerLevels()-this->_nbVirtualFinerLevels.getValue());
//
// 			const SparseGridTopology::Hexa& hexa = finestSparseGrid->getHexa( finestChildren[i] );
//
//
// 			for(int w=0;w<8;++w) // sommets
// 			{
//
// 				cerr<<map_idxq_idxass[ hexa[w] ] <<",";
// 			}
// 			cerr<<endl;
// 		}


// 		for( std::map<int,int>::iterator it = map_idxq_idxass.begin();it!=map_idxq_idxass.end();++it)
// 		{
// 			cerr<<(*it).first<<" "<<(*it).second<<endl;
// 		}
//
// 		for( std::map<int,int>::iterator it = map_idxq_idxcutass.begin();it!=map_idxq_idxcutass.end();++it)
// 		{
// 			cerr<<(*it).second<<",";
// 		}
// 		cerr<<endl;
//
// 		for( std::map<int,bool>::iterator it = map_idxq_coarse.begin();it!=map_idxq_coarse.end();++it)
// 		{
// 			cerr<<(*it).second<<",";
// 		}
// 		cerr<<endl;


// 		cerr<<map_idxcoarse_idxfine<<endl;


    NewMatMatrix Kg; // stiffness of contrained nodes
    Kg.resize(sizeass*3,8*3);
    NewMatMatrix  A; // [Kf -G] ==  Kf (stiffness of free nodes) with the constaints
    A.resize(sizeass*3,sizeass*3);
    NewMatMatrix  Ainv;
// 		Ainv.resize(sizeass*3,sizeass*3);



    for ( int i=0; i<sizeass; ++i)
    {
        int col = map_idxq_idxcutass[i];

        if( map_idxq_coarse[i] )
        {
            for(int lig=0; lig<sizeass; ++lig)
            {
                for(int m=0; m<3; ++m)
                    for(int n=0; n<3; ++n)
                        Kg.add( lig*3+m,col*3+n,assembledStiffness.element(lig*3+m,i*3+n) );
            }
        }
        else
        {
            for(int lig=0; lig<sizeass; ++lig)
            {
                for(int m=0; m<3; ++m)
                    for(int n=0; n<3; ++n)
                        A.add( lig*3+m,col*3+n,assembledStiffness.element(lig*3+m,i*3+n) );
            }
        }
    }





// 		  put -G entries into A
    for(int i=0; i<8; ++i) // for all constrained nodes
    {
        A.add( map_idxcoarse_idxfine[i]*3   , (sizeass-8+i)*3   , -1.0);
        A.add( map_idxcoarse_idxfine[i]*3+1 , (sizeass-8+i)*3+1 , -1.0);
        A.add( map_idxcoarse_idxfine[i]*3+2 , (sizeass-8+i)*3+2 , -1.0);
    }







    Ainv = A.i();







    NewMatMatrix  Ainvf;
// 		Ainvf.resize((sizeass-8)*3,sizeass*3);
    Ainv.getSubMatrix( 0,0, (sizeass-8)*3,sizeass*3,Ainvf);






    NewMatMatrix  W;
// 		W.resize((sizeass-8)*3,8*3);
    W = - Ainvf * Kg;





    NewMatMatrix  WB;
    WB.resize(sizeass*3,8*3);
    for(int i=0; i<sizeass*3; ++i)
    {
        int idx = i/3;
        int mod = i%3;
        if( map_idxq_coarse[idx] )
            WB.add( i , map_idxq_idxcutass[idx]*3+mod , 1.0);
        else
            for(int j=0; j<8*3; ++j)
            {
                WB.add( i,j, W.element( map_idxq_idxcutass[idx]*3+mod, j));
            }
    }


// 		cerr<<"KB2 = ";
// 		assembledStiffness.printMatlab(cerr);
// 		cerr<<"A2 = ";
// 		A.printMatlab(cerr);
// 		cerr<<"Kg2 = ";
// 		Kg.printMatlab(cerr);
// 		cerr<<"Ainv2 = ";
// 		Ainv.printMatlab(cerr);
// 		cerr<<"Ainvf2 = ";
// 		Ainvf.printMatlab(cerr);
// 		cerr<<"W2 = ";
// 		W.printMatlab(cerr);
// 		cerr<<"WB2 = ";
// 		WB.printMatlab(cerr);


// 		for( map<int,int>::iterator it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end();++it)
// 		{
// 			if( map_idxq_coarse[ (*it).second ] )
// 				cerr<< map_idxq_idxcutass[(*it).second] <<" "<<finestSparseGrid->getPointPos( (*it).first )<<endl;
// 		}


    NewMatMatrix  mask;
    mask.resize(sizeass*3,8*3);

    Coord a = this->_sparseGrid->getPointPos(coarsehexa[0]);
    Coord b = this->_sparseGrid->getPointPos(coarsehexa[6]);
    Coord dx( b[0]-a[0],0,0),dy( 0,b[1]-a[1],0), dz( 0,0,b[2]-a[2]);
    Coord inv_d2( 1.0/(dx*dx),1.0/(dy*dy),1.0/(dz*dz) );
    for( map<int,int>::iterator it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end(); ++it)
    {
        int localidx = (*it).second; // indice du noeud fin dans l'assemblage


        if( map_idxq_coarse[ (*it).second ] )
        {
// 				cerr<<map_idxq_idxcutass[ (*it).second ]<<" "<<finestSparseGrid->getPointPos( (*it).first )<<endl;
            int localcoarseidx = map_idxq_idxcutass[ (*it).second ];
            mask.set( localidx*3  , localcoarseidx*3   , 1);
            mask.set( localidx*3+1, localcoarseidx*3+1 , 1);
            mask.set( localidx*3+2, localcoarseidx*3+2 , 1);
        }
        else
        {

            // find barycentric coord
            Coord p = finestSparseGrid->getPointPos( (*it).first ) - a;

            Real fx = p*dx*inv_d2[0];
            Real fy = p*dy*inv_d2[1];
            Real fz = p*dz*inv_d2[2];


            helper::fixed_array<Real,8> baryCoefs;
            baryCoefs[0] = (1-fx) * (1-fy) * (1-fz);
            baryCoefs[1] = fx * (1-fy) * (1-fz);
            baryCoefs[2] = fx * (fy) * (1-fz);
            baryCoefs[3] = (1-fx) * (fy) * (1-fz);
            baryCoefs[4] = (1-fx) * (1-fy) * (fz);
            baryCoefs[5] = fx * (1-fy) * (fz);
            baryCoefs[6] = fx * (fy) * (fz);
            baryCoefs[7] = (1-fx) * (fy) * fz;


// 				cerr<<localidx<<"        "<<baryCoefs<<endl<<finestSparseGrid->getPointPos( (*it).first )<<" = ";

            for(int i=0; i<8; ++i)
            {
                if( baryCoefs[i]>1.0e-5 )
                {
// 						cerr<<"("<<i<<") "<<this->_sparseGrid->getPointPos( i )<<" + "<<endl;
                    mask.set( localidx*3  , i*3   , 1);
                    mask.set( localidx*3+1, i*3+1 , 1);
                    mask.set( localidx*3+2, i*3+2 , 1);
                }
            }
// 				cerr<<endl<<endl<<endl;
        }
    }


// 		for(int i=0;i<sizeass*3;++i)
// 		{
// 			for(int j=0;j<8*3;++j)
// 			{
// 				if( mask.element(i,j) != WEIGHT_MASK[i][j])
// 				{
// 					cerr<<"MASK ERROR "<<i/3<<" "<<mask.Row(i).Sum()<<"\n";
// 					break;
// 				}
// 			}
// 		}



// 		Coord a = this->_sparseGrid->getPointPos(coarsehexa[0]);
// 		Coord b = this->_sparseGrid->getPointPos(coarsehexa[6]);
//
// 		helper::vector< defaulttype::Vector6 > inPlan(sizeass); // is a point in planes  0yz, 1yz, x0z, x1z, xy0, xy1
//
// 		for( map<int,int>::iterator it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end();++it)
// 		{
// 			Coord p = finestSparseGrid->getPointPos( (*it).first );
// 			int localidx = (*it).second;
//
// // 			if( map_idxq_coarse[ (*it).second ] )
// // 			{
// // 				// GROSSIER
// // 			}
// // 			else
// 			{
// 				if( p[0] == a[0] ) // plan 0yz
// 				{
// 					inPlan[ localidx ][0] = 1;
// 				}
// 				if( p[0] == b[0] ) // plan 1yz
// 				{
// 					inPlan[ localidx ][1] = 1;
// 				}
// 				if( p[1] == a[1] ) // plan x0z
// 				{
// 					inPlan[ localidx ][2] = 1;
// 				}
// 				if( p[1] == b[1] ) // plan x1z
// 				{
// 					inPlan[ localidx ][3] = 1;
// 				}
// 				if( p[2] == a[2] ) // plan xy0
// 				{
// 					inPlan[ localidx ][4] = 1;
// 				}
// 				if( p[2] == b[2] ) // plan xy1
// 				{
// 					inPlan[ localidx ][5] = 1;
// 				}
//
// 				switch( static_cast<int>(inPlan[ localidx ].sum()) )
// 				{
// 					case 0: // in the middle
// 					{
// 						for(int i=0;i<8;++i)
// 						{
// 							mask( localidx*3  , i*3   ) = 1;
// 							mask( localidx*3+1, i*3+1 ) = 1;
// 							mask( localidx*3+2, i*3+2 ) = 1;
// 						}
// 						break;
// 					}
// 					case 1: // on a plane
// 					{
// 						helper::fixed_array<int,4> whichCoarseNodesInLocalIndices;
// 						for(int i=0;i<6;++i)
// 						{
// 							if( inPlan[ localidx ][i] )
// 							{
// 								switch(i)
// 								{
// 									case 0:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(0,3,4,7);
// 										break;
// 									case 1:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(1,2,5,6);
// 										break;
// 									case 2:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(0,1,4,5);
// 										break;
// 									case 3:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(2,3,6,7);
// 										break;
// 									case 4:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(0,1,2,3);
// 										break;
// 									case 5:
// 										whichCoarseNodesInLocalIndices = helper::fixed_array<int,4>(4,5,6,7);
// 										break;
// 								}
// 								break;
// 							}
// 						}
//
// 						for(int i=0;i<4;++i)
// 						{
//
// 						}
//
// 						break;
// 					}
// 							case 2: // on an edge
// 								break;
// 								case 3: // a coarse node
// 									break;
// 					default:
// 						cerr<<"HomogenizedHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesDirectlyFromTheFinestToCoarse   ERROR  WEIGHT_MASK\n";
//  				}
//
// 			}
// 		}




    // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
    NewMatMatrix WBmeca;
    WBmeca.resize(sizeass*3,8*3);
    for(int i=0; i<sizeass*3; ++i)
    {
        for(int j=0; j<8*3; ++j)
        {
            if( mask.element(i,j) /*WEIGHT_MASK[i][j]*/ )
                WBmeca.set(i,j,WB.element(i,j));
        }
    }


// 		cerr<<"WB : "<<WB<<endl;
    cerr<<"WBmeca brut : "<<WBmeca<<endl;



    // normalize the coefficient to obtain sum(coefs)==1
    for(int i=0; i<sizeass*3; ++i)
    {
        Real sum = 0.0;
        for(int j=0; j<8*3; ++j)
        {
            sum += WBmeca.element(i,j);
        }
        for(int j=0; j<8*3; ++j)
        {
            WBmeca.set(i,j, WBmeca.element(i,j) / sum );
// 				WB.set(i,j, WB.element(i,j) / sum );
        }
    }

    // 		cerr<<"mask : "<<mask<<endl;
// 		cerr<<"WB : "<<WB<<endl;
    cerr<<"WBmeca normalized : "<<WBmeca<<endl;

// 		WBmeca=WB;

    NewMatMatrix Kc, Mc; // coarse stiffness
// 		Kc.resize(8*3,8*3);
// 		Mc.resize(8*3,8*3);
    Kc = WBmeca.t() * assembledStiffness * WBmeca;
    Mc = WBmeca.t() * assembledMass * WBmeca;





    for(int i=0; i<8*3; ++i)
        for(int j=0; j<8*3; ++j)
        {
            K[i][j]=Kc.element(i,j);
            M[i][j]=Mc.element(i,j);
        }




    if( !_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
    {
        WB = WBmeca;
    }


    for(unsigned i=0 ; i < finestChildren.size() ; ++i )
    {
        const SparseGridTopology::Hexa& hexa = finestSparseGrid->getHexa( finestChildren[i] );
        for(int j=0; j<8; ++j)
        {
            for( int k=0; k<8*3; ++k)
            {
                _finalWeights[finestChildren[i]].second[j*3  ][k] = WB.element( map_idxq_idxass[ hexa[j] ]*3   ,k);
                _finalWeights[finestChildren[i]].second[j*3+1][k] = WB.element( map_idxq_idxass[ hexa[j] ]*3+1 ,k);
                _finalWeights[finestChildren[i]].second[j*3+2][k] = WB.element( map_idxq_idxass[ hexa[j] ]*3+2 ,k);
            }
        }
        _finalWeights[finestChildren[i]].first = elementIndice;
    }

    /*
    		cerr<<"Kf = ";
    		printMatlab( cerr,finestStiffnesses[0] );
    		cerr<<"Kc = ";
    		printMatlab( cerr,K );*/


}





template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::findFinestChildren( helper::vector<int>& finestChildren, const int elementIndice, int level)
{
    if (level == this->_nbVirtualFinerLevels.getValue())
    {
        finestChildren.push_back( elementIndice );
    }
    else
    {
        helper::fixed_array<int,8> finerChildren;
        if (level == 0)
        {
            finerChildren = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
        }
        else
        {
            finerChildren = this->_sparseGrid->_virtualFinerLevels[this->_nbVirtualFinerLevels.getValue()-level]->_hierarchicalCubeMap[elementIndice];
        }

        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            findFinestChildren( finestChildren, finerChildren[i], level+1 );
        }
    }
}



template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesIterativly( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level)
{
    // 		  NonUniformHexahedronFEMForceFieldAndMassT::computeMechanicalMatricesByCondensation(K,M,elementIndice,level);


    if (level == this->_nbVirtualFinerLevels.getValue())
    {
        computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
// 		  printMatlab( cerr, K );
    }
    else
    {
        helper::fixed_array<int,8> finerChildren;

        topology::SparseGridTopology* sparseGrid,*finerSparseGrid;

        if (level == 0)
        {
            sparseGrid = this->_sparseGrid;
            finerSparseGrid = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-1];
        }
        else
        {
            sparseGrid = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level];
            finerSparseGrid = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level-1];
        }

        finerChildren = sparseGrid->_hierarchicalCubeMap[elementIndice];


        helper::fixed_array<ElementStiffness,8> finerK;
        helper::fixed_array<ElementMass,8> finerM;

        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if (finerChildren[i] != -1)
            {
                computeMechanicalMatricesIterativly(finerK[i], finerM[i], finerChildren[i], level+1);
            }

// 			cerr<<"K "<<i<<" : "<<finerK[i]<<endl;

        }




//           cerr<<"\n***LEVEL "<<level<<"    element "<<elementIndice<<endl;



        // assemble the matrix of 8 child
        Mat<27*3, 27*3, Real> assembledStiffness;
        Mat<27*3, 27*3, Real> assembledStiffnessWithRigidVoid;
        Mat<27*3, 27*3, Real> assembledMass;


        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if( finerChildren[i]!=-1)
            {
                for(int j=0; j<8; ++j) // vertices1
                {
                    int v1 = FineHexa_FineNode_IndiceForAssembling[i][j];

                    for(int k=0; k<8; ++k) // vertices2
                    {
                        int v2 = FineHexa_FineNode_IndiceForAssembling[i][k];

                        for(int m=0; m<3; ++m)
                            for(int n=0; n<3; ++n)
                            {
                                assembledStiffness[ v1*3+m ][ v2*3+n ] += finerK[i][j*3+m][k*3+n];
                                assembledStiffnessWithRigidVoid[ v1*3+m ][ v2*3+n ] += finerK[i][j*3+m][k*3+n];
                                assembledMass[ v1*3+m ][ v2*3+n ] += finerM[i][j*3+m][k*3+n];
                            }
                    }
                }
            }
            else
            {
// 				cerr<<"WARNING: a child is void (during assembly)\n";
                for(int j=0; j<8; ++j) // vertices1
                {
                    int v1 = FineHexa_FineNode_IndiceForAssembling[i][j];

                    for(int k=0; k<8; ++k) // vertices2
                    {
                        int v2 = FineHexa_FineNode_IndiceForAssembling[i][k];

                        for(int m=0; m<3; ++m)
                            for(int n=0; n<3; ++n)
                            {
                                assembledStiffnessWithRigidVoid[ v1*3+m ][ v2*3+n ] += RIGID_STIFFNESS[j*3+m][k*3+n];
                            }
                    }
                }
            }
        }


        Mat<27*3, 8*3, Real> Kg; // stiffness of contrained nodes
        Mat<27*3, 27*3, Real> A; // [Kf -G]  Kf (stiffness of free nodes) with the constaints
        Mat<27*3, 27*3, Real> Ainv;


        for ( int i=0; i<27; ++i)
        {
            int col = FineHexa_FineNode_IndiceForCutAssembling_27[i];

            if( IS_CONSTRAINED_27[i] )
            {
                for(int lig=0; lig<27; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            Kg[ lig*3+m ][ col*3+n ] = assembledStiffnessWithRigidVoid[lig*3+m][i*3+n];
                }
            }
            else
            {
                for(int lig=0; lig<27; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            A[ lig*3+m ][ col*3+n ] = assembledStiffnessWithRigidVoid[lig*3+m][i*3+n];
                }
            }

        }


        // put -G entries into A
        for(int i=0; i<8; ++i) // for all constrained nodes
        {
            A[ CoarseToFine[i]*3   ][ (27-8+i)*3   ] = -1.0;
            A[ CoarseToFine[i]*3+1 ][ (27-8+i)*3+1 ] = -1.0;
            A[ CoarseToFine[i]*3+2 ][ (27-8+i)*3+2 ] = -1.0;
        }


// 		  cerr<<"KB = "; printMatlab( cerr, assembledStiffness );
// 		  cerr<<"A = ";
// 		  printMatlab( cerr, A );
// 		  cerr<<"Kg = ";
// 		  printMatlab( cerr, Kg );




        Ainv.invert(A);




// 		  Mat<8*3, 27*3, Real> Ainvg;
// 		  for(int i=0;i<8;++i)
// 		  {
// 				  for(int m=0;m<3;++m)
// 					  for(int n=0;n<3;++n)
// 						  Ainvg[i*3+m] = - Ainv.line( (27-8+i)*3+m );
// 		  }
//
// 		  K = Ainvg * Kg;




        Mat<(27-8)*3, 27*3, Real> Ainvf;
        for(int i=0; i<27-8; ++i)
        {
            for(int m=0; m<3; ++m)
// 				  for(int n=0;n<3;++n)
                Ainvf[i*3+m] = - Ainv.line( i*3+m );
        }




        Mat<(27-8)*3, 8*3, Real> W;
        W = Ainvf * Kg;


        Mat<27*3, 8*3, Real> WB;
        for(int i=0; i<27*3; ++i)
        {
            int idx = i/3;
            int mod = i%3;
            if( IS_CONSTRAINED_27[idx] )
                WB[ i ][ FineHexa_FineNode_IndiceForCutAssembling_27[idx]*3+mod ] = 1.0;
            else
                WB[ i ] = W[ FineHexa_FineNode_IndiceForCutAssembling_27[idx]*3+mod ];
        }



// 		  cerr<<"Ainv = ";
// 		  printMatlab( cerr, Ainv );
// 		  cerr<<"Ainvf = ";
// 		  printMatlab( cerr, Ainvf );
// 		  cerr<<"W = ";
// 		  printMatlab( cerr, W );
// 		  cerr<<"WB = ";
// 		  printMatlab( cerr, WB );



        // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
        Mat<27*3, 8*3, Real> WBmeca;
        for(int i=0; i<27*3; ++i)
        {
            for(int j=0; j<8*3; ++j)
            {
                if( WEIGHT_MASK[i][j] )
                    WBmeca[i][j]=WB[i][j];
            }
        }


        helper::vector<Real> sum_wbmeca(27*3,0);
        // normalize the coefficient to obtain sum(coefs)==1
        for(int i=0; i<27*3; ++i)
        {
// 			  Real sum = 0.0;
            for(int j=0; j<8*3; ++j)
            {
                sum_wbmeca[i] += WBmeca[i][j];
            }
            for(int j=0; j<8*3; ++j)
            {
                WBmeca[i][j] /= sum_wbmeca[i];
            }
        }


// 		  cerr<<"\nWsofa = ";
// 		  printMatlab( cerr, WB );


        K = WBmeca.multTranspose( assembledStiffness * WBmeca );


// 		  cerr<<"\nWsofa = ";
// 		  printMatlab( cerr, W );
// 		  cerr<<"\nWBsofa = ";
// 		  printMatlab( cerr, WB );



// 		  cerr<<"K is sym : "<<K.isSymetric()<<endl;



// 		  cerr<<"\nAinv1sofa = ";
// 		  printMatlab( cerr, Ainv1 );
// 		  cerr<<"\nKsofa = ";
// 		  printMatlab( cerr, K );


// 		  M = WBmeca.multTranspose( assembledMass * WBmeca );


        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if (finerChildren[i] != -1)
            {
                this->addFineToCoarse(M, finerM[i], i);
            }
        }








// 		  cerr<<WB[16*3+1]<<endl;

// 		  helper::fixed_array< Mat<8*3, 8*3, Real>, 8 >  Welem; // weights matrices per elem : from the coarse elem to each fine element



        if( !_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
        {
            WB = WBmeca;
        }
        else
        {
// 			  for(int i=0;i<27*3;++i)
// 			  {
// 				  for(int j=0;j<8*3;++j)
// 				  {
// 						  WB[i][j] *= WEIGHT_MASK_CROSSED[i][j];
// 				  }
// 			  }
//
//
// // 			  cerr<<"WEIGHT_MASK_CROSSED : \n";
// // 			  cerr<<WB[16*3+1]<<endl;
//
//
// 		  // normalize the coefficient to obtain sum(coefs)==1
// 			  for(int i=0;i<27*3;++i)
// 			  {
// // 				  Real sum = 0.0;
// // 				  for(int j=0;j<8*3;++j)
// // 				  {
// // 					  sum += (WBmeca[i][j]);
// // 				  }
//
// 				  for(int j=0;j<8*3;++j)
// 				  {
// 					  WB[i][j] /= sum_wbmeca[i];
// 				  }
// 			  }



            for(int i=0; i<27*3; ++i)
            {
                for(int j=0; j<8*3; ++j)
                {
                    if( !WEIGHT_MASK_CROSSED_DIFF[i][j] )
                        WB[i][j] = WBmeca[i][j];
                    else
                    {
// 						  WB[i][j] *= WEIGHT_MASK_CROSSED_DIFF[i][j]*2.5;
                        WB[i][j] = WB[i][j]/fabs(WB[i][j]) * WEIGHT_MASK_CROSSED_DIFF[i][j] * this->f_poissonRatio.getValue() * .3;
                    }
                }
            }



        }

// 		  cerr<<"normalize : \n";
// 		  cerr<<WB[16*3+1]<<endl;


        for(int elem=0; elem<8; ++elem)
        {
            if( finerChildren[elem] != -1)
            {
                for(int i=0; i<8; ++i)
                {
                    // 				  Welem[elem][i*3  ] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3 ];
                    // 				  Welem[elem][i*3+1] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+1 ];
                    // 				  Welem[elem][i*3+2] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+2 ];

                    _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]][i*3  ] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3  ];
                    _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]][i*3+1] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+1];
                    _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]][i*3+2] = WB [ FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+2];
                }
            }

// 			  if(finerChildren[elem]==2)
// 			  {
// 				  cerr<<"BUILD\n";
// 				  cerr<<this->_nbVirtualFinerLevels.getValue()-level-1<<endl;
// 				  cerr<<_weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]]<<endl;
// 			  }
        }





// 		  cerr<<FineHexa_FineNode_IndiceForAssembling[ 1 ][ 2 ]<<" : "<<WB [ FineHexa_FineNode_IndiceForAssembling[ 0 ][ 3 ]*3  ]<<endl;




// 		  cerr<<"\nWcsofa = ";
// 		  printMatlab( cerr, _weights[0][0] );
// 		  cerr<<"\nKcsofa = ";
// 		  printMatlab( cerr, finerK[0] );

// // 		  std::map<int,helper::fixed_array<int,8> > maptmp;
// //
// // 		  for(int i=0;i<27;++i)
// // 			  for(int j=0;j<8;++j)
// // 				  maptmp[i][
// //
// // 		  for(int i=0;i<8;++i)
// // 			  for(int j=0;j<8;++j)
// // 				  maptmp[ FineHexa_FineNode_IndiceForAssembling[i][j] ][i] = j;
// //
// // 		  cerr<<"MAP = {";
// // 		  for( std::map<int,helper::fixed_array<int,8> >::iterator it = maptmp.begin();it != maptmp.end() ;++it)
// // 		  {
// // 			  cerr<<"{";
// // 			  for(int i=0;i<8;++i)
// // 				  cerr<<(*it).second[i]<<",";
// // 			cerr<<"},\n";
// // 		  }
// // 		  cerr<<"}\n";


        // put weights into the mapping
// 		  for(int i=0;i<8;++i)
// 		  {
// 			  for(int j=0;j<8;++j)
// 			  {
// 				  if( !IS_CONSTRAINED[i][j] )
// 				  {
// 					  _mapping->_weights[
// 				  }
//
// 			  }
// 		  }



    }


}






template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeMechanicalMatricesIterativlyWithRamifications( ElementStiffness &K, ElementMass &M, const int elementIndice,  int level)
{
//         		 cerr<<"\n\nNonUniformHexahedronFEMForceFieldAndMassT::computeMechanicalMatricesIterativlyWithRamifications(K,M,"<<elementIndice<<" "<<level<<"\n";


    if (level == this->_nbVirtualFinerLevels.getValue())
    {
        computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
    }
    else
    {
        topology::SparseGridRamificationTopology* sparseGrid,*finerSparseGrid;

        if (level == 0)
        {
            sparseGrid = dynamic_cast<topology::SparseGridRamificationTopology*>(this->_sparseGrid);
            finerSparseGrid = dynamic_cast<topology::SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-1]);
        }
        else
        {
            sparseGrid = dynamic_cast<topology::SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level]);
            finerSparseGrid = dynamic_cast<topology::SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level-1]);
        }

        // trouver les finer elements par ramification
        helper::fixed_array<helper::vector<int>,8 >& finerChildrenRamificationOriginal = sparseGrid->_hierarchicalCubeMapRamification[ elementIndice ];

        helper::fixed_array<helper::vector<ElementStiffness>,8> finerK;
        helper::fixed_array<helper::vector<ElementMass>,8> finerM;


        const SparseGridTopology::Hexa& coarsehexa = sparseGrid->getHexa( elementIndice );


        helper::fixed_array< Coord, 27 > finePositions; // coord of each fine positions
        for(int i=0; i<27; ++i)
        {
            for(int j=0; j<8; ++j)
            {
                finePositions[i] += sparseGrid->getPointPos( coarsehexa[j] ) * MIDDLE_INTERPOLATION[i][j];
            }
        }


        helper::fixed_array< std::set<int>, 27 > fineNodesPerPositions; // list of fine nodes at each fine positions
        for ( int i=0; i<8; ++i) //for 8 virtual finer element positions
        {
            finerK[i].resize( finerChildrenRamificationOriginal[i].size() );
            finerM[i].resize( finerChildrenRamificationOriginal[i].size() );

            for(unsigned j=0; j<finerChildrenRamificationOriginal[i].size(); ++j) // for all finer elements
            {
                computeMechanicalMatricesIterativlyWithRamifications(finerK[i][j], finerM[i][j], finerChildrenRamificationOriginal[i][j], level+1);

                const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexa( finerChildrenRamificationOriginal[i][j] );
                for( int k=0; k<8; ++k) //fine nodes
                {
                    for(int l=0; l<27; ++l)
                    {
                        if( fabs(  (finePositions[l]-finerSparseGrid->getPointPos( finehexa[k] ) ).norm2() )<1.0e-5 )
                        {
                            fineNodesPerPositions[l].insert( finehexa[k] );
                            break;
                        }
                    }
                }
            }
        }
        // donner un indice fictif <0 aux points vides
        int fictifidx = -1;
        for(int i=0; i<27; ++i)
        {
            if( fineNodesPerPositions[i].empty() ) // pas de points ici
            {
                fineNodesPerPositions[i].insert(fictifidx);
                --fictifidx;
            }
        }


// 			  cerr<<"fineNodesPerPositions : "<<endl;
// 			  for(int i=0;i<27;++i)
// 			  {
// 				  cerr<<i<<" : ";
// 				  for(std::set<int>::iterator it=fineNodesPerPositions[i].begin();it!=fineNodesPerPositions[i].end();++it)
// 					  cerr<<*it<<", ";
// 				  cerr<<endl;
// 			  }


        helper::fixed_array<helper::vector<helper::fixed_array<int,8 > >,8 > finerChildrenRamification; // listes des hexas à chaque position, avec des indices fictifs pour les vides
        helper::fixed_array<helper::vector<bool>,8 > isFinerChildrenVirtual; // a boolean, true if ficitf, only created for void
        for ( int i=0; i<8; ++i) //for 8 virtual finer element positions
        {
            if( finerChildrenRamificationOriginal[i].empty() ) // vide
            {
                // construire un element fictif
                helper::fixed_array<int,8 > fictifelem;
                for(int j=0; j<8; ++j) // fine fictif nodes
                {
                    fictifelem[j] = *fineNodesPerPositions[FINE_ELEM_IN_COARSE_IN_ASS_FRAME[i][j]].begin();
                    // TODO: plutot que de prendre que le premier voisin non vide, il faudrait creer plusieurs vides en consequence...
                }
                finerChildrenRamification[i].push_back( fictifelem );
                isFinerChildrenVirtual[i].push_back( 1 );
            }
            else
            {
                for(unsigned j=0; j<finerChildrenRamificationOriginal[i].size(); ++j)
                {
                    const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexa( finerChildrenRamificationOriginal[i][j] );
                    helper::fixed_array<int,8 > elem;
                    for(int k=0; k<8; ++k) // fine fictif nodes
                    {
                        elem[k] = finehexa[k];
                    }
                    finerChildrenRamification[i].push_back(elem);
                    isFinerChildrenVirtual[i].push_back( 0 );
                }
            }
        }






// 			  helper::vector<int> finerChildren;





        std::map<int,int> map_idxq_idxass; // map a fine point idx to a assembly (local) idx
        int idxass = 0;


        for(int i=0; i<27; ++i)
        {
            for( std::set<int>::iterator it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
            {
                map_idxq_idxass[*it] = idxass;
                idxass++;
            }
        }

        int sizeass = idxass; // taille de l'assemblage i.e., le nombre de noeuds fins



// 			  cerr<<"map_idxq_idxass : "<<endl;
// 			  for(std::map<int,int>::iterator it = map_idxq_idxass.begin();it != map_idxq_idxass.end();++it)
// 			  {
// 				  cerr<<(*it).first<<" "<<(*it).second<<endl;
// 			  }




// 			  cerr<<"sizeass : "<<sizeass<<endl;
        NewMatMatrix assembledStiffness,assembledStiffnessStatic,assembledMass;
        assembledStiffness.resize(sizeass*3,sizeass*3);
        assembledStiffnessStatic.resize(sizeass*3,sizeass*3);
        assembledMass.resize(sizeass*3,sizeass*3);
// 			  cerr<<assembledStiffness.rowSize()<<"x"<<assembledStiffness.colSize()<<endl;





        for(int i=0 ; i < 8 ; ++i ) // finer places
        {
            for( unsigned c=0; c<finerChildrenRamification[i].size(); ++c)
            {

                helper::fixed_array<int,8>& finehexa = finerChildrenRamification[i][c];

                if( isFinerChildrenVirtual[i][c] ) // void
                {
                    for(int j=0; j<8; ++j) // vertices1
                    {
                        int v1 = map_idxq_idxass[finehexa[j]];

                        for(int k=0; k<8; ++k) // vertices2
                        {
                            int v2 = map_idxq_idxass[finehexa[k]];

                            for(int m=0; m<3; ++m)
                                for(int n=0; n<3; ++n)
                                {
                                    assembledStiffnessStatic.add( v1*3+m, v2*3+n, RIGID_STIFFNESS[j*3+m][k*3+n] );
                                }
                        }
                    }
                }
                else
                {

                    // assembly
                    for(int j=0; j<8; ++j) // vertices1
                    {
                        int v1 = map_idxq_idxass[finehexa[j]];

                        for(int k=0; k<8; ++k) // vertices2
                        {
                            int v2 = map_idxq_idxass[finehexa[k]];

                            for(int m=0; m<3; ++m)
                                for(int n=0; n<3; ++n)
                                {
                                    assembledStiffness.add( v1*3+m, v2*3+n, finerK[i][c][j*3+m][k*3+n] );
                                    assembledStiffnessStatic.add( v1*3+m, v2*3+n, finerK[i][c][j*3+m][k*3+n] );
                                    assembledMass.add( v1*3+m, v2*3+n, finerM[i][c][j*3+m][k*3+n] );
                                }
                        }
                    }
                }
            }
        }


// 			  cerr<<"KB2=";
// 			  assembledStiffnessStatic.printMatlab( cerr );


        std::map<int,int> map_idxq_idxcutass; // map a fine point idx to a the cut assembly (local) idx
        int idxcutass = 0,idxcutasscoarse = 0;
        std::map<int,int> map_idxq_coarse; // a fine idx -> -1->non coarse, x-> idx coarse node
        helper::fixed_array<helper::vector<int> ,8> map_idxcoarse_idxfine;

// 			  NewMatMatrix  mask;
// 			  mask.resize(sizeass*3,8*3);

// 			  std::map<int,std::pair< helper::vector<int>,unsigned > > map_mask; // for each fine node -> a list of depensing coase nodes and in which axes (0==all, 1==x, 2==y, 3==z)


        for(int i=0; i<27; ++i)
        {
            if( i==0 || i==2||i==6||i==8||i==18||i==20||i==24||i==26)// est un sommet coarse
            {
                int whichCoarseNode = -1; // what is the idx for this coarse node?
                switch(i)
                {
                case 0:
                    whichCoarseNode=0;
                    break;
                case 2:
                    whichCoarseNode=1;
                    break;
                case 6:
                    whichCoarseNode=3;
                    break;
                case 8:
                    whichCoarseNode=2;
                    break;
                case 18:
                    whichCoarseNode=4;
                    break;
                case 20:
                    whichCoarseNode=5;
                    break;
                case 24:
                    whichCoarseNode=7;
                    break;
                case 26:
                    whichCoarseNode=6;
                    break;
                }

                for( std::set<int>::iterator it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
                {
                    map_idxq_idxcutass[*it] = idxcutasscoarse;
                    map_idxq_coarse[*it] = whichCoarseNode;
                    map_idxcoarse_idxfine[ whichCoarseNode ].push_back( *it );
                    idxcutasscoarse++;

                    //mask
// 						  int localidx = map_idxq_idxass[*it];
// 						  mask.set( localidx*3  , whichCoarseNode*3   , 1);
// 						  mask.set( localidx*3+1, whichCoarseNode*3+1 , 1);
// 						  mask.set( localidx*3+2, whichCoarseNode*3+2 , 1);


// 						  helper::vector<int> coarsedepending; coarsedepending.push_back(whichCoarseNode);
// 						  map_mask[ *it ] = std::pair< helper::vector<int> ,unsigned >( coarsedepending, 0 );
                }
            }
            else
            {
                for( std::set<int>::iterator it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
                {
                    map_idxq_idxcutass[*it] = idxcutass;
                    map_idxq_coarse[*it] = -1;
                    idxcutass++;

// 						helper::vector<int> coarsedepending;

                    //mask
// 						int localidx = map_idxq_idxass[*it];
// 						for(int j=0;j<8;++j)
// 						{
// 							if( MIDDLE_INTERPOLATION[i][j] != 0 )
// 							{
// 								mask.set( localidx*3  , j*3   , 1);
// 								mask.set( localidx*3+1, j*3+1 , 1);
// 								mask.set( localidx*3+2, j*3+2 , 1);
//
// // 								coarsedepending.push_back(j);
// 							}
// 						}


// 						map_mask[ *it ] = std::pair< helper::vector<int> ,unsigned >( coarsedepending, MIDDLE_AXES[i] );
                }
            }
        }


// 			  cerr<<"map_idxq_idxcutass : "<<endl;
// 			  for(std::map<int,int>::iterator it = map_idxq_idxcutass.begin();it != map_idxq_idxcutass.end();++it)
// 			  {
// 				  cerr<<(*it).first<<" "<<(*it).second<<endl;
// 			  }


        NewMatMatrix Kg; // stiffness of contrained nodes
        Kg.resize(sizeass*3,idxcutasscoarse*3);
        NewMatMatrix  A; // [Kf -G] ==  Kf (stiffness of free nodes) with the constaints
        A.resize(sizeass*3,sizeass*3);
        NewMatMatrix  Ainv;

// 			  cerr<<"map_idxq_coarse : \n";
// 			  for( std::map<int,int>::iterator it = map_idxq_coarse.begin();it!= map_idxq_coarse.end();++it)
// 			  {
// 				  cerr<<(*it).second<<endl;
// 			  }

// 			  cerr<<"cutting :\n";
// 			  for ( int i=0;i<sizeass;++i)
        for( std::map<int,int>::iterator it = map_idxq_idxcutass.begin(); it!=map_idxq_idxcutass.end(); ++it)
        {
// 				  int col = map_idxq_idxcutass[i];
            int colcut = (*it).second;
            int colnoncut = map_idxq_idxass[(*it).first];

// 				  cerr<<(*it).first<<" "<<colcut<<endl;

            if( map_idxq_coarse[(*it).first] != -1 )
            {
                for(int lig=0; lig<sizeass; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            Kg.add( lig*3+m,colcut*3+n,assembledStiffnessStatic.element(lig*3+m,colnoncut*3+n) );
                }
            }
            else
            {
                for(int lig=0; lig<sizeass; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            A.add( lig*3+m,colcut*3+n,assembledStiffnessStatic.element(lig*3+m,colnoncut*3+n) );
                }
            }
        }


// 		  put -G entries into A
        int d=0;
        for(int i=0; i<8; ++i) // for all constrained nodes
        {
            for(unsigned j=0; j<map_idxcoarse_idxfine[i].size(); ++j)
            {
                A.add( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3   , (sizeass-idxcutasscoarse+d)*3   , -1.0);
                A.add( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3+1 , (sizeass-idxcutasscoarse+d)*3+1 , -1.0);
                A.add( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3+2 , (sizeass-idxcutasscoarse+d)*3+2 , -1.0);
                ++d;
            }
        }

// 			  cerr<<"A2 = ";
// 			  A.printMatlab( cerr );
// 			  cerr<<"Kg2 = ";
// 			  Kg.printMatlab( cerr );

        Ainv = A.i();

        NewMatMatrix  Ainvf;
        Ainv.getSubMatrix( 0,0, (sizeass-idxcutasscoarse)*3,sizeass*3,Ainvf);



        //// ajouter un H qui lie tous les coins superposés ensemble et n'en garder que 8 pour avoir un W 27x8
        NewMatMatrix H;
        H.resize( idxcutasscoarse*3, 8*3 );
        for(int i=0; i<8; ++i)
        {
            for(unsigned j=0; j<map_idxcoarse_idxfine[i].size(); ++j)
            {
// 					  cerr<<i<<" "<<j<<" "<<map_idxcoarse_idxfine[i]<<endl;
                H.set( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3  , i*3  ,1);
                H.set( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3+1, i*3+1,1);
                H.set( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3+2, i*3+2,1);
            }
        }

// 			  cerr<<"H = ";
// 			  H.printMatlab(cerr);
// 			  NewMatMatrix HKg2;
// 			  HKg2 = Kg*H;
// 			  cerr<<"HKg2 = ";
// 			  HKg2.printMatlab(cerr);




        NewMatMatrix  W;
        W = - Ainvf * Kg * H;

// 			  cerr<<"W"<<elementIndice<<"=";
// 			  W.printMatlab( cerr );
//
//
// 			  cerr<<"W : "<<W.rowSize()<<"x"<<W.colSize()<<endl;
//
//
        NewMatMatrix  WB;
        WB.resize(sizeass*3,8*3);
// 			  cerr<<"WB : "<<WB.rowSize()<<"x"<<WB.colSize()<<endl;


        for( std::map<int,int>::iterator it= map_idxq_coarse.begin(); it!=map_idxq_coarse.end(); ++it)
        {
            if( it->second != -1 )
            {
// 					  cerr<<it->first<<" "<<it->second<<endl;
                WB.add( map_idxq_idxass[it->first]*3  , it->second*3  , 1.0);
                WB.add( map_idxq_idxass[it->first]*3+1, it->second*3+1, 1.0);
                WB.add( map_idxq_idxass[it->first]*3+2, it->second*3+2, 1.0);
            }
            else
            {
                for(int j=0; j<8*3; ++j)
                {
                    WB.add( map_idxq_idxass[it->first]*3  ,j, W.element( map_idxq_idxcutass[it->first]*3  , j));
                    WB.add( map_idxq_idxass[it->first]*3+1,j, W.element( map_idxq_idxcutass[it->first]*3+1, j));
                    WB.add( map_idxq_idxass[it->first]*3+2,j, W.element( map_idxq_idxcutass[it->first]*3+2, j));
                }
            }
        }


// 			  cerr<<"WB2 = ";
// 			  WB.printMatlab( cerr );
//
//
//
// 			  cerr<<"mask = ";
// 			  mask.printMatlab( cerr );


        // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
        NewMatMatrix WBmeca;
        WBmeca.resize(sizeass*3,8*3);


        for(int i=0; i<27; ++i)
        {
            for( std::set<int>::iterator it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
            {
                int localidx = map_idxq_idxass[ *it ];

                int nbDependingCoarseNodes = 0;
                for(int j=0; j<8; ++j)
                {
                    if( MIDDLE_INTERPOLATION[i][j] )
                    {
                        ++nbDependingCoarseNodes;
                    }
                }


                if( nbDependingCoarseNodes==1 || nbDependingCoarseNodes==8 ) // fine node on a coarse node or in the middle of the coarse cube
                {
                    for(int j=0; j<8*3; ++j)
                    {
                        WBmeca.set( localidx*3  , j, WB.element(localidx*3  , j) ); // directly copy all
                        WBmeca.set( localidx*3+1, j, WB.element(localidx*3+1, j) );
                        WBmeca.set( localidx*3+2, j, WB.element(localidx*3+2, j) );
                    }
                }
                else if( nbDependingCoarseNodes==2 ) // fine node on an edge
                {
                    switch( MIDDLE_AXES[i] )
                    {
                    case 1: //x
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                WBmeca.set( localidx*3  , j*3, WB.element(localidx*3  , j*3) ); // copy just the right influence in the right axe
                                WBmeca.set( localidx*3+1, j*3+1, WB.element(localidx*3, j*3) );
                                WBmeca.set( localidx*3+2, j*3+2, WB.element(localidx*3, j*3) );
                            }
                        }
                        break;
                    case 2: //y
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                WBmeca.set( localidx*3  , j*3, WB.element(localidx*3+1  , j*3+1) );
                                WBmeca.set( localidx*3+1, j*3+1, WB.element(localidx*3+1, j*3+1) );
                                WBmeca.set( localidx*3+2, j*3+2, WB.element(localidx*3+1, j*3+1) );
                            }
                        }
                        break;
                    case 3: //z
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                WBmeca.set( localidx*3  , j*3, WB.element(localidx*3+2  , j*3+2) );
                                WBmeca.set( localidx*3+1, j*3+1, WB.element(localidx*3+2, j*3+2) );
                                WBmeca.set( localidx*3+2, j*3+2, WB.element(localidx*3+2, j*3+2) );
                            }
                        }
                        break;
                    }
                }
                else if( nbDependingCoarseNodes==4 ) // fine node on a face
                {
                    switch( MIDDLE_AXES[i] )
                    {
                    case 1: //x
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                Real coef = WB.element(localidx*3+1, j*3+1)+WB.element(localidx*3+2, j*3+2);
                                WBmeca.set( localidx*3  , j*3, coef );
                                WBmeca.set( localidx*3+1, j*3+1, coef );
                                WBmeca.set( localidx*3+2, j*3+2, coef );
                            }
                        }
                        break;
                    case 2: //y
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                Real coef = WB.element(localidx*3, j*3)+WB.element(localidx*3+2, j*3+2);
                                WBmeca.set( localidx*3  , j*3, coef );
                                WBmeca.set( localidx*3+1, j*3+1, coef );
                                WBmeca.set( localidx*3+2, j*3+2, coef );
                            }
                        }
                        break;
                    case 3: //z
                        for(int j=0; j<8; ++j)
                        {
                            if( MIDDLE_INTERPOLATION[i][j] )
                            {
                                Real coef = WB.element(localidx*3, j*3)+WB.element(localidx*3+1, j*3+1);
                                WBmeca.set( localidx*3  , j*3, coef );
                                WBmeca.set( localidx*3+1, j*3+1, coef );
                                WBmeca.set( localidx*3+2, j*3+2, coef );
                            }
                        }
                        break;
                    }
                }
            }
        }


        // normalize the coefficient to obtain sum(coefs)==1
        for(int i=0; i<sizeass*3; ++i)
        {
            Real sum = 0.0;
            for(int j=0; j<8*3; ++j)
            {
                sum += WBmeca.element(i,j);
            }
            for(int j=0; j<8*3; ++j)
            {
                WBmeca.set(i,j, WBmeca.element(i,j) / sum );
            }
        }


        NewMatMatrix Kc, Mc; // coarse stiffness
        Kc = WBmeca.t() * assembledStiffness * WBmeca;
        Mc = WBmeca.t() * assembledMass * WBmeca;





        for(int i=0; i<8*3; ++i)
            for(int j=0; j<8*3; ++j)
            {
                K[i][j]=Kc.element(i,j);
                M[i][j]=Mc.element(i,j);
            }

// 			  cerr<<"K"<<elementIndice<<"=";
// 			  printMatlab( cerr, K);cerr<<endl;
// 			  cerr<<"M"<<elementIndice<<"=";
// 			  printMatlab( cerr, M);cerr<<endl;




        if( !_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
        {
            WB = WBmeca;
        }


        for(int i=0 ; i < 8 ; ++i ) // finer places
        {
            for(unsigned j=0; j<finerChildrenRamificationOriginal[i].size(); ++j) // finer element
            {
                const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexa( finerChildrenRamificationOriginal[i][j] );
                for(int k=0 ; k < 8 ; ++k ) // fine nodes
                {
                    for( int l=0; l<8*3; ++l) // toutes les cols de W
                    {
                        _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]][k*3  ][l] = WB.element( map_idxq_idxass[ finehexa[k] ]*3   ,l);
                        _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]][k*3+1][l] = WB.element( map_idxq_idxass[ finehexa[k] ]*3+1 ,l);
                        _weights[this->_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]][k*3+2][l] = WB.element( map_idxq_idxass[ finehexa[k] ]*3+2 ,l);
                    }
                }
            }
        }

// 			  cerr<<"WBmeca =";
// 			  WBmeca.printMatlab(cerr);
// 			  cerr<<"WB =";
// 			  WB.printMatlab(cerr);


// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 9 ]*3   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 9 ]*3+1   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 9 ]*3+2   ,l)<<" ";
// 			  cerr<<endl;cerr<<endl;
//
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 9 ]*3   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 9 ]*3+1   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 9 ]*3+2   ,l)<<" ";
// 			  cerr<<endl;cerr<<endl;
//
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 16 ]*3   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 16 ]*3+1   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WB.element( map_idxq_idxass[ 16 ]*3+2   ,l)<<" ";
// 			  cerr<<endl;cerr<<endl;
//
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 16 ]*3   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 16 ]*3+1   ,l)<<" ";
// 			  cerr<<endl;
// 			  for( int l=0;l<8*3;++l) // toutes les cols de W
// 				  cerr<< WBmeca.element( map_idxq_idxass[ 16 ]*3+2   ,l)<<" ";
// 			  cerr<<endl;cerr<<endl;


    }


}




template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeFinalWeights( const Weight &W, const int coarseElementIndice, const int elementIndice,  int level)
{
// 		  for(int i=0;i<level*3;++i)cerr<<" ";
// 		  cerr<<"computeFinalWeights "<<elementIndice<<"  "<<level<<endl;

    if( elementIndice == -1 ) return;

    Weight A = _weights[ this->_nbVirtualFinerLevels.getValue()-level ][elementIndice]* W;

    if (level == this->_nbVirtualFinerLevels.getValue())
    {
// 			  if( elementIndice==2 )
// 			  {
// 				  cerr<<"COMPUTE_FINAL\n";
// 				  cerr<<this->_nbVirtualFinerLevels.getValue()-level<<endl;
// 				  printMatlab(cerr,_weights[0][2]);
// 				  printMatlab(cerr,W);
// 			  }

// 			  _weights[ this->_nbVirtualFinerLevels.getValue()-level ][elementIndice] = A;
        _finalWeights[ elementIndice ] = std::pair<int,Weight>(coarseElementIndice, A);
    }
    else
    {
        topology::SparseGridTopology* sparseGrid;

        sparseGrid = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level];

        helper::fixed_array<int,8> finerChildren = sparseGrid->_hierarchicalCubeMap[elementIndice];

        for(int i=0; i<8; ++i)
            computeFinalWeights( A, coarseElementIndice, finerChildren[i], level+1);
    }
}



template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::computeFinalWeightsRamification( const Weight &W, const int coarseElementIndice, const int elementIndice,  int level)
{
// 		  for(int i=0;i<level*3;++i)cerr<<" ";
// 		  cerr<<"computeFinalWeights "<<elementIndice<<"  "<<level<<endl;

    if( elementIndice == -1 ) return;

    Weight A = _weights[ this->_nbVirtualFinerLevels.getValue()-level ][elementIndice]* W;

    if (level == this->_nbVirtualFinerLevels.getValue())
    {
// 			  if( elementIndice==2 )
// 			  {
// 				  cerr<<"COMPUTE_FINAL\n";
// 				  cerr<<this->_nbVirtualFinerLevels.getValue()-level<<endl;
// 				  printMatlab(cerr,_weights[0][2]);
// 				  printMatlab(cerr,W);
// 			  }

// 			  _weights[ this->_nbVirtualFinerLevels.getValue()-level ][elementIndice] = A;
        _finalWeights[ elementIndice ] = std::pair<int,Weight>(coarseElementIndice, A);
    }
    else
    {
        topology::SparseGridRamificationTopology* sparseGrid;

        sparseGrid = dynamic_cast< topology::SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level]);

        helper::fixed_array<helper::vector<int>,8 >& finerChildrenRamification = sparseGrid->_hierarchicalCubeMapRamification[ elementIndice ];

        for(int w=0; w<8; ++w)
            for(unsigned v=0; v<finerChildrenRamification[w].size(); ++v)
                computeFinalWeights( A, coarseElementIndice, finerChildrenRamification[w][v], level+1);
    }
}



template<class T>
void HomogenizedHexahedronFEMForceFieldAndMass<T>::draw()
{


    if (!this->getContext()->getShowForceFields()) return;
    if (!this->mstate) return;
    if (this->getContext()->getShowWireFrame()) return;



    const VecCoord& x = *this->mstate->getX();


// 		  glDisable(GL_LIGHTING);

    glColor3f(0.9, 0.9, 0.2);

    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);


    glLineWidth( 3 );
    glBegin(GL_LINES);
    for( SparseGridTopology::SeqEdges::const_iterator it = this->_sparseGrid->getEdges().begin() ; it != this->_sparseGrid->getEdges().end(); ++it)
    {
// 			  helper::gl::drawCylinder( x[(*it)[0]], x[(*it)[1]], _drawSize );

        helper::gl::glVertexT( x[(*it)[0]] );
        helper::gl::glVertexT( x[(*it)[1]] );
    }
    glEnd();


    glColor3f(0.95, 0.95, 0.7);
    for(unsigned i=0; i<x.size(); ++i)
    {
        helper::gl::drawSphere( x[i], _drawSize*1.5 );
    }

    glDisable(GL_LIGHTING);
    glDisable(GL_COLOR_MATERIAL);

}


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif

