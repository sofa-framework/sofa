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

#include <sofa/component/solidmechanics/fem/nonuniform/HexahedronCompositeFEMForceFieldAndMass.h>
#include <sofa/component/solidmechanics/fem/nonuniform/NonUniformHexahedronFEMForceFieldAndMass.inl>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/component/topology/container/grid/SparseGridRamificationTopology.h>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/LU>

namespace sofa::component::solidmechanics::fem::nonuniform
{

using topology::container::grid::SparseGridTopology;
using EigenMatrix = Eigen::MatrixXd;

template <class DataTypes>
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::FineHexa_FineNode_IndiceForAssembling[8][8]=
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





template <class DataTypes>
const bool HexahedronCompositeFEMForceFieldAndMass<DataTypes>::IS_CONSTRAINED_27[27] =
{
    1,0,1, 0,0,0, 1,0,1, //tranche devant
    0,0,0, 0,0,0, 0,0,0,  //milieu
    1,0,1, 0,0,0, 1,0,1
};


template <class DataTypes>
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::FineHexa_FineNode_IndiceForCutAssembling_27[27]=
{0,    0,    1,    1,    2,    3,    3,    4,    2,    5,    6,    7    ,8,    9,    10,    11,    12,    13,    4    ,14,    5,    15,    16,    17,    7,    18,    6};



template <class DataTypes>
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::CoarseToFine[8]=
{ 0, 2, 8, 6, 18, 20, 26, 24 };



template <class DataTypes>
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK[27*3][8*3]=
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
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK_CROSSED[27*3][8*3]=
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
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::WEIGHT_MASK_CROSSED_DIFF[27*3][8*3]=
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
const float HexahedronCompositeFEMForceFieldAndMass<DataTypes>::MIDDLE_INTERPOLATION[27][8]=
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
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::MIDDLE_AXES[27]=
{
    0,       1,              0,
    2,          3,                2,
    0,        1,                0,

    3,            2,        3,
    1,    0,             1,
    3,            2,        3,

    0,       1,              0,
    2,          3,        2,
    0,       1,              0
};



template <class DataTypes>
const int HexahedronCompositeFEMForceFieldAndMass<DataTypes>::FINE_ELEM_IN_COARSE_IN_ASS_FRAME[8][8]=
{
    {0,1,4,3,9,10,13,12},
    {1,2,5,4,10,11,14,13},
    {3,4,7,6,12,13,16,15},
    {4,5,8,7,13,14,17,16},
    {9,10,13,12,18,19,22,21},
    {10,11,14,13,19,20,23,22},
    {12,13,16,15,21,22,25,24},
    {13,14,17,16,22,23,26,25}
};


template <class DataTypes>
const float HexahedronCompositeFEMForceFieldAndMass<DataTypes>::RIGID_STIFFNESS[8*3][8*3]=
{
    {(float)2.26667e+11,(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)2.125e+10},
    {(float)4.25e+10,(float)2.26667e+11,(float)4.25e+10,(float)4.25e+10,(float)2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)-4.25e+10},
    {(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)-4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-7.08333e+10},
    {(float)-5.66667e+10,(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)2.125e+10},
    {(float)-4.25e+10,(float)2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.26667e+11,(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-5.66667e+10,(float)-2.125e+10},
    {(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)-4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)-2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-2.125e+10,(float)-5.66667e+10},
    {(float)-7.08333e+10,(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)-2.125e+10,(float)2.26667e+11,(float)4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)4.25e+10},
    {(float)-4.25e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-2.83333e+10,(float)-2.125e+10},
    {(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)4.25e+10,(float)2.125e+10,(float)-7.08333e+10},
    {(float)2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)4.25e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)4.25e+10},
    {(float)4.25e+10,(float)-5.66667e+10,(float)4.25e+10,(float)4.25e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)2.83333e+10,(float)-4.25e+10},
    {(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)4.25e+10,(float)-5.66667e+10},
    {(float)2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)2.26667e+11,(float)4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)-2.125e+10},
    {(float)2.125e+10,(float)2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)4.25e+10},
    {(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.83333e+10},
    {(float)-7.08333e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)4.25e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)4.25e+10,(float)2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)-2.125e+10},
    {(float)-2.125e+10,(float)-2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)4.25e+10,(float)4.25e+10,(float)-7.08333e+10,(float)2.125e+10},
    {(float)-4.25e+10,(float)2.125e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-5.66667e+10,(float)4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)-4.25e+10,(float)2.26667e+11,(float)2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)-2.83333e+10},
    {(float)-5.66667e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)2.125e+10,(float)2.83333e+10,(float)2.125e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)2.125e+10,(float)2.26667e+11,(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)-4.25e+10},
    {(float)-2.125e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)4.25e+10,(float)4.25e+10,(float)2.83333e+10,(float)2.125e+10},
    {(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)4.25e+10,(float)2.125e+10,(float)2.83333e+10},
    {(float)-2.83333e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)2.125e+10,(float)2.125e+10,(float)-7.08333e+10,(float)2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)-4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)2.125e+10,(float)-5.66667e+10,(float)4.25e+10,(float)4.25e+10,(float)2.26667e+11,(float)-4.25e+10,(float)-4.25e+10},
    {(float)2.125e+10,(float)-7.08333e+10,(float)-4.25e+10,(float)2.125e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)-2.125e+10,(float)-2.83333e+10,(float)2.125e+10,(float)-2.125e+10,(float)2.83333e+10,(float)4.25e+10,(float)4.25e+10,(float)-5.66667e+10,(float)-4.25e+10,(float)4.25e+10,(float)-7.08333e+10,(float)-2.125e+10,(float)-4.25e+10,(float)2.83333e+10,(float)2.125e+10,(float)-4.25e+10,(float)2.26667e+11,(float)4.25e+10},
    {(float)2.125e+10,(float)-4.25e+10,(float)-7.08333e+10,(float)2.125e+10,(float)-2.125e+10,(float)-5.66667e+10,(float)4.25e+10,(float)-2.125e+10,(float)-7.08333e+10,(float)4.25e+10,(float)-4.25e+10,(float)-5.66667e+10,(float)-2.125e+10,(float)4.25e+10,(float)2.83333e+10,(float)-2.125e+10,(float)2.125e+10,(float)-2.83333e+10,(float)-4.25e+10,(float)2.125e+10,(float)2.83333e+10,(float)-4.25e+10,(float)4.25e+10,(float)2.26667e+11}
};

template <class DataTypes>
void HexahedronCompositeFEMForceFieldAndMass<DataTypes>::init()
{

    // init topology, virtual levels, calls computeMechanicalMatricesByCondensation, handles masses
    NonUniformHexahedronFEMForceFieldAndMassT::init();


    if(d_drawSize.getValue()==-1 && this->_sparseGrid != nullptr)
        d_drawSize.setValue( (float)((this->_sparseGrid->getMax()[0]-this->_sparseGrid->getMin()[0]) * .004f) );

}

template <class DataTypes>
void HexahedronCompositeFEMForceFieldAndMass<DataTypes>::reinit()
{
    msg_warning() << "Composite mechanical properties can't be updated, changes on mechanical properties (young, poisson, density) are not taken into account.";
    if(d_drawSize.getValue()==-1)
        d_drawSize.setValue( (float)((this->_sparseGrid->getMax()[0]-this->_sparseGrid->getMin()[0]) * .004f) );
}

/////////////////////////////////////////////////
/////////////////////////////////////////////////
/////////////////////////////////////////////////


template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeMechanicalMatricesByCondensation( )
{
    if( this->d_nbVirtualFinerLevels.getValue() == 0 )
    {
        for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
        {
            //Get the 8 indices of the coarser Hexa
            const auto& points = this->_sparseGrid->getHexahedra()[i];
            //Get the 8 points of the coarser Hexa
            type::fixed_array<Coord,8> nodes;

            for (unsigned int k=0; k<8; ++k) nodes[k] =  this->_sparseGrid->getPointPos(points[k]);


            //       //given an elementIndice, find the 8 others from the sparse grid
            //       //compute MaterialStiffness
            MaterialStiffness material;
            this->computeMaterialStiffness(material, this->getYoungModulusInElement(i), this->getPoissonRatioInElement(i));


            HexahedronFEMForceFieldAndMassT::computeElementStiffness((*this->d_elementStiffnesses.beginEdit())[i], material, nodes, i, this->_sparseGrid->getStiffnessCoef(i )); // classical stiffness

            HexahedronFEMForceFieldAndMassT::computeElementMass((*this->d_elementMasses.beginEdit())[i],nodes,i,this->_sparseGrid->getMassCoef( i ));
        }
        return;
    }





    _weights.resize( this->d_nbVirtualFinerLevels.getValue() );
    const int finestLevel = this->_sparseGrid->getNbVirtualFinerLevels()-this->d_nbVirtualFinerLevels.getValue();

    for(int i=0; i<this->d_nbVirtualFinerLevels.getValue(); ++i)
    {
        _weights[i].resize( this->_sparseGrid->_virtualFinerLevels[finestLevel+i]->getNbHexahedra() );
    }

    _finalWeights.resize( _weights[0].size() );


    if( d_finestToCoarse.getValue() )
        for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
            computeMechanicalMatricesDirectlyFromTheFinestToCoarse((*this->d_elementStiffnesses.beginEdit())[i], (*this->d_elementMasses.beginEdit())[i], i );
    else
    {
        auto* sparseGridRamification = dynamic_cast<component::topology::container::grid::SparseGridRamificationTopology*>( this->_sparseGrid );
        if( d_useRamification.getValue() && sparseGridRamification )
        {
            for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
                computeMechanicalMatricesRecursivelyWithRamifications((*this->d_elementStiffnesses.beginEdit())[i], (*this->d_elementMasses.beginEdit())[i], i, 0 );


            for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
            {
                Weight A; A.identity();

                auto& finerChildrenRamification = sparseGridRamification->_hierarchicalCubeMapRamification[ i ];

                for(int w=0; w<8; ++w)
                    for(unsigned v=0; v<finerChildrenRamification[w].size(); ++v)
                        computeFinalWeightsRamification( A, i, finerChildrenRamification[w][v], 1 );
            }

        }
        else
        {
            for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
                computeMechanicalMatricesRecursively((*this->d_elementStiffnesses.beginEdit())[i], (*this->d_elementMasses.beginEdit())[i], i, 0 );

            for (unsigned int i=0; i<this->getIndexedElements()->size(); ++i)
            {
                Weight A; A.identity();

                auto finerChildren = this->_sparseGrid->_hierarchicalCubeMap[i];

                for(int w=0; w<8; ++w)
                    computeFinalWeights( A, i, finerChildren[w], 1 );
            }
        }


    }


    _weights.resize(0);

}



template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeMechanicalMatricesDirectlyFromTheFinestToCoarse( ElementStiffness &K, ElementMass &M, const Index elementIndice)
{
    type::vector<Index> finestChildren;

    //find them
    findFinestChildren( finestChildren, elementIndice );


    SparseGridTopology::SPtr finestSparseGrid = this->_sparseGrid->_virtualFinerLevels[ this->_sparseGrid->getNbVirtualFinerLevels()-this->d_nbVirtualFinerLevels.getValue() ];

    dmsg_info()<<"finestChildren.size() : "<<finestChildren.size()<<msgendl;
    dmsg_info()<<"finestSparseGrid->getNbHexahedra() : "<<finestSparseGrid->getNbHexahedra()<<msgendl;

    std::size_t sizeass=2;
    for(int i=0; i<this->d_nbVirtualFinerLevels.getValue(); ++i)
        sizeass = (sizeass-1)*2+1;
    sizeass = sizeass*sizeass*sizeass;

    EigenMatrix assembledStiffness = EigenMatrix::Zero(sizeass * 3, sizeass * 3);
    EigenMatrix assembledMass = EigenMatrix::Zero(sizeass * 3, sizeass * 3);
    
    dmsg_info()<<assembledStiffness.rows()<<"x"<<assembledStiffness.cols()<<msgendl;

    type::vector<ElementStiffness> finestStiffnesses(finestChildren.size());
    type::vector<ElementMass> finestMasses(finestChildren.size());


    std::map<Index, Index> map_idxq_idxass; // map a fine point idx to a assembly (local) idx
    Index idxass = 0;



    // compute the classical mechanical matrices at the finest level
    for(unsigned i=0 ; i < finestChildren.size() ; ++i )
    {
        this->computeClassicalMechanicalMatrices(finestStiffnesses[i],finestMasses[i],finestChildren[i],this->_sparseGrid->getNbVirtualFinerLevels()-this->d_nbVirtualFinerLevels.getValue());

        const SparseGridTopology::Hexa& hexa = finestSparseGrid->getHexahedron( finestChildren[i] );


        for(int w=0; w<8; ++w) // sommets
        {
            // idx for assembly
            if( !map_idxq_idxass[ hexa[w] ] )
            {
                map_idxq_idxass[ hexa[w] ] = /*FineHexa_FineNode_IndiceForAssembling(i,w);*/idxass;
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
                        assembledStiffness(v1 * 3 + m, v2 * 3 + n) += finestStiffnesses[i](j*3+m,k*3+n);
                        assembledMass(v1 * 3 + m, v2 * 3 + n) += finestMasses[i](j*3+m,k*3+n);
                    }
            }
        }
    }




    std::map<Index, Index> map_idxq_idxcutass; // map a fine point idx to a the cut assembly (local) idx
    Index idxcutass = 0;
    std::map<Index,bool> map_idxq_coarse;
    type::fixed_array<Index,8> map_idxcoarse_idxfine;
    const SparseGridTopology::Hexa& coarsehexa = this->_sparseGrid->getHexahedron( elementIndice );

    for(Size i=0; i<sizeass; ++i)
    {
        for( auto it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end(); ++it)
            if( (*it).second==i)
            {
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



    EigenMatrix Kg = EigenMatrix::Zero(sizeass * 3, 8 * 3); // stiffness of contrained nodes
    EigenMatrix A = EigenMatrix::Zero(sizeass * 3, sizeass * 3); // [Kf -G] ==  Kf (stiffness of free nodes) with the constraints

    for ( std::size_t i=0; i<sizeass; ++i)
    {
        int col = map_idxq_idxcutass[i];

        if( map_idxq_coarse[i] )
        {
            for(Size lig=0; lig<sizeass; ++lig)
            {
                for(int m=0; m<3; ++m)
                    for(int n=0; n<3; ++n)
                        Kg(lig * 3 + m, col * 3 + n) += assembledStiffness(lig*3+m,i*3+n);
            }
        }
        else
        {
            for(Size lig=0; lig<sizeass; ++lig)
            {
                for(int m=0; m<3; ++m)
                    for(int n=0; n<3; ++n)
                        A(lig * 3 + m, col * 3 + n) += assembledStiffness(lig*3+m,i*3+n);
            }
        }
    }

    // 		  put -G entries into A
    for(int i=0; i<8; ++i) // for all constrained nodes
    {
        A(map_idxcoarse_idxfine[i] * 3, (sizeass - 8 + i) * 3) +=  -1.0;
        A(map_idxcoarse_idxfine[i] * 3 + 1, (sizeass - 8 + i) * 3 + 1) +=  -1.0;
        A(map_idxcoarse_idxfine[i] * 3 + 2, (sizeass - 8 + i) * 3 + 2) +=  -1.0;
    }

    EigenMatrix Ainv = A.inverse();
    EigenMatrix Ainvf = Ainv.block( 0,0, (sizeass-8)*3,sizeass*3);
    EigenMatrix W = - Ainvf * Kg;
    EigenMatrix WB = EigenMatrix::Zero(sizeass * 3, 8 * 3);

    for(Size i=0; i<sizeass*3; ++i)
    {
        int idx = i/3;
        int mod = i%3;
        if( map_idxq_coarse[idx] )
            WB(i, map_idxq_idxcutass[idx] * 3 + mod) += 1.0;
        else
            for(int j=0; j<8*3; ++j)
            {
                WB(i, j) += W( map_idxq_idxcutass[idx]*3+mod, j);
            }
    }


    EigenMatrix mask = EigenMatrix::Zero(sizeass * 3, 8 * 3);

    Coord a = this->_sparseGrid->getPointPos(coarsehexa[0]);
    Coord b = this->_sparseGrid->getPointPos(coarsehexa[6]);
    Coord dx( b[0]-a[0],0,0),dy( 0,b[1]-a[1],0), dz( 0,0,b[2]-a[2]);
    Coord inv_d2( 1.0f/(dx*dx),1.0f/(dy*dy),1.0f/(dz*dz) );
    for( auto it = map_idxq_idxass.begin(); it!=map_idxq_idxass.end(); ++it)
    {
        Index localidx = (*it).second; // indice du noeud fin dans l'assemblage


        if( map_idxq_coarse[ (*it).second ] )
        {
            Index localcoarseidx = map_idxq_idxcutass[ (*it).second ];
            mask( localidx*3  , localcoarseidx*3   ) = 1;
            mask( localidx*3+1, localcoarseidx*3+1 ) = 1;
            mask( localidx*3+2, localcoarseidx*3+2 ) = 1;
        }
        else
        {

            // find barycentric coord
            Coord p = finestSparseGrid->getPointPos( (*it).first ) - a;

            Real fx = p*dx*inv_d2[0];
            Real fy = p*dy*inv_d2[1];
            Real fz = p*dz*inv_d2[2];


            type::fixed_array<Real,8> baryCoefs;
            baryCoefs[0] = (1-fx) * (1-fy) * (1-fz);
            baryCoefs[1] = fx * (1-fy) * (1-fz);
            baryCoefs[2] = fx * (fy) * (1-fz);
            baryCoefs[3] = (1-fx) * (fy) * (1-fz);
            baryCoefs[4] = (1-fx) * (1-fy) * (fz);
            baryCoefs[5] = fx * (1-fy) * (fz);
            baryCoefs[6] = fx * (fy) * (fz);
            baryCoefs[7] = (1-fx) * (fy) * fz;

            for(int i=0; i<8; ++i)
            {
                if( baryCoefs[i]>1.0e-5 )
                {
                    mask( localidx*3  , i*3  ) = 1;
                    mask( localidx*3+1, i*3+1) = 1;
                    mask( localidx*3+2, i*3+2) = 1;
                }
            }
        }
    }

    // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
    EigenMatrix WBmeca = EigenMatrix::Zero(sizeass * 3, 8 * 3);
    for(Size i=0; i<sizeass*3; ++i)
    {
        for(Size j=0; j<8*3; ++j)
        {
            if( mask(i,j) /*WEIGHT_MASK(i,j)*/ )
                WBmeca(i,j) = WB(i,j);
        }
    }

    dmsg_info()<<"WBmeca brut : "<<WBmeca<<msgendl;

    // normalize the coefficient to obtain sum(coefs)==1
    for(Size i=0; i<sizeass*3; ++i)
    {
        SReal sum = 0.0;
        for(Size j=0; j<8*3; ++j)
        {
            sum += WBmeca(i,j);
        }
        for(Size j=0; j<8*3; ++j)
        {
            WBmeca(i,j) = WBmeca(i,j) / sum ;
        }
    }
    dmsg_info()<<"WBmeca normalized : "<<WBmeca<<msgendl;
    EigenMatrix Kc, Mc; // coarse stiffness
    Kc = WBmeca.transpose() * assembledStiffness * WBmeca;
    Mc = WBmeca.transpose() * assembledMass * WBmeca;
    for(int i=0; i<8*3; ++i)
        for(int j=0; j<8*3; ++j)
        {
            K(i,j)=(Real)Kc(i,j);
            M(i,j)=(Real)Mc(i,j);
        }
    if( !d_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
    {
        WB = WBmeca;
    }

    for(unsigned i=0 ; i < finestChildren.size() ; ++i )
    {
        const SparseGridTopology::Hexa& hexa = finestSparseGrid->getHexahedron( finestChildren[i] );
        for(int j=0; j<8; ++j)
        {
            for( int k=0; k<8*3; ++k)
            {
                _finalWeights[finestChildren[i]].second(j*3  ,k) = (Real)WB( map_idxq_idxass[ hexa[j] ]*3   ,k);
                _finalWeights[finestChildren[i]].second(j*3+1,k) = (Real)WB( map_idxq_idxass[ hexa[j] ]*3+1 ,k);
                _finalWeights[finestChildren[i]].second(j*3+2,k) = (Real)WB( map_idxq_idxass[ hexa[j] ]*3+2 ,k);
            }
        }
        _finalWeights[finestChildren[i]].first = elementIndice;
    }

}





template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::findFinestChildren( type::vector<Index>& finestChildren, const Index elementIndice, int level)
{
    if (level == this->d_nbVirtualFinerLevels.getValue())
    {
        finestChildren.push_back( elementIndice );
    }
    else
    {
        type::fixed_array<Index,8> finerChildren;
        if (level == 0)
        {
            finerChildren = this->_sparseGrid->_hierarchicalCubeMap[elementIndice];
        }
        else
        {
            finerChildren = this->_sparseGrid->_virtualFinerLevels[this->d_nbVirtualFinerLevels.getValue()-level]->_hierarchicalCubeMap[elementIndice];
        }

        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            findFinestChildren( finestChildren, finerChildren[i], level+1 );
        }
    }
}



template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeMechanicalMatricesRecursively( ElementStiffness &K, ElementMass &M, const Index elementIndice,  int level)
{
    using namespace sofa::defaulttype;

    if (level == this->d_nbVirtualFinerLevels.getValue())
    {
        this->computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
    }
    else
    {
        type::fixed_array<Index,8> finerChildren;

        topology::container::grid::SparseGridTopology::SPtr sparseGrid, finerSparseGrid;

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

        type::fixed_array<ElementStiffness,8> finerK;
        type::fixed_array<ElementMass,8> finerM;


        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if (finerChildren[i] != sofa::InvalidID)
            {
                computeMechanicalMatricesRecursively(finerK[i], finerM[i], finerChildren[i], level+1);
            }
        }

        // assemble the matrix of 8 child
        type::Mat<27*3, 27*3, Real> assembledStiffness;
        type::Mat<27*3, 27*3, Real> assembledStiffnessWithRigidVoid;
        type::Mat<27*3, 27*3, Real> assembledMass;


        for ( int i=0; i<8; ++i) //for 8 virtual finer element
        {
            if( finerChildren[i]!= sofa::InvalidID)
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
                                assembledStiffness( v1*3+m , v2*3+n ) += finerK[i](j*3+m,k*3+n);
                                assembledStiffnessWithRigidVoid( v1*3+m , v2*3+n ) += finerK[i](j*3+m,k*3+n);
                                assembledMass( v1*3+m , v2*3+n ) += finerM[i](j*3+m,k*3+n);
                            }
                    }
                }
            }
            else
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
                                assembledStiffnessWithRigidVoid( v1*3+m , v2*3+n ) += RIGID_STIFFNESS[j*3+m][k*3+n];
                            }
                    }
                }
            }
        }


        type::Mat<27*3, 8*3, Real> Kg; // stiffness of contrained nodes
        type::Mat<27*3, 27*3, Real> A; // [Kf -G]  Kf (stiffness of free nodes) with the constraints
        type::Mat<27*3, 27*3, Real> Ainv;


        for ( int i=0; i<27; ++i)
        {
            int col = FineHexa_FineNode_IndiceForCutAssembling_27[i];

            if( IS_CONSTRAINED_27[i] )
            {
                for(int lig=0; lig<27; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            Kg( lig*3+m , col*3+n ) = assembledStiffnessWithRigidVoid(lig*3+m,i*3+n);
                }
            }
            else
            {
                for(int lig=0; lig<27; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            A( lig*3+m , col*3+n ) = assembledStiffnessWithRigidVoid(lig*3+m,i*3+n);
                }
            }

        }


        // put -G entries into A
        for(int i=0; i<8; ++i) // for all constrained nodes
        {
            A( CoarseToFine[i]*3   , (27-8+i)*3   ) = -1.0;
            A( CoarseToFine[i]*3+1 , (27-8+i)*3+1 ) = -1.0;
            A( CoarseToFine[i]*3+2 , (27-8+i)*3+2 ) = -1.0;
        }
        const bool canInvert = Ainv.invert(A);
        assert(canInvert);
        SOFA_UNUSED(canInvert);
        type::Mat<(27-8)*3, 27*3, Real> Ainvf;
        for(int i=0; i<27-8; ++i)
        {
            for(int m=0; m<3; ++m)
                Ainvf(i*3+m) = - Ainv.line( i*3+m );
        }

        type::Mat<(27-8)*3, 8*3, Real> W;
        W = Ainvf * Kg;


        type::Mat<27*3, 8*3, Real> WB;
        for(int i=0; i<27*3; ++i)
        {
            int idx = i/3;
            int mod = i%3;
            if( IS_CONSTRAINED_27[idx] )
                WB( i )[ FineHexa_FineNode_IndiceForCutAssembling_27[idx]*3+mod ] = 1.0;
            else
                WB( i ) = W( FineHexa_FineNode_IndiceForCutAssembling_27[idx]*3+mod );
        }

        // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
        type::Mat<27*3, 8*3, Real> WBmeca;
        for(int i=0; i<27*3; ++i)
        {
            for(int j=0; j<8*3; ++j)
            {
                if( WEIGHT_MASK[i][j] )
                    WBmeca(i,j)=WB(i,j);
            }
        }


        type::vector<Real> sum_wbmeca(27*3,0);
        // normalize the coefficient to obtain sum(coefs)==1
        for(int i=0; i<27*3; ++i)
        {
            for(int j=0; j<8*3; ++j)
            {
                sum_wbmeca[i] += WBmeca(i,j);
            }
            for(int j=0; j<8*3; ++j)
            {
                WBmeca(i,j) /= sum_wbmeca[i];
            }
        }

        K = WBmeca.multTranspose( assembledStiffness * WBmeca );

        M = WBmeca.multTranspose( assembledMass * WBmeca );

        if( !d_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
        {
            WB = WBmeca;
        }
        else
        {
            const auto poissonRatio = this->getPoissonRatioInElement(0);
            for(int i=0; i<27*3; ++i)
            {
                for(int j=0; j<8*3; ++j)
                {
                    if( !WEIGHT_MASK_CROSSED_DIFF[i][j] )
                        WB(i,j) = WBmeca(i,j);
                    else
                    {
                        WB(i,j) = (Real)(WB(i,j) / fabs(WB(i,j)) * WEIGHT_MASK_CROSSED_DIFF[i][j] * poissonRatio * .3);
                    }
                }
            }
        }


        for(int elem=0; elem<8; ++elem)
        {
            if( finerChildren[elem] != sofa::InvalidID)
            {
                for(int i=0; i<8; ++i)
                {
                    _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]](i*3  ) = WB ( FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3  );
                    _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]](i*3+1) = WB ( FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+1);
                    _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildren[elem]](i*3+2) = WB ( FineHexa_FineNode_IndiceForAssembling[ elem ][ i ]*3+2);
                }
            }
        }
    }
}


template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeMechanicalMatricesRecursivelyWithRamifications( ElementStiffness &K, ElementMass &M, const Index elementIndice,  int level)
{
    using namespace sofa::defaulttype;
    using namespace sofa::component::topology::container::grid;

    if (level == this->d_nbVirtualFinerLevels.getValue())
    {
        this->computeClassicalMechanicalMatrices(K,M,elementIndice,this->_sparseGrid->getNbVirtualFinerLevels()-level);
    }
    else
    {
        component::topology::container::grid::SparseGridRamificationTopology* sparseGrid,*finerSparseGrid;

        if (level == 0)
        {
            sparseGrid = dynamic_cast<SparseGridRamificationTopology*>(this->_sparseGrid);
            finerSparseGrid = dynamic_cast<SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-1].get());
        }
        else
        {
            sparseGrid = dynamic_cast<SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level].get());
            finerSparseGrid = dynamic_cast<SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level-1].get());
        }

        // trouver les finer elements par ramification
        auto& finerChildrenRamificationOriginal = sparseGrid->_hierarchicalCubeMapRamification[ elementIndice ];

        type::fixed_array<type::vector<ElementStiffness>,8> finerK;
        type::fixed_array<type::vector<ElementMass>,8> finerM;


        const SparseGridTopology::Hexa& coarsehexa = sparseGrid->getHexahedron( elementIndice );


        type::fixed_array< Coord, 27 > finePositions; // coord of each fine positions
        for(int i=0; i<27; ++i)
        {
            for(int j=0; j<8; ++j)
            {
                finePositions[i] += sparseGrid->getPointPos( coarsehexa[j] ) * MIDDLE_INTERPOLATION[i][j];
            }
        }


        type::fixed_array< std::set<Index>, 27 > fineNodesPerPositions; // list of fine nodes at each fine positions
        for ( int i=0; i<8; ++i) //for 8 virtual finer element positions
        {
            finerK[i].resize( finerChildrenRamificationOriginal[i].size() );
            finerM[i].resize( finerChildrenRamificationOriginal[i].size() );

            for(unsigned j=0; j<finerChildrenRamificationOriginal[i].size(); ++j) // for all finer elements
            {
                computeMechanicalMatricesRecursivelyWithRamifications(finerK[i][j], finerM[i][j], finerChildrenRamificationOriginal[i][j], level+1);

                const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexahedron( finerChildrenRamificationOriginal[i][j] );
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
        Index fictifidx = sofa::InvalidID;
        for(int i=0; i<27; ++i)
        {
            if( fineNodesPerPositions[i].empty() ) // pas de points ici
            {
                fineNodesPerPositions[i].insert(fictifidx);
                --fictifidx;
            }
        }


        type::fixed_array<type::vector<type::fixed_array<Index,8 > >,8 > finerChildrenRamification; // listes des hexahedra � chaque position, avec des indices fictifs pour les vides
        type::fixed_array<type::vector<bool>,8 > isFinerChildrenVirtual; // a boolean, true if ficitf, only created for void
        for ( int i=0; i<8; ++i) //for 8 virtual finer element positions
        {
            if( finerChildrenRamificationOriginal[i].empty() ) // vide
            {
                // construire un element fictif
                type::fixed_array<Index,8 > fictifelem;
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
                    const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexahedron( finerChildrenRamificationOriginal[i][j] );
                    type::fixed_array<Index,8 > elem;
                    for(int k=0; k<8; ++k) // fine nodes
                    {
                        elem[k] = finehexa[k];
                    }
                    finerChildrenRamification[i].push_back(elem);
                    isFinerChildrenVirtual[i].push_back( 0 );
                }
            }
        }
        std::map<Index,Index> map_idxq_idxass; // map a fine point idx to a assembly (local) idx
        int idxass = 0;


        for(int i=0; i<27; ++i)
        {
            for( auto it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
            {
                map_idxq_idxass[*it] = idxass;
                idxass++;
            }
        }

        int sizeass = idxass; // taille de l'assemblage i.e., le nombre de noeuds fins
        EigenMatrix assembledStiffness = EigenMatrix::Zero(sizeass * 3, sizeass * 3);
        EigenMatrix assembledStiffnessStatic = EigenMatrix::Zero(sizeass * 3, sizeass * 3);
        EigenMatrix assembledMass = EigenMatrix::Zero(sizeass * 3, sizeass * 3);

        for(int i=0 ; i < 8 ; ++i ) // finer places
        {
            for( unsigned c=0; c<finerChildrenRamification[i].size(); ++c)
            {

                auto& finehexa = finerChildrenRamification[i][c];

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
                                    assembledStiffnessStatic( v1*3+m, v2*3+n) += RIGID_STIFFNESS[j*3+m][k*3+n] ;
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
                                    assembledStiffness( v1*3+m, v2*3+n) += finerK[i][c](j*3+m, k*3+n);
                                    assembledStiffnessStatic( v1*3+m, v2*3+n) += finerK[i][c](j*3+m, k*3+n);
                                    assembledMass( v1*3+m, v2*3+n) += finerM[i][c](j*3+m, k*3+n);
                                }
                        }
                    }
                }
            }
        }

        std::map<Index, Index> map_idxq_idxcutass; // map a fine point idx to a the cut assembly (local) idx
        Index idxcutass = 0,idxcutasscoarse = 0;
        std::map<Index, Index> map_idxq_coarse; // a fine idx -> sofa::InvalidID->non coarse, x-> idx coarse node
        type::fixed_array<type::vector<Index> ,8> map_idxcoarse_idxfine;

        EigenMatrix mask = EigenMatrix::Zero(sizeass * 3, 8 * 3);
        for(int i=0; i<27; ++i)
        {
            if( i==0 || i==2||i==6||i==8||i==18||i==20||i==24||i==26)// est un sommet coarse
            {
                int whichCoarseNode = sofa::InvalidID; // what is the idx for this coarse node?
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

                for( auto it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
                {
                    map_idxq_idxcutass[*it] = idxcutasscoarse;
                    map_idxq_coarse[*it] = whichCoarseNode;
                    map_idxcoarse_idxfine[ whichCoarseNode ].push_back( *it );
                    idxcutasscoarse++;

                    //mask
                    int localidx = map_idxq_idxass[*it];
                    mask( localidx*3  , whichCoarseNode*3  ) = 1;
                    mask( localidx*3+1, whichCoarseNode*3+1) = 1;
                    mask( localidx*3+2, whichCoarseNode*3+2) = 1;
                }
            }
            else
            {
                for( auto it = fineNodesPerPositions[i].begin() ; it != fineNodesPerPositions[i].end() ; ++it )
                {
                    map_idxq_idxcutass[*it] = idxcutass;
                    map_idxq_coarse[*it] = sofa::InvalidID;
                    idxcutass++;
                    // 						mask
                    int localidx = map_idxq_idxass[*it];
                    for(int j=0; j<8; ++j)
                    {
                        if( MIDDLE_INTERPOLATION[i][j] != 0 )
                        {
                            mask( localidx*3  , j*3  ) = 1;
                            mask( localidx*3+1, j*3+1) = 1;
                            mask( localidx*3+2, j*3+2) = 1;
                        }
                    }
                }
            }
        }

        EigenMatrix Kg = EigenMatrix::Zero(sizeass * 3, idxcutasscoarse * 3);// stiffness of contrained nodes
        EigenMatrix A = EigenMatrix::Zero(sizeass * 3, sizeass * 3); // [Kf -G] ==  Kf (stiffness of free nodes) with the constraints
        
        for( auto it = map_idxq_idxcutass.begin(); it!=map_idxq_idxcutass.end(); ++it)
        {
            int colcut = (*it).second;
            int colnoncut = map_idxq_idxass[(*it).first];

            if( map_idxq_coarse[(*it).first] != sofa::InvalidID )
            {
                for(int lig=0; lig<sizeass; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            Kg( lig*3+m,colcut*3+n) += assembledStiffnessStatic(lig*3+m,colnoncut*3+n);
                }
            }
            else
            {
                for(int lig=0; lig<sizeass; ++lig)
                {
                    for(int m=0; m<3; ++m)
                        for(int n=0; n<3; ++n)
                            A( lig*3+m,colcut*3+n) += assembledStiffnessStatic(lig*3+m,colnoncut*3+n);
                }
            }
        }


        // 		  put -G entries into A
        int d=0;
        for(int i=0; i<8; ++i) // for all constrained nodes
        {
            for(unsigned j=0; j<map_idxcoarse_idxfine[i].size(); ++j)
            {
                A( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3   , (sizeass-idxcutasscoarse+d)*3  ) += -1.0;
                A( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3+1 , (sizeass-idxcutasscoarse+d)*3+1) += -1.0;
                A( map_idxq_idxass[map_idxcoarse_idxfine[i][j]]*3+2 , (sizeass-idxcutasscoarse+d)*3+2) += -1.0;
                ++d;
            }
        }

        EigenMatrix Ainv = A.inverse();
        EigenMatrix Ainvf = Ainv.block( 0,0, (sizeass-idxcutasscoarse)*3,sizeass*3);



        //// ajouter un H qui lie tous les coins superpos�s ensemble et n'en garder que 8 pour avoir un W 27x8
        EigenMatrix H = EigenMatrix::Zero(idxcutasscoarse * 3, 8 * 3);
        for(int i=0; i<8; ++i)
        {
            for(unsigned j=0; j<map_idxcoarse_idxfine[i].size(); ++j)
            {
                H( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3  , i*3  ) =1;
                H( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3+1, i*3+1) =1;
                H( map_idxq_idxcutass[map_idxcoarse_idxfine[i][j]]*3+2, i*3+2) =1;
            }
        }



        EigenMatrix W = - Ainvf * Kg * H;
        EigenMatrix WB = EigenMatrix::Zero(sizeass * 3, 8 * 3);

        for( auto it= map_idxq_coarse.begin(); it!=map_idxq_coarse.end(); ++it)
        {
            if( it->second != sofa::InvalidID )
            {
                WB( map_idxq_idxass[it->first]*3  , it->second*3  ) += 1.0;
                WB( map_idxq_idxass[it->first]*3+1, it->second*3+1) += 1.0;
                WB( map_idxq_idxass[it->first]*3+2, it->second*3+2) += 1.0;
            }
            else
            {
                for(int j=0; j<8*3; ++j)
                {
                    WB( map_idxq_idxass[it->first]*3  ,j) += W( map_idxq_idxcutass[it->first]*3  , j);
                    WB( map_idxq_idxass[it->first]*3+1,j) += W( map_idxq_idxcutass[it->first]*3+1, j);
                    WB( map_idxq_idxass[it->first]*3+2,j) += W( map_idxq_idxcutass[it->first]*3+2, j);
                }
            }
        }



        // apply the mask to take only concerned values (an edge stays an edge, a face stays a face, if corner=1 opposite borders=0....)
        EigenMatrix WBmeca = EigenMatrix::Zero(sizeass * 3, 8 * 3);

        for(int i=0; i<sizeass*3; ++i)
        {
            for(int j=0; j<8*3; ++j)
            {
                if( mask(i,j)  )
                    WBmeca(i,j)=WB(i,j);
            }
        }

        // normalize the coefficient to obtain sum(coefs)==1
        for(int i=0; i<sizeass*3; ++i)
        {
            SReal sum = 0.0;
            for(int j=0; j<8*3; ++j)
            {
                sum += WBmeca(i,j);
            }
            for(int j=0; j<8*3; ++j)
            {
                WBmeca(i,j) =WBmeca(i,j) / sum ;
            }
        }


        EigenMatrix Kc, Mc; // coarse stiffness
        Kc = WBmeca.transpose() * assembledStiffness * WBmeca;
        Mc = WBmeca.transpose() * assembledMass * WBmeca;




        for(int i=0; i<8*3; ++i)
            for(int j=0; j<8*3; ++j)
            {
                K(i,j)=(Real)Kc(i,j);
                M(i,j)=(Real)Mc(i,j);
            }

        if( !d_completeInterpolation.getValue() ) // take WBmeca as the object interpolation
        {
            WB = WBmeca;
        }


        for(int i=0 ; i < 8 ; ++i ) // finer places
        {
            for(unsigned j=0; j<finerChildrenRamificationOriginal[i].size(); ++j) // finer element
            {
                const SparseGridTopology::Hexa& finehexa = finerSparseGrid->getHexahedron( finerChildrenRamificationOriginal[i][j] );
                for(int k=0 ; k < 8 ; ++k ) // fine nodes
                {
                    for( int l=0; l<8*3; ++l) // toutes les cols de W
                    {
                        _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]](k*3 , l) = (Real)WB( map_idxq_idxass[ finehexa[k] ]*3   ,l);
                        _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]](k*3+1 , l) = (Real)WB( map_idxq_idxass[ finehexa[k] ]*3+1 ,l);
                        _weights[this->d_nbVirtualFinerLevels.getValue()-level-1][finerChildrenRamificationOriginal[i][j]](k*3+2 , l) = (Real)WB( map_idxq_idxass[ finehexa[k] ]*3+2 ,l);
                    }
                }
            }
        }
    }
}




template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeFinalWeights( const Weight &W, const Index coarseElementIndice, const Index elementIndice,  int level)
{
    using namespace sofa::defaulttype;

    if( elementIndice == sofa::InvalidID ) return;

    Weight A = _weights[ this->d_nbVirtualFinerLevels.getValue()-level ][elementIndice]* W;

    if (level == this->d_nbVirtualFinerLevels.getValue())
    {
        _finalWeights[ elementIndice ] = std::pair<Index,Weight>(coarseElementIndice, A);

    }
    else
    {
        topology::container::grid::SparseGridTopology::SPtr sparseGrid;

        sparseGrid = this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level];

        auto& finerChildren = sparseGrid->_hierarchicalCubeMap[elementIndice];

        for(int i=0; i<8; ++i)
            computeFinalWeights( A, coarseElementIndice, finerChildren[i], level+1);
    }
}



template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::computeFinalWeightsRamification( const Weight &W, const Index coarseElementIndice, const Index elementIndice,  int level)
{

    using namespace sofa::defaulttype;

    if( elementIndice == sofa::InvalidID ) return;

    Weight A = _weights[ this->d_nbVirtualFinerLevels.getValue()-level ][elementIndice]* W;

    if (level == this->d_nbVirtualFinerLevels.getValue())
    {
        _finalWeights[ elementIndice ] = std::pair<Index,Weight>(coarseElementIndice, A);
    }
    else
    {
        topology::container::grid::SparseGridRamificationTopology* sparseGrid;

        sparseGrid = dynamic_cast<topology::container::grid::SparseGridRamificationTopology*>(this->_sparseGrid->_virtualFinerLevels[this->_sparseGrid->getNbVirtualFinerLevels()-level].get());

        auto& finerChildrenRamification = sparseGrid->_hierarchicalCubeMapRamification[ elementIndice ];

        for(int w=0; w<8; ++w)
            for(unsigned v=0; v<finerChildrenRamification[w].size(); ++v)
                computeFinalWeights( A, coarseElementIndice, finerChildrenRamification[w][v], level+1);
    }
}



template<class T>
void HexahedronCompositeFEMForceFieldAndMass<T>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    if (!this->mstate) return;
    if (vparams->displayFlags().getShowWireFrame()) return;


    if( d_drawColor.getValue() == -1 ) return;

    if( d_drawType.getValue() == -1 ) return HexahedronFEMForceFieldAndMassT::draw(vparams);

    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    const VecCoord& x = this->mstate->read(core::vec_id::read_access::position)->getValue();


    sofa::type::RGBAColor colour;
    switch(d_drawColor.getValue() )
    {
    case 3:
        colour=sofa::type::RGBAColor(0.2f, 0.8f, 0.2f,1.0f);
        break;

    case 2:
        colour=sofa::type::RGBAColor(0.2f, 0.3f, 0.8f,1.0f);
        break;

    case 1:
        colour=sofa::type::RGBAColor(0.95f, 0.3f, 0.2f,1.0f);
        break;

    case 0:
    default:
        colour=sofa::type::RGBAColor(0.9f, 0.9f, 0.2f,1.0f);
    }


    if( d_drawType.getValue() == 0 )
    {
        vparams->drawTool()->setLightingEnabled(true);

        for( SparseGridTopology::SeqEdges::const_iterator it = this->_sparseGrid->getEdges().begin() ; it != this->_sparseGrid->getEdges().end(); ++it)
        {
            vparams->drawTool()->drawCylinder( x[(*it)[0]], x[(*it)[1]], d_drawSize.getValue(), colour );
        }

        vparams->drawTool()->setLightingEnabled(false);
    }
    else
    {
        std::vector< type::Vec3 > points;

        vparams->drawTool()->setLightingEnabled(false);

        for( SparseGridTopology::SeqEdges::const_iterator it = this->_sparseGrid->getEdges().begin() ; it != this->_sparseGrid->getEdges().end(); ++it)
        {
            points.push_back( x[(*it)[0]] );
            points.push_back( x[(*it)[1]] );
        }
        vparams->drawTool()->drawLines(points, d_drawSize.getValue(),colour);
    }





    if (vparams->displayFlags().getShowBehaviorModels())
    {
        colour=sofa::type::RGBAColor(0.95f, 0.95f, 0.7f,1.0f);

        std::vector< sofa::type::Vec3 > points;
        for(unsigned i=0; i<x.size(); ++i)
            points.push_back( x[i] );
        vparams->drawTool()->drawSpheres(points, d_drawSize.getValue()*1.5f,colour);
    }

    if( d_drawType.getValue()!=2 ) return;
    topology::container::grid::SparseGridRamificationTopology* sgr = dynamic_cast<topology::container::grid::SparseGridRamificationTopology*>(this->_sparseGrid);
    if( sgr==nullptr) return;

    {

        std::vector< sofa::type::Vec3 > points;
        for(unsigned i=0; i<sgr->getConnexions()->size(); ++i)
        {
            type::vector< topology::container::grid::SparseGridRamificationTopology::Connexion *>& con = (*sgr->getConnexions())[i];

            if( con.empty() ) continue;



            Index a = (*this->getIndexedElements())[con[0]->_hexaIdx][0];
            Index b = (*this->getIndexedElements())[con[0]->_hexaIdx][1];
            Index d = (*this->getIndexedElements())[con[0]->_hexaIdx][3];
            Index c = (*this->getIndexedElements())[con[0]->_hexaIdx][2];
            Index e = (*this->getIndexedElements())[con[0]->_hexaIdx][4];
            Index f = (*this->getIndexedElements())[con[0]->_hexaIdx][5];
            Index h = (*this->getIndexedElements())[con[0]->_hexaIdx][7];
            Index g = (*this->getIndexedElements())[con[0]->_hexaIdx][6];



            Coord pa = x[a];
            Coord pb = x[b];
            Coord pc = x[c];
            Coord pd = x[d];
            Coord pe = x[e];
            Coord pf = x[f];
            Coord pg = x[g];
            Coord ph = x[h];


            switch( con.size() )
            {
            case 1:
                colour=sofa::type::RGBAColor(0.7f, 0.7f, 0.1f, .4f);
                break;
            case 2:
                colour=sofa::type::RGBAColor(0.1f, 0.9f, 0.1f, .4f);
                break;
            case 3:
                colour=sofa::type::RGBAColor(0.9f, 0.1f, 0.1f, .4f);
                break;
            case 4:
                colour=sofa::type::RGBAColor(0.1f, 0.1f, 0.9f, .4f);
                break;
            case 5:
            default:
                colour=sofa::type::RGBAColor(0.2f, 0.2f, 0.2f, .4f);
                break;
            }

            points.push_back(pa);
            points.push_back(pb);
            points.push_back(pc);

            points.push_back(pa);
            points.push_back(pc);
            points.push_back(pd);

            points.push_back(pe);
            points.push_back(pf);
            points.push_back(pg);

            points.push_back(pe);
            points.push_back(pg);
            points.push_back(ph);

            points.push_back(pc);
            points.push_back(pd);
            points.push_back(ph);

            points.push_back(pc);
            points.push_back(ph);
            points.push_back(pg);

            points.push_back(pa);
            points.push_back(pb);
            points.push_back(pf);

            points.push_back(pa);
            points.push_back(pf);
            points.push_back(pe);

            points.push_back(pa);
            points.push_back(pd);
            points.push_back(ph);

            points.push_back(pa);
            points.push_back(ph);
            points.push_back(pe);

            points.push_back(pb);
            points.push_back(pc);
            points.push_back(pg);

            points.push_back(pb);
            points.push_back(pg);
            points.push_back(pf);

        }
        vparams->drawTool()->drawTriangles(points, colour);
    }

}


} // namespace sofa::component::solidmechanics::fem::nonuniform

