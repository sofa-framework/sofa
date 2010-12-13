/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef MappingTypes_H
#define MappingTypes_H

#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace defaulttype
{

template<class In, class Out, class Material, int nbRef, int order>
class LinearBlendTypes;

template< class Primitive, class Real, int Dim>
class DataTypesInfo
{
public:
    enum {primitive_order = 0};
};





template<class TCoord, class TDeriv, class TReal>
class StdVectorTypes;

template<class TCoord, class TDeriv, class TReal>
class ExtVectorTypes;

template<int _spatial_dimensions, int _material_dimensions, int _order, typename _Real>
struct DeformationGradientTypes;

template< class Real, int Dim>
class DataTypesInfo<StdVectorTypes<Vec<Dim, Real>, Vec<Dim, Real>, Real>,Real,Dim>
{
public:
    enum {primitive_order = 0};
};

template< class Real, int Dim>
class DataTypesInfo<ExtVectorTypes<Vec<Dim, Real>, Vec<Dim, Real>, Real>,Real,Dim>
{
public:
    enum {primitive_order = 0};
};

template< class Real, int Dim>
class DataTypesInfo<DeformationGradientTypes<Dim,Dim,0,Real>,Real,Dim>
{
public:
    enum {primitive_order = 0}; // DeformationGradientTypes<Dim,Dim,0,Real>::order};
};

template< class Real, int Dim>
class DataTypesInfo<DeformationGradientTypes<Dim,Dim,1,Real>,Real,Dim>
{
public:
    enum {primitive_order = 1}; // DeformationGradientTypes<Dim,Dim,1,Real>::order};
};

template< class Real, int Dim>
class DataTypesInfo<DeformationGradientTypes<Dim,Dim,2,Real>,Real,Dim>
{
public:
    enum {primitive_order = 2}; // DeformationGradientTypes<Dim,Dim,2,Real>::order};
};


} // namespace defaulttype
} // namespace sofa



#endif
