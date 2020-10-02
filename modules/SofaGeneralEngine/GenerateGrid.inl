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
#ifndef SOFA_COMPONENT_ENGINE_GENERATEGRID_INL
#define SOFA_COMPONENT_ENGINE_GENERATEGRID_INL

#include "GenerateGrid.h"


namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
GenerateGrid<DataTypes>::GenerateGrid()
    : f_outputX ( initData (&f_outputX, "output_position", "output array of 3d points") )
    , f_tetrahedron( initData (&f_tetrahedron, "tetrahedra", "output mesh tetrahedra") )
    , f_hexahedron( initData (&f_hexahedron, "hexahedra", "output mesh hexahedra") )
    , f_length( initData (&f_length,(Real)1.0, "length", "length of each grid cube") )
    , f_height( initData (&f_height,(Real)1.0, "height", "height of each grid cube") )
    , f_width( initData (&f_width,(Real)1.0, "width", "width of each grid cube") )
	, f_origin( initData (&f_origin,Coord(), "origin", "Grid origin point") )
    , f_resolutionLength( initData (&f_resolutionLength,(size_t)3, "resLength", "Number of Cubes in the length direction") )
    , f_resolutionWidth( initData (&f_resolutionWidth,(size_t)3, "resWidth", "Number of Cubes in the width direction") )
    , f_resolutionHeight( initData (&f_resolutionHeight,(size_t)3, "resHeight", "Number of Cubes in the height direction") )
{
    addAlias(&f_outputX,"position");
}


template <class DataTypes>
void GenerateGrid<DataTypes>::init()
{
    addInput(&f_length);
    addInput(&f_height);
    addInput(&f_width);
    addInput(&f_origin);
    addInput(&f_resolutionLength);
    addInput(&f_resolutionWidth);
    addInput(&f_resolutionHeight);
    addOutput(&f_outputX);
    addOutput(&f_tetrahedron);
    addOutput(&f_hexahedron);
    setDirtyValue();
}

template <class DataTypes>
void GenerateGrid<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void GenerateGrid<DataTypes>::update()
{
    cleanDirty();

	helper::WriteAccessor<Data<VecCoord> > out = f_outputX;
	SeqTetrahedra  &tetras = *(f_tetrahedron.beginEdit());
	SeqHexahedra  &hexas = *(f_hexahedron.beginEdit());

	const Real length = f_length.getValue();
	const Real height = f_height.getValue();
	const Real width = f_width.getValue();
	const Coord origin = f_origin.getValue();

	const size_t freqL=f_resolutionLength.getValue();
	const size_t freqH=f_resolutionHeight.getValue();
	const size_t freqW=f_resolutionWidth.getValue();


	size_t  nbVertices= (freqL+1)*(freqH+1)*(freqW+1);
	out.resize(nbVertices);

	size_t i,j,k,index;
	Coord pos;

	for(index=0,k=0;k<=freqH;++k) {
		for(j=0;j<=freqW;++j) {
			for(i=0;i<=freqL;i++) {
				pos=Coord(i*length,j*width,k*height);
				pos+=origin;
				out[index++]=pos;
			}
		}
	}


	size_t nbHexahedra=(freqL)*(freqH)*(freqW);
	hexas.resize(nbHexahedra);
	tetras.resize(nbHexahedra*6);

	typedef sofa::core::topology::Topology::PointID PointID;
	Hexahedron hexahedron;
	for(index=0,i=0;i<freqL;i++) {
		for(j=0;j<freqW;++j) {
			for(k=0;k<freqH;++k) {
				hexahedron[0]=(PointID)(i+j*(freqL+1)+k*(freqL+1)*(freqW+1));
				hexahedron[1]=(PointID)(i+1+j*(freqL+1)+k*(freqL+1)*(freqW+1));
				hexahedron[2]=(PointID)(i+1+(j+1)*(freqL+1)+k*(freqL+1)*(freqW+1));
				hexahedron[3]=(PointID)(i+(j+1)*(freqL+1)+k*(freqL+1)*(freqW+1));
				hexahedron[4]=(PointID)(i+j*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
				hexahedron[5]=(PointID)(i+1+j*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
				hexahedron[6]=(PointID)(i+1+(j+1)*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
				hexahedron[7]=(PointID)(i+(j+1)*(freqL+1)+(k+1)*(freqL+1)*(freqW+1));
				hexas[index]=hexahedron;
				/// decompose hexahedron into 6 tetra
				tetras[6*index]=Tetrahedron(hexahedron[0],hexahedron[5],hexahedron[1],hexahedron[6]);
				tetras[6*index+1]=Tetrahedron(hexahedron[0],hexahedron[1],hexahedron[3],hexahedron[6]);
				tetras[6*index+2]=Tetrahedron(hexahedron[1],hexahedron[3],hexahedron[6],hexahedron[2]);
				tetras[6*index+3]=Tetrahedron(hexahedron[6],hexahedron[3],hexahedron[0],hexahedron[7]);
				tetras[6*index+4]=Tetrahedron(hexahedron[6],hexahedron[7],hexahedron[0],hexahedron[5]);
				tetras[6*index+5]=Tetrahedron(hexahedron[7],hexahedron[5],hexahedron[4],hexahedron[0]);
				index++;
			}
		}
	}





	f_tetrahedron.endEdit();
	f_hexahedron.endEdit();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
