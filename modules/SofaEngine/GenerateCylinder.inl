/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_COMPONENT_ENGINE_GENERATECYLINDER_INL
#define SOFA_COMPONENT_ENGINE_GENERATECYLINDER_INL

#include "GenerateCylinder.h"
#include <sofa/helper/rmath.h> //M_PI

namespace sofa
{

namespace component
{

namespace engine
{

template <class DataTypes>
GenerateCylinder<DataTypes>::GenerateCylinder()
    : f_outputX ( initData (&f_outputX, "output_position", "output array of 3d points") )
    , f_tetrahedron( initData (&f_tetrahedron, "tetrahedra", "output mesh tetrahedra") )
    , f_radius( initData (&f_radius,(Real)0.2, "radius", "input cylinder radius") )
    , f_height( initData (&f_height,(Real)1.0, "height", "input cylinder height") )
	, f_origin( initData (&f_origin,Coord(), "origin", "cylinder origin point") )
    , f_resolutionCircumferential( initData (&f_resolutionCircumferential,(size_t)6, "resCircumferential", "Resolution in the circumferential direction") )
   , f_resolutionRadial( initData (&f_resolutionRadial,(size_t)3, "resRadial", "Resolution in the radial direction") )
  , f_resolutionHeight( initData (&f_resolutionHeight,(size_t)5, "resHeight", "Resolution in the height direction") )
{
    addAlias(&f_outputX,"position");
}


template <class DataTypes>
void GenerateCylinder<DataTypes>::init()
{
    addInput(&f_radius);
    addInput(&f_height);
    addInput(&f_origin);
    addInput(&f_resolutionCircumferential);
    addInput(&f_resolutionRadial);
    addInput(&f_resolutionHeight);
    addOutput(&f_outputX);
    addOutput(&f_tetrahedron);
    setDirtyValue();
}

template <class DataTypes>
void GenerateCylinder<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void GenerateCylinder<DataTypes>::update()
{
    cleanDirty();

	helper::WriteAccessor<Data<VecCoord> > out = f_outputX;
	SeqTetrahedra  &tetras = *(f_tetrahedron.beginEdit());

	const Real radius = f_radius.getValue();
	const Real height = f_height.getValue();
	const Coord origin = f_origin.getValue();

	const size_t freqTheta=f_resolutionCircumferential.getValue();
	const size_t freqR=f_resolutionRadial.getValue();
	const size_t freqZ=f_resolutionHeight.getValue();


	size_t  nbVertices= (freqTheta*freqR+1)*freqZ;
	out.resize(nbVertices);

	Real zValue,r,theta,xValue,yValue;
	size_t i,j,k,index;
	Coord pos;

	for(index=0,i=0;i<freqZ;i++) {
		// vertex index = i*(freQTheta*freqR+1)
		zValue=i*height/(freqZ-1);
		pos=Coord(0,0,zValue);
		pos+=origin;
		out[index++]=pos;
		for(j=1;j<=freqR;++j) {
			r=j*radius/(freqR);
			for(k=0;k<freqTheta;++k) {
				theta=k*2*M_PI/freqTheta;
				xValue= r*cos(theta);
				yValue= r*sin(theta);
				pos=Coord(xValue,yValue,zValue);
				pos+=origin;
				out[index++]=pos;
			}
		}
	}


	size_t nbTetrahedra=3*freqTheta*(freqZ-1) + 6*(freqR-1)*freqTheta*(freqZ-1);
	tetras.resize(nbTetrahedra);

	size_t  offsetZ=(freqTheta*freqR+1);
	size_t prevk;
	index=0;
	size_t prism[6];
	size_t hexahedron[8];

	for(i=1;i<freqZ;i++) {
		size_t  centerIndex0=i*offsetZ;
		prevk=freqTheta;

		for(k=1;k<=freqTheta;++k) {
			/// create triangular prism
			prism[0]=centerIndex0;
			prism[1]=centerIndex0+prevk;
			prism[2]=centerIndex0+k;
			prism[3]=prism[0]-offsetZ;
			prism[4]=prism[1]-offsetZ;
			prism[5]=prism[2]-offsetZ;
			/// decompose triangular prism into 3 tetrahedra
			tetras[index++]=Tetrahedron(prism[1],prism[0],prism[2],prism[3]);
			tetras[index++]=Tetrahedron(prism[1],prism[2],prism[4],prism[3]);
			tetras[index++]=Tetrahedron(prism[3],prism[4],prism[5],prism[2]);

			prevk=k;
		}

		for(j=1;j<freqR;++j) {
			prevk=freqTheta;
			for(k=1;k<=freqTheta;++k) {
				/// create hexahedron
				hexahedron[0]=centerIndex0+k;
				hexahedron[1]=centerIndex0+prevk;
				hexahedron[2]=centerIndex0+k+freqTheta;
				hexahedron[3]=centerIndex0+prevk+freqTheta;
				hexahedron[4]=hexahedron[0]-offsetZ;
				hexahedron[5]=hexahedron[1]-offsetZ;
				hexahedron[6]=hexahedron[2]-offsetZ;
				hexahedron[7]=hexahedron[3]-offsetZ;
				/// decompose hexahedron into 6 tetra
				tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[5],hexahedron[4],hexahedron[7]);
				tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[1],hexahedron[5],hexahedron[3]);
				tetras[index++]=Tetrahedron(hexahedron[5],hexahedron[0],hexahedron[3],hexahedron[7]);
				tetras[index++]=Tetrahedron(hexahedron[0],hexahedron[3],hexahedron[7],hexahedron[2]);
				tetras[index++]=Tetrahedron(hexahedron[7],hexahedron[0],hexahedron[2],hexahedron[4]);
				tetras[index++]=Tetrahedron(hexahedron[6],hexahedron[7],hexahedron[2],hexahedron[4]);
				prevk=k;	
			}
			centerIndex0+=freqTheta;
		}

	}
	f_tetrahedron.endEdit();
}


} // namespace engine

} // namespace component

} // namespace sofa

#endif
