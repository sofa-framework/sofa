/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_INL
#define SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_INL

#include "DisplacementMatrixEngine.h"


namespace sofa
{

namespace component
{

namespace engine
{

///////////////////////////////////////////////////////////////
/// DisplacementTransformEngine
///////////////////////////////////////////////////////////////
template < class DataTypes, class OutputType >
DisplacementTransformEngine< DataTypes, OutputType >::DisplacementTransformEngine()
:
Inherit()
, d_x0( initData( &d_x0, "x0", "Rest position" ) )
, d_x( initData( &d_x, "x", "Current position" ) )
, d_displacements( initData( &d_displacements, "displacements", "Displacement transforms with respect to original rigid positions") )
{
    addInput( &d_x0 );
    addInput( &d_x );
    addOutput( &d_displacements );
    setDirtyValue();
}

template < class DataTypes, class OutputType >
void DisplacementTransformEngine< DataTypes, OutputType >::init()
{
    // parent method
    Inherit::init();

    // Computation of inverse matrix
    const VecCoord& x0 = d_x0.getValue();
    inverses.resize(x0.size());
    for( size_t i=0; i<x0.size(); ++i )
    {
        setInverse( inverses[i], x0[i] );
    }
}

template < class DataTypes, class OutputType >
void DisplacementTransformEngine< DataTypes, OutputType >::update()
{
    // parent method
    Inherit::init();

    const VecCoord& x = d_x.getValue();
    const VecCoord& x0 = d_x0.getValue();
    const size_t size = x.size();
    const size_t size0 = x0.size();

    // Check the size of x0
    if( size != size0 )
    {
        serr << "x and x0 have not the same size: respectively " << size << " and " << size0 << sendl;
        return;
    }

    cleanDirty();

    // Clean the output
    helper::vector< OutputType >& displacements = *d_displacements.beginWriteOnly();
    displacements.resize(size);

    for( unsigned int i = 0; i < size; ++i )
    {
        mult( displacements[i], inverses[i], x[i] );
    }
    d_displacements.endEdit();
    //serr << "update(), displaceMats  = " << d_displaceMats.getValue() << sendl;
}

///////////////////////////////////////////////////////////////
/// DisplacementMatrixEngine
///////////////////////////////////////////////////////////////
template < class DataTypes >
DisplacementMatrixEngine< DataTypes >::DisplacementMatrixEngine()
: Inherit()
, d_scales( initData(&d_scales, "scales", "Scale transformation added to the rigid transformation"))
{
    this->addInput( &d_scales );
    this->d_displacements.setName( "displaceMats" );
    this->addAlias( &this->d_displacements, "displaceMats" );
}

template < class DataTypes >
void DisplacementMatrixEngine< DataTypes >::init()
{
    // parent method
    Inherit::init();

    // Init of the scale matrices in case if the user did not initialize them
    const VecCoord& x0 = this->d_x0.getValue();
    helper::vector< sofa::defaulttype::Vec<3,Real> >& scales = *d_scales.beginWriteOnly();
    if (scales.size() == 0)
        for( size_t i=0; i<x0.size(); ++i )
        {
            scales.push_back(sofa::defaulttype::Vec<3,Real>(1,1,1));
        }
    d_scales.endEdit();

    // Init of the product between the scale matrices and the inverse
    this->reinit();
}

template < class DataTypes >
void DisplacementMatrixEngine< DataTypes >::reinit()
{
    // parent method
    Inherit::reinit();

    const VecCoord& x0 = this->d_x0.getValue();
    const helper::vector< sofa::defaulttype::Vec<3,Real> >& scales = this->d_scales.getValue();
    const size_t size0 = x0.size();
    const size_t sizeS = scales.size();

    if( size0 != sizeS)
    {
        serr << "x0 and S have not the same size: respectively " << ", " << size0 << " and " << sizeS << sendl;
        return;
    }

    this->SxInverses.resize(size0);
    for( unsigned int i = 0; i < size0; ++i )
    {
        Matrix4x4 S;
        S[0][0] = (float)scales[i][0];
        S[1][1] = (float)scales[i][1];
        S[2][2] = (float)scales[i][2];
        S[3][3] = (float)1;

        this->SxInverses[i] =  S * this->inverses[i];
    }

    this->update();
}

template < class DataTypes >
void DisplacementMatrixEngine< DataTypes >::update()
{
    const VecCoord& x = this->d_x.getValue();
    const VecCoord& x0 = this->d_x0.getValue();
    const helper::vector< sofa::defaulttype::Vec<3,Real> >& scales = this->d_scales.getValue();
    const size_t size = x.size();
    const size_t size0 = x0.size();
    const size_t sizeS = scales.size();

    // Check the size of x0
    if( size != size0 || size != sizeS)
    {
        serr << "x, x0 and S have not the same size: respectively " << size << ", " << size0 << " and " << sizeS << sendl;
        return;
    }

    this->cleanDirty();

    helper::vector< Matrix4x4 >& displacements = *this->d_displacements.beginWriteOnly();
    displacements.resize(size);
    for( unsigned int i = 0; i < size; ++i )
    {
        x[i].toMatrix(displacements[i]);
        displacements[i] = displacements[i] * this->SxInverses[i];
    }
    this->d_displacements.endEdit();
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
