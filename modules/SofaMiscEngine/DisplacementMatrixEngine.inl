#ifndef SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_INL
#define SOFA_COMPONENT_ENGINE_DisplacementMatrixEngine_INL

#include "DisplacementMatrixEngine.h"


namespace sofa
{

namespace component
{

namespace engine
{

template < class DataTypes, class OutputType >
DisplacementTransformEngine< DataTypes, OutputType >::DisplacementTransformEngine()
: d_x0( initData( &d_x0, "x0", "Rest position" ) )
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

/////////////////////////////////////////////

template < class DataTypes >
DisplacementMatrixEngine< DataTypes >::DisplacementMatrixEngine()
: DisplacementTransformEngine()
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
    DisplacementTransformEngine::init();

    // Init of the scale matrices in case if the user did not initialize them
    const VecCoord& x0 = this->d_x0.getValue();
    helper::vector< sofa::defaulttype::Vec<3,Real> >& scales = *d_scales.beginWriteOnly();
    if (scales.size() == 0)
        for( size_t i=0; i<x0.size(); ++i )
        {
            scales.push_back(sofa::defaulttype::Vec<3,Real>(1,1,1));
        }
    d_scales.endEdit();
}

template < class DataTypes >
void DisplacementMatrixEngine< DataTypes >::update()
{
    // parent method
    DisplacementTransformEngine::update();

    // Variable
    const VecCoord& x = this->d_x.getValue();
    const helper::vector< sofa::defaulttype::Vec<3,Real> >& scales = d_scales.getValue();
    const size_t size = x.size();
    const size_t sizeS = scales.size();

    // Check the size of x0
    if( size != sizeS )
    {
        serr << "x, and S have not the same size: respectively " << size << " and " << sizeS << sendl;
        return;
    }

    // Convert the scale vector into a 4x4 matrix to allow the multiplication
    helper::vector< Matrix4x4 > S;
    for( size_t i=0; i<scales.size(); ++i )
    {
        Matrix4x4 s;
        s[0][0] = scales[i][0];
        s[1][1] = scales[i][1];
        s[2][2] = scales[i][2];
        s[3][3] = (Real)1;
        S.push_back(s);
    }

    helper::vector< Matrix4x4 >& displacements = *this->d_displacements.beginWriteOnly();
    for( unsigned int i = 0; i < size; ++i )
    {
        displacements[i] = displacements[i] * S[i];
    }
    this->d_displacements.endEdit();

}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
