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
    const VecCoord& x0 = d_x0.getValue();
    inverses.resize(x0.size());
    for( size_t i=0; i<x0.size(); i++ )
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
//    serr << "update(), displaceMats  = " << d_displaceMats.getValue() << sendl;
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif
