#define FLEXIBLE_COMPUTEWEIGHTENGINE_CPP

#include "ComputeWeightEngine.h"

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace engine
{

SOFA_DECL_CLASS( ComputeWeightEngine )

using namespace defaulttype;

int ComputeWeightEngineClass = core::RegisterObject("Computes the weight and indices of a set of vertices of an VisualModelImpl using a given shape function.")
    .add< ComputeWeightEngine >()
;


ComputeWeightEngine::ComputeWeightEngine()
    : l_shapeFunction(initLink("shapeFunction", "Shape function object"))
    , l_visualModel( initLink( "visual", "Visual model"))
    , d_indices( initData( &d_indices, "indices", "Indices" ) )
    , d_weights( initData( &d_weights, "weights", "Weights" ) )
{
    addOutput( &d_indices );
    addOutput( &d_weights );
}


void ComputeWeightEngine::init()
{
    // Check that visual model and shapeFunction are present.
    // Check that shapeFunction::nbRef is not greater than 4.

    if ( !l_visualModel )
    {
        serr << "visual must be defined" << sendl;
    }

    if ( !l_shapeFunction )
    {
        serr << "shapeFunction must be defined" << sendl;
    }
    else
    {
        Data<unsigned int>* nbRef = dynamic_cast< Data<unsigned int>* >( l_shapeFunction->findData( "nbRef" ) );
        if( nbRef && nbRef->getValue() > 4 )
        {
            serr << "shapeFunction should not have a nbRef greater than 4. Current value is " << nbRef->getValue() << sendl;
        }
    }
    update();
}


void ComputeWeightEngine::reinit()
{
    init();
}


void ComputeWeightEngine::update()
{
    if( !l_visualModel || !l_shapeFunction )
        return;

    // Get vertices from the visual model.
    // We can not use another method as vertices might be duplicated
    // and this is the only method that takes it into account.
    sofa::defaulttype::ResizableExtVector<sofa::defaulttype::Vec< 3, ExtVec3fTypes::Real > > vertices ( l_visualModel.get()->getVertices() );
    size_t nb_vertices = vertices.size();

    // Get indices and weight
    sofa::helper::vector< Indices >& indices = *d_indices.beginEdit();
    sofa::helper::vector< Weights >& weights = *d_weights.beginEdit();

    indices.resize( nb_vertices );
    weights.resize( nb_vertices );

    BaseShapeFunction* sf = l_shapeFunction.get();
    for(unsigned i = 0; i < nb_vertices; ++i )
    {
        BaseShapeFunction::MaterialToSpatial M;
        BaseShapeFunction::VRef ref;
        BaseShapeFunction::VReal w;

        // Compute weights and indices for the given element
        sf->computeShapeFunction( vertices[i], M, ref, w );
        unsigned j = 0;
        for( ; j < ref.size(); ++j )
        {
            // Fill indices and weight
            indices[i][j] = ref[j];
            weights[i][j] = w[j];
        }
        for( ; j < 4; ++j )
        {
            // Set to 0.
            indices[i][j] = 0;
            weights[i][j] = 0;
        }
    }

    d_indices.endEdit();
    d_weights.endEdit();

}

} // namespace engine

} // namespace component

} // namespace sofa
