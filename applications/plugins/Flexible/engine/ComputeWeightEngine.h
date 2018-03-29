#ifndef FLEXIBLE_COMPUTEWEIGHTENGINE_H
#define FLEXIBLE_COMPUTEWEIGHTENGINE_H

#include "../shapeFunction/BaseShapeFunction.h"
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/core/DataEngine.h>
#include <sofa/helper/vector.h>



namespace sofa
{

namespace component
{

namespace engine
{

/*
 * Engine which compute the weight and indices of a set of vertices of
 * an VisualModelImpl using a given shape function.
 *
 * We can not point directly to the visual model vertices as the
 * method VisualModelImpl::getVertices is not bound to a data.
 */
class SOFA_Flexible_API ComputeWeightEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( ComputeWeightEngine , sofa::core::DataEngine );


    // Creator
    ComputeWeightEngine();


    // Shape function
    typedef core::behavior::ShapeFunctionTypes< 3, SReal > ShapeFunctionType;
    typedef core::behavior::BaseShapeFunction< ShapeFunctionType > BaseShapeFunction;
    typedef SingleLink< ComputeWeightEngine, BaseShapeFunction, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkShapeFunction;
    LinkShapeFunction l_shapeFunction;

    // Input type
    typedef SingleLink< ComputeWeightEngine, component::visualmodel::VisualModelImpl, BaseLink::FLAG_STOREPATH|BaseLink::FLAG_STRONGLINK> LinkVisual;
    LinkVisual l_visualModel;

    // Typedefs
    typedef defaulttype::Vec4u Indices;
    typedef defaulttype::Vector4 Weights;



    void init();
    void reinit();
    // Update the engine
    void update();

protected:

    // Indices
    Data< helper::vector< Indices > > d_indices; ///< Indices
    // Weights
    Data< helper::vector< Weights > > d_weights; ///< Weights

};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_COMPUTEWEIGHTENGINE_H
