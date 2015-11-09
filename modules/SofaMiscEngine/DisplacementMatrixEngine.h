#ifndef FLEXIBLE_DisplacementMatrixENGINE_H
#define FLEXIBLE_DisplacementMatrixENGINE_H

#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{



/*
 * Engine which computes a displacement matrix with respect to an origin matrix, as D(t) = M(t).M(0)^-1
 * Adapted from ComputeDualQuatEngine
 * François Faure, 2015
 *
 */
template < class DataTypes >
class DisplacementMatrixEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE( DisplacementMatrixEngine, DataTypes ), sofa::core::DataEngine );


    // Typedefs
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord; // rigid
    typedef typename DataTypes::VecCoord VecCoord;


    // Constructor
    DisplacementMatrixEngine();


    // To simplify the template name in the xml file
    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    // To simplify the template name in the xml file
    static std::string templateName(const DisplacementMatrixEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    // Update the engine
    void update();

    // First position
    Data< VecCoord > d_x0;  // initial matrices
    // Current position
    Data< VecCoord > d_x;   // current matrices

    // The displacement matrices, as vector of Mat4f
    typedef defaulttype::Mat4x4f Mat4;
    Data< helper::vector< Mat4 > > d_displaceMats;


};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_DisplacementMatrixENGINE_H
