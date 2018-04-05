#ifndef FLEXIBLE_COMPUTEDUALQUATENGINE_H
#define FLEXIBLE_COMPUTEDUALQUATENGINE_H

#include <sofa/core/DataEngine.h>
#include <sofa/defaulttype/Vec.h>

namespace sofa
{

namespace component
{

namespace engine
{



/*
 * Engine which converts a vector of Affines or Rigid to a vector of
 * Dual Quaternions.
 *
 * If x0 is provided, the dual quaternion are computed from x0^-1*x.
 *
 */
template < class DataTypes >
class ComputeDualQuatEngine : public sofa::core::DataEngine
{
public:
    SOFA_CLASS( SOFA_TEMPLATE( ComputeDualQuatEngine, DataTypes ), sofa::core::DataEngine );


    // Typedefs
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;


    // Constructor
    ComputeDualQuatEngine();


    // To simplify the template name in the xml file
    virtual std::string getTemplateName() const
    {
        return templateName(this);
    }


    // To simplify the template name in the xml file
    static std::string templateName(const ComputeDualQuatEngine<DataTypes>* = NULL)
    {
        return DataTypes::Name();
    }

    // Update the engine
    void update();

    // First position
    Data< VecCoord > d_x0; ///< Rest position
    // Current position
    Data< VecCoord > d_x; ///< Current position

    // The dual quaternions, as vector of Vec4f
    Data< helper::vector< defaulttype::Vec4f > > d_dualQuats; ///< Dual quaternions, computed from x (or x*x0^-1 if x0 is provided). DualQuats are stored as two vec4f elements, first the orientation, then the dual.


};


} // namespace engine

} // namespace component

} // namespace sofa

#endif // FLEXIBLE_COMPUTEDUALQUATENGINE_H
