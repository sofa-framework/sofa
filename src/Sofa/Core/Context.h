
#ifndef	SOFA_CORE_Properties_h
#define	SOFA_CORE_Properties_h

#include <iostream>
using std::endl;
#include "Sofa/Components/Common/Vec.h"
#include <Sofa/Components/Common/RigidTypes.h>


namespace Sofa
{

namespace Core
{

class Context
{

public:
    typedef Components::Common::Vec3f Vec;
    typedef Sofa::Components::Common::RigidTypes::Coord Frame;
    virtual ~Context() {}


    /// @name Getters
    /// @{
    /// Gravity in the local coordinate system
    virtual const Vec& getGravity() const;

    /// Projection from the local coordinate system to the world coordinate system
    virtual const Frame& getWorldTransform() const;

    /// Simulation timestep
    virtual float getDt() const;


    /// Animation flag
    virtual bool getAnimate() const;


    /// MultiThreading activated
    virtual bool getMultiThreadSimulation() const;

    /// Display flags: Collision Models
    virtual bool getShowCollisionModels() const;

    /// Display flags: Behavior Models
    virtual bool getShowBehaviorModels() const;

    /// Display flags: Visual Models
    virtual bool getShowVisualModels() const;

    /// Display flags: Mappings
    virtual bool getShowMappings() const;

    /// Display flags: ForceFields
    virtual bool getShowForceFields() const;

    /// @}

    /// @name Global Parameters Setters
    /// @{

    /// Gravity in local coordinates
    virtual void setGravity( const Vec& g );
    /// Projection from the local frame to the world frame
    virtual void setWorldTransform( const Frame& f );
    /// Simulation timestep
    virtual void setDt( float dt );

    /// Animation flag
    virtual void setAnimate(bool val);


    /// MultiThreading activated
    virtual void setMultiThreadSimulation(bool val);

    /// Display flags: Collision Models
    virtual void setShowCollisionModels(bool val);

    /// Display flags: Behavior Models
    virtual void setShowBehaviorModels(bool val);

    /// Display flags: Visual Models
    virtual void setShowVisualModels(bool val);

    /// Display flags: Mappings
    virtual void setShowMappings(bool val);

    /// Display flags: ForceFields
    virtual void setShowForceFields(bool val);

    /// @}


    void copyContextFrom( const Context* f )
    {
        *this = *f;
    }
    static Context getDefault()
    {
        Context d;
        d.setGravity( Vec(0,-9.81,0) );
        d.setWorldTransform( Frame::identity() );
        return d;
    }

    inline friend std::ostream& operator << (std::ostream& out, const Context& c )
    {
        out<<endl<<"gravity = "<<c.getGravity();
        out<<endl<<"worldTransform = "<<c.getWorldTransform();
        return out;
    }
private:
    Vec gravity_;
    Frame worldTransform_;
    float dt_;
    bool animate_;
    bool showCollisionModels_;
    bool showBehaviorModels_;
    bool showVisualModels_;
    bool showMappings_;
    bool showForceFields_;
    bool multiThreadSimulation_;
};

}//Core
}//Sofa

#endif


