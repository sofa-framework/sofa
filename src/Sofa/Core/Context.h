
#ifndef	SOFA_CORE_Context_h
#define	SOFA_CORE_Context_h

#include <iostream>
using std::endl;
#include "Sofa/Components/Common/Vec.h"
#include <Sofa/Components/Common/Frame.h>


namespace Sofa
{

namespace Core
{

class Context
{

public:
    typedef Components::Common::Vec3f Vec;
    typedef Components::Common::Frame Frame;
    virtual ~Context()
    {}


    /// @name Getters
    /// @{
    /// Gravity in the local coordinate system
    virtual const Vec& getGravity() const;

    /// Projection from the local coordinate system to the world coordinate system
    virtual const Frame& getLocalToWorld() const;

    /// Projection from the world coordinate system to the local coordinate system
    virtual const Frame& getWorldToLocal() const;

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
    virtual void setLocalToWorld( const Frame& f );

    /// Projection from the local frame to the world frame
    virtual void setWorldToLocal( const Frame& f );

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


    void copyContextFrom( const Context* f );

    static Context getDefault();

    friend std::ostream& operator << (std::ostream& out, const Context& c );

private:
    Vec gravity_;
    Frame localToWorld_;
    Frame worldToLocal_;
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

