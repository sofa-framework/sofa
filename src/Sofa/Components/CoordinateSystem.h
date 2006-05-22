#pragma once

//#include <Sofa/Abstract/BaseObject.h>
#include <Sofa/Components/Common/SolidTypes.h>
#include <Sofa/Core/BasicMechanicalModel.h>
#include <Sofa/Components/Common/vector.h>

namespace Sofa
{
namespace Core
{
class Context;
class Topology;
}

namespace Components
{
namespace Graph
{
class GNode;
}

using namespace Core::Encoding;

/** Defines the local coordinate system with respect to its parent.
*/
class CoordinateSystem : public Core::BasicMechanicalModel
{
public:
    typedef Common::SolidTypes<float>::Vec Vec;
    typedef Common::SolidTypes<float>::Rot Rot;
    typedef Common::SolidTypes<float>::Coord Frame;
    typedef Common::SolidTypes<float>::Deriv Velocity;

    CoordinateSystem();
    virtual ~CoordinateSystem()
    {}

    virtual void updateContext( Core::Context* );


    const Frame&  getFrame() const;
    CoordinateSystem* setFrame( const Frame& f );
    CoordinateSystem* setFrame( const Vec& translation, const Rot& rotation );
    CoordinateSystem* setFrame( const Vec& translation ) { Rot r = Rot::identity(); return setFrame(translation, r); }

    const Velocity&  getVelocity() const;
    CoordinateSystem* setVelocity( const Velocity& f );
    const Vec& getLinearVelocity() const;
    CoordinateSystem* setLinearVelocity( const Vec& linearVelocity);
    const Vec& getAngularVelocity() const;
    CoordinateSystem* setAngularVelocity( const Vec& angularVelocity );

    //=================================
    // interface of BasicMechanicalModel

    BasicMechanicalModel* resize(int )
    {return this;}

    virtual void init()
    {}
    ;

    virtual void setTopology(Core::Topology* /*topo*/)
    {}

    virtual Core::Topology* getTopology()
    {
        return NULL;
    }

    virtual void beginIteration(double /*dt*/)
    {}

    virtual void endIteration(double /*dt*/)
    {}

    virtual void propagateX()
    {}

    virtual void propagateV()
    {}

    virtual void propagateDx()
    {}

    virtual void resetForce();

    virtual void accumulateForce()
    {}

    virtual void accumulateDf()
    {}

    virtual void applyConstraints()
    {}

    /// @name Integration related methods
    /// @{

    void vAlloc(VecId )
    {}

    void vFree(VecId )
    {}

    void vOp(VecId v, VecId a = VecId::null(), VecId b = VecId::null(), double f=1.0);

    virtual double vDot(VecId a, VecId b);

    virtual void setX(VecId v);

    virtual void setV(VecId v);

    virtual void setF(VecId v);

    virtual void setDx(VecId v);

    /// @}

    /// @name Debug
    /// @{
    virtual void printDOF( VecId, std::ostream& =std::cerr );
    /// @}

    // interface of BasicMechanicalModel
    //=================================


protected:
    Frame& getX();
    const Frame& getX() const;

    Velocity& getV();
    const Velocity& getV() const;

    Velocity& getF();
    const Velocity& getF() const;

    Velocity& getDx();
    const Velocity& getDx() const;

private:
    Common::vector<Frame*> coords_;
    Common::vector<Velocity*> derivs_;
    Frame* x_;
    Velocity* v_;
    Velocity* f_;
    Velocity* dx_;
    Frame* getCoord(unsigned);
    Velocity* getDeriv(unsigned);
};

} // namespace Components

} // namespace Sofa


