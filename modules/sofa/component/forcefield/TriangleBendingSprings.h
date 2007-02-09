//
// C++ Interface: TriangleBendingSprings
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2007
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef SOFA_COMPONENT_FORCEFIELD_TRIANGLEBENDINGSPRINGS_H
#define SOFA_COMPONENT_FORCEFIELD_TRIANGLEBENDINGSPRINGS_H

#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/MechanicalObject.h>
#include <map>

namespace sofa
{

namespace component
{

namespace forcefield
{

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.


	@author The SOFA team </www.sofa-framework.org>
*/
template<class DataTypes>
class TriangleBendingSprings : public sofa::component::forcefield::StiffSpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;

    TriangleBendingSprings();

    ~TriangleBendingSprings();

    /// Searches triangle topology and creates the bending springs
    virtual void init();

    DataField<Real> stiffness;
    DataField<Real> dampingRatio;

    virtual void draw()
    {
    }

protected:
    typedef std::pair<unsigned,unsigned> IndexPair;
    void addSpring( unsigned, unsigned );
    void registerTriangle( unsigned, unsigned, unsigned, std::map<IndexPair, unsigned>& );
    component::MechanicalObject<DataTypes>* dof;

};

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
