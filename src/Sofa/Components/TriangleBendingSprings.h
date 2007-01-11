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
#ifndef TriangleBendingSprings_h
#define TriangleBendingSprings_h

#include "StiffSpringForceField.h"
#include "Sofa/Core/MechanicalObject.h"
#include <map>

namespace Sofa
{

namespace Components
{

/**
Bending springs added between vertices of triangles sharing a common edge.
The springs connect the vertices not belonging to the common edge. It compresses when the surface bends along the common edge.


	@author The SOFA team </www.sofa-framework.org>
*/
template<class DataTypes>
class TriangleBendingSprings : public Sofa::Components::StiffSpringForceField<DataTypes>
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::VecCoord VecCoord;

    TriangleBendingSprings(Core::MechanicalModel<DataTypes>* object);

    ~TriangleBendingSprings();

    /// Searches triangle topology and creates the bending springs
    virtual void init();

    DataField<Real> stiffness;
    DataField<Real> dampingRatio;

protected:
    typedef std::pair<unsigned,unsigned> IndexPair;
    void addSpring( unsigned, unsigned );
    void registerTriangle( unsigned, unsigned, unsigned, std::map<IndexPair, unsigned>& );
    Core::MechanicalObject<DataTypes>* dof;

};

}

}

#endif
