#ifndef SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_INL
#define SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_INL

#include <sofa/component/engine/RigidToQuatEngine.h>

namespace sofa
{

namespace component
{

namespace engine
{
template <class DataTypes>
RigidToQuatEngine<DataTypes>::RigidToQuatEngine()
    : f_positions( initData (&f_positions, "positions", "Positions (Vector of 3)") )
    , f_orientations( initData (&f_orientations, "orientations", "Orientations (Quaternion)") )
    , f_rigids( initData (&f_rigids, "rigids", "Rigid (Position + Orientation)") )
{
    //
    addAlias(&f_positions,"position");
    addAlias(&f_orientations,"orientation");
    addAlias(&f_rigids,"rigid");
}

template <class DataTypes>
RigidToQuatEngine<DataTypes>::~RigidToQuatEngine()
{

}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::init()
{
    addInput(&f_rigids);

    addOutput(&f_positions);
    addOutput(&f_orientations);

    setDirtyValue();
}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::reinit()
{
    update();
}

template <class DataTypes>
void RigidToQuatEngine<DataTypes>::update()
{
    cleanDirty();

    helper::ReadAccessor< Data< helper::vector<RigidVec3> > > rigids = f_rigids;
    helper::WriteAccessor< Data< helper::vector<Vec3> > > positions = f_positions;
    helper::WriteAccessor< Data< helper::vector<Quat> > > orientations = f_orientations;

    unsigned int sizeRigids = rigids.size();
    positions.resize(sizeRigids);
    orientations.resize(sizeRigids);
    for (unsigned int i=0 ; i< sizeRigids ; i++)
    {
        RigidVec3 r = rigids[i];
        positions[i] = r.getCenter();
        orientations[i] = r.getOrientation();
    }
}

} // namespace engine

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_ENGINE_RIGIDTOQUATENGINE_INL
