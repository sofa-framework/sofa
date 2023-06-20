/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/component/solidmechanics/spring/VectorSpringForceField.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/io/XspLoader.h>
#include <sofa/core/topology/TopologyData.inl>


namespace sofa::component::solidmechanics::spring
{

template<class DataTypes>
void VectorSpringForceField<DataTypes>::createEdgeInformation(Index, Spring &t,
        const core::topology::BaseMeshTopology::Edge & e,
        const sofa::type::vector<Index> & ancestors,
        const sofa::type::vector<SReal> & coefs)
{
    const typename DataTypes::VecCoord& x0 = this->getObject1()->read(core::ConstVecCoordId::restPosition())->getValue();
    t.restVector = x0[e[1]] - x0[e[0]];
    if (ancestors.size()>0)
    {
        t.kd=t.ks=0;
        const type::vector<Spring> &sa=getSpringArray().getValue();
        unsigned int i;
        for (i=0; i<ancestors.size(); ++i)
        {
            t.kd+=typename DataTypes::Real(sa[i].kd*coefs[i]);
            t.ks+=typename DataTypes::Real(sa[i].ks*coefs[i]);
        }
    }
    else
    {
        t.kd=getStiffness();
        t.ks=getViscosity();
    }
}

template <class DataTypes>
class VectorSpringForceField<DataTypes>::Loader : public sofa::helper::io::XspLoaderDataHook
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    VectorSpringForceField<DataTypes>* dest;
    Loader(VectorSpringForceField<DataTypes>* dest) : dest(dest) {}
    virtual void addVectorSpring(size_t m1, size_t m2, SReal ks, SReal kd, SReal initpos, SReal restx, SReal resty, SReal restz)
    {
        SOFA_UNUSED(initpos);
        dest->addSpring(m1,m2,ks,kd,Coord(Real(restx),Real(resty),Real(restz)));
    }
};

template <class DataTypes>
bool VectorSpringForceField<DataTypes>::load(const char *filename)
{
    if (filename && filename[0])
    {
        Loader loader(this);
        return helper::io::XspLoader::Load(filename, loader);
    }
    else return false;
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::resizeArray(std::size_t n)
{
    type::vector<Spring>& springArrayData = *(springArray.beginEdit());
    springArrayData.resize(n);
    springArray.endEdit();
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::addSpring(int m1, int m2, SReal ks, SReal kd, Coord restVector)
{
    type::vector<Spring>& springArrayData = *(springArray.beginEdit());

    if (useTopology && m_topology)
    {
        sofa::core::topology::BaseMeshTopology::EdgeID e = m_topology->getEdgeIndex(unsigned(m1),unsigned(m2));
        if (e != sofa::InvalidID)
            springArrayData[e]=Spring(Real(ks),Real(kd),restVector);
    }
    else
    {
        springArrayData.push_back(Spring(Real(ks),Real(kd),restVector));
        edgeArray.push_back(core::topology::BaseMeshTopology::Edge(m1,m2));
    }
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField()
    : VectorSpringForceField(nullptr, nullptr)
{
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField(MechanicalState* _object)
    : VectorSpringForceField(_object, _object)
{
}

template <class DataTypes>
VectorSpringForceField<DataTypes>::VectorSpringForceField(MechanicalState* _object1, MechanicalState* _object2)
    : Inherit(_object1, _object2)
    , m_potentialEnergy( 0.0 ), useTopology( false )
    , springArray( initData(&springArray, "springs", "springs data"))
    , m_filename( initData(&m_filename,std::string(""),"filename","File name from which the spring informations are loaded") )
    , m_stiffness( initData(&m_stiffness,SReal(1.0),"stiffness","Default edge stiffness used in absence of file information") )
    , m_viscosity( initData(&m_viscosity, SReal(1.0),"viscosity","Default edge viscosity used in absence of file information") )
    , m_useTopology( initData(&m_useTopology, false, "useTopology", "Activate/Desactivate topology mode of the component (springs on each edge)"))
    , l_topology(initLink("topology", "link to the topology container"))    
    , m_topology(nullptr)
{
}

template<class DataTypes>
VectorSpringForceField<DataTypes>::~VectorSpringForceField()
{

}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::init()
{
    if (m_useTopology.getValue()) 
    {
        if (l_topology.empty())
        {
            msg_info() << "link to Topology container should be set to ensure right behavior. First Topology found in current context will be used.";
            l_topology.set(this->getContext()->getMeshTopologyLink());
        }

        m_topology = l_topology.get();
        msg_info() << "Topology path used: '" << l_topology.getLinkedPath() << "'";

        if (m_topology == nullptr)
        {
            msg_error() << "No topology component found at path: " << l_topology.getLinkedPath() << ", nor in current context: " << this->getContext()->name;
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    if(m_topology)
    {
        springArray.createTopologyHandler(m_topology);
        springArray.setCreationCallback([this](Index edgeIndex, Spring& t,
            const core::topology::BaseMeshTopology::Edge& e,
            const sofa::type::vector<Index>& ancestors,
            const sofa::type::vector<SReal>& coefs)
        {
            createEdgeInformation(edgeIndex, t, e, ancestors, coefs);
        });
    }

    this->Inherit::init();
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::bwdInit()
{
    this->Inherit::bwdInit();
    type::vector<Spring>& springArrayData = *(springArray.beginEdit());

    if (springArrayData.empty())
    {
        if (!m_filename.getValue().empty())
        {
            // load the springs from a file
            load(m_filename.getFullPath().c_str());
            return;
        }

        if (m_topology != nullptr)
        {
            // create springs based on the mesh topology
            useTopology = true;
            createDefaultSprings();
            this->f_listening.setValue(true);
        }
        else
        {
            int n = this->mstate1->getSize();
            msg_info() << "VectorSpringForceField: linking "<<n<<" pairs of points.";
            springArrayData.resize(n);
            edgeArray.resize(n);
            for (int i=0; i<n; ++i)
            {
                edgeArray[i][0] = i;
                edgeArray[i][1] = i;
                springArrayData[i].ks=Real(m_stiffness.getValue());
                springArrayData[i].kd=Real(m_viscosity.getValue());
                springArrayData[i].restVector = Coord();
            }
        }
        springArray.endEdit();
    }
}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::createDefaultSprings()
{
    msg_info() << "Creating "<< m_topology->getNbEdges() <<" Vector Springs from EdgeSetTopology";

    type::vector<Spring>& springArrayData = *(springArray.beginEdit());

    springArrayData.resize(m_topology->getNbEdges());
    const VecCoord& x0 = this->mstate1->read(core::ConstVecCoordId::restPosition())->getValue();
    unsigned int i;
    for (i=0; i<m_topology->getNbEdges(); ++i)
    {
        springArrayData[i].ks=Real(m_stiffness.getValue());
        springArrayData[i].kd=Real(m_viscosity.getValue());
        springArrayData[i].restVector = x0[m_topology->getEdge(i)[1]]-x0[m_topology->getEdge(i)[0]];
    }

    springArray.endEdit();
}

template<class DataTypes>
void VectorSpringForceField<DataTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& data_f1, DataVecDeriv& data_f2, const DataVecCoord& data_x1, const DataVecCoord& data_x2, const DataVecDeriv& data_v1, const DataVecDeriv& data_v2 )
{

    VecDeriv&       f1 = *data_f1.beginEdit();
    const VecCoord& x1 =  data_x1.getValue();
    const VecDeriv& v1 =  data_v1.getValue();
    VecDeriv&       f2 = *data_f2.beginEdit();
    const VecCoord& x2 =  data_x2.getValue();
    const VecDeriv& v2 =  data_v2.getValue();

    m_potentialEnergy = 0;

    f1.resize(x1.size());
    f2.resize(x2.size());

    type::vector<Spring>& springArrayData = *(springArray.beginEdit());

    if(useTopology)
    {

        Deriv force;
        for (unsigned int i=0; i<m_topology->getNbEdges(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=m_topology->getEdge(i);
            const Spring &s=springArrayData[i];
            // paul---------------------------------------------------------------
            Deriv current_direction = x2[e[1]]-x1[e[0]];
            Deriv squash_vector = current_direction - s.restVector;
            Deriv relativeVelocity = v2[e[1]]-v1[e[0]];
            force = (squash_vector * s.ks) + (relativeVelocity * s.kd);

            f1[e[0]]+=force;
            f2[e[1]]-=force;
        }

    }
    else
    {

        Deriv force;
        for (size_t i=0; i<edgeArray.size(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=edgeArray[i];
            const Spring &s=springArrayData[i];
            // paul---------------------------------------------------------------
            Deriv current_direction = x2[e[1]]-x1[e[0]];
            Deriv squash_vector = current_direction - s.restVector;
            Deriv relativeVelocity = v2[e[1]]-v1[e[0]];
            force = (squash_vector * s.ks) + (relativeVelocity * s.kd);

            f1[e[0]]+=force;
            f2[e[1]]-=force;
        }
    }
    springArray.endEdit();

    data_f1.endEdit();
    data_f2.endEdit();
}

template<class DataTypes>
void VectorSpringForceField<DataTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& data_df1, DataVecDeriv& data_df2, const DataVecDeriv& data_dx1, const DataVecDeriv& data_dx2)
{
    VecDeriv&        df1 = *data_df1.beginEdit();
    VecDeriv&        df2 = *data_df2.beginEdit();
    const VecDeriv&  dx1 =  data_dx1.getValue();
    const VecDeriv&  dx2 =  data_dx2.getValue();
    Real kFactor       =  Real(sofa::core::mechanicalparams::kFactorIncludingRayleighDamping(mparams, this->rayleighStiffness.getValue()));

    Deriv dforce,d;

    df1.resize(dx1.size());
    df2.resize(dx2.size());

    const type::vector<Spring>& springArrayData = springArray.getValue();

    if(useTopology)
    {

        for (unsigned int i=0; i<m_topology->getNbEdges(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=m_topology->getEdge(i);
            const Spring &s=springArrayData[i];
            d = dx2[e[1]]-dx1[e[0]];
            dforce = d*(s.ks*kFactor);
            df1[e[0]]+=dforce;
            df2[e[1]]-=dforce;
        }

    }
    else
    {

        for (size_t i=0; i<edgeArray.size(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=edgeArray[i];
            const Spring &s=springArrayData[i];
            d = dx2[e[1]]-dx1[e[0]];
            dforce = d*(s.ks*kFactor);
            df1[e[0]]+=dforce;
            df2[e[1]]-=dforce;
        }
    }

    data_df1.endEdit();
    data_df2.endEdit();

}

template <class DataTypes>
void VectorSpringForceField<DataTypes>::buildDampingMatrix(core::behavior::DampingMatrix*)
{
    // No damping in this ForceField
}

template<class DataTypes>
void VectorSpringForceField<DataTypes>::draw(const core::visual::VisualParams* vparams)
{
    using namespace sofa::type;
    using namespace sofa::defaulttype;

    if (!((this->mstate1 == this->mstate2)?vparams->displayFlags().getShowForceFields():vparams->displayFlags().getShowInteractionForceFields()))
        return;
    //const VecCoord& p = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x1 =this->mstate1->read(core::ConstVecCoordId::position())->getValue();
    const VecCoord& x2 =this->mstate2->read(core::ConstVecCoordId::position())->getValue();


    std::vector< Vec3 > points;
    if(useTopology)
    {
        for (unsigned int i=0; i<springArray.getValue().size(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=m_topology->getEdge(i);
            //const Spring &s=springArray[i];

            points.push_back(Vec3(x1[e[0]]));
            points.push_back(Vec3(x2[e[1]]));
        }

    }
    else
    {

        for (unsigned int i=0; i<springArray.getValue().size(); i++)
        {
            const core::topology::BaseMeshTopology::Edge &e=edgeArray[i];
            //const Spring &s=springArray[i];

            points.push_back(Vec3(x1[e[0]]));
            points.push_back(Vec3(x2[e[1]]));
        }
    }
    vparams->drawTool()->drawLines(points, 3, sofa::type::RGBAColor::red());
}

} // namespace sofa::component::solidmechanics::spring
