/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#define SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#include <SofaSphFluid/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <SofaBaseTopology/TopologySubsetData.inl>
#include <SofaBaseTopology/PointSetTopologyModifier.h>
#include <sofa/core/topology/TopologyChange.h>
#include <vector>
#include <iterator>
#include <iostream>
#include <ostream>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace misc
{

template<class DataTypes>
class ParticleSource : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticleSource,DataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));

    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef helper::vector<Real> VecDensity;

    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalModel;



    ParticleSource();

    virtual ~ParticleSource();

    Real rrand()
    {
        return (Real)(rand()*1.0 / RAND_MAX);
    }

    int N;
    Real lasttime;
    Real maxdist;
    //int lastparticle;
    typedef typename VecCoord::template rebind<unsigned int>::other VecIndex;
    sofa::component::topology::PointSubsetData< VecIndex > lastparticles; ///< lastparticles indices
    VecCoord lastpos;

    class PSPointHandler : public sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, VecIndex >
    {
    public:
        typedef typename ParticleSource<DataTypes>::VecIndex VecIndex;
        typedef VecIndex container_type;
        typedef typename container_type::value_type value_type;

        PSPointHandler(ParticleSource<DataTypes>* _ps, sofa::component::topology::PointSubsetData<VecIndex >* _data)
            : sofa::component::topology::TopologySubsetDataHandler<core::topology::BaseMeshTopology::Point, VecIndex >(_data), ps(_ps) {}

        void applyDestroyFunction(unsigned int index, value_type& /*T*/)
        {
            dmsg_info("ParticleSource") << "PSRemovalFunction";
            if(ps)
            {
                /*topology::PointSubset::const_iterator it = std::find(ps->lastparticles.begin(),ps->lastparticles.end(), (unsigned int)index);
                 if (it != ps->lastparticles.end())
                 {
                    ps->lastpos.erase( ps->lastpos.begin()+(it-ps->lastparticles.begin()) );
                    //ps->lastparticles.getArray().erase(it);
                     helper::removeValue(ps->lastparticles,(unsigned int)index);
                 }*/
                VecIndex& _lastparticles = *ps->lastparticles.beginEdit();

                unsigned int size = _lastparticles.size();
                for (unsigned int i = 0; i < size; ++i)
                {
                    if ((unsigned int)_lastparticles[i] == index)
                    {
                        if (i < size-1)
                        {
                            _lastparticles[i] = _lastparticles[size-1];
                            ps->lastpos[i] = ps->lastpos[size-1];
                        }
                        _lastparticles.pop_back();
                        ps->lastpos.pop_back();
                        return;
                    }
                }
                ps->lastparticles.endEdit();
            }
        }


        bool applyTestCreateFunction(unsigned int /*index*/,
                const sofa::helper::vector< unsigned int > & /*ancestors*/,
                const sofa::helper::vector< double > & /*coefs*/) {return false;}

    protected:
        ParticleSource<DataTypes> *ps;
    };


    void init() override;
        
    void reset() override;
   
    virtual void animateBegin(double /*dt*/, double time);
    
    

    template <class DataDeriv>
    void projectResponseT(DataDeriv& res) ///< project dx to constrained space
    {
        if (!this->mstate) return;
        if (lastparticles.getValue().empty()) return;
        //msg_info() << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
        double time = this->getContext()->getTime();
        if (time < f_start.getValue() || time > f_stop.getValue()) return;

        helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->lastparticles; ///< lastparticles indices
        // constraint the last value
        for (unsigned int s=0; s<_lastparticles.size(); s++)
        {
            //HACK: TODO understand why these conditions can be reached
            if (_lastparticles[s] >= (unsigned int) this->mstate->getSize()) continue;

            res[_lastparticles[s]] = Deriv();
        }
    }

    using core::behavior::ProjectiveConstraintSet<DataTypes>::projectResponse;
    void projectResponse(VecDeriv& dx)
    {
        projectResponseT(dx);
    }

    void projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dxData) override; ///< project dx to constrained space
        
    void projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  vData) override; ///< project dx to constrained space (dx models a velocity) override    

    void projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData) override; ///< project x to constrained space (x models a position) override
    

    void projectJacobianMatrix(const sofa::core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {

    }

    virtual void animateEnd(double /*dt*/, double /*time*/)
    {

    }

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void draw(const core::visual::VisualParams* vparams) override;

public:
    Data< Coord > f_translation; ///< translation applied to center(s)
    Data< Real > f_scale; ///< scale applied to center(s)
    Data< helper::vector<Coord> > f_center; ///< Source center(s)
    Data< Coord > f_radius; ///< Source radius
    Data< Deriv > f_velocity; ///< Particle initial velocity
    Data< Real > f_delay; ///< Delay between particles creation
    Data< Real > f_start; ///< Source starting time
    Data< Real > f_stop; ///< Source stopping time
    Data< bool > f_canHaveEmptyVector;

protected:
    PSPointHandler* pointHandler;

};

#if !defined(SOFA_COMPONENT_MISC_PARTICLESOURCE_CPP)
extern template class SOFA_SPH_FLUID_API ParticleSource<sofa::defaulttype::Vec3Types>;
extern template class SOFA_SPH_FLUID_API ParticleSource<sofa::defaulttype::Vec2Types>;

#endif

} // namespace misc

} // namespace component

} // namespace sofa

#endif

