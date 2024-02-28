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
#ifndef SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#define SOFA_COMPONENT_MISC_PARTICLESOURCE_H
#include <SofaSphFluid/config.h>

#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/topology/TopologySubsetIndices.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/type/trait/Rebind.h>


namespace sofa::component::misc
{

template<class DataTypes>
class ParticleSource : public core::behavior::ProjectiveConstraintSet<DataTypes>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE(ParticleSource,DataTypes), SOFA_TEMPLATE(core::behavior::ProjectiveConstraintSet,DataTypes));

    using Index = sofa::Index;
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Deriv Deriv;
    typedef typename DataTypes::VecDeriv VecDeriv;
    typedef typename DataTypes::MatrixDeriv MatrixDeriv;
    typedef typename DataTypes::MatrixDeriv::RowType MatrixDerivRowType;
    typedef type::vector<Real> VecDensity;

    typedef Data<VecCoord> DataVecCoord;
    typedef Data<VecDeriv> DataVecDeriv;
    typedef Data<MatrixDeriv> DataMatrixDeriv;
    //int lastparticle;
    using VecIndex = type::rebind_to<VecCoord, Index>;
    typedef sofa::core::topology::TopologySubsetIndices SetIndex;
    typedef typename SetIndex::container_type SetIndexArray;

    typedef core::behavior::MechanicalState<DataTypes> MechanicalModel;

    ParticleSource();

    virtual ~ParticleSource();

    void init() override;

    void reset() override;


    Real rrand()
    {
        return (Real)(rand()*1.0 / RAND_MAX);
    }

    virtual void animateBegin(double /*dt*/, double time);
    
    //template <class DataDeriv>
    //void projectResponseT(DataDeriv& res) ///< project dx to constrained space
    //{
    //    msg_info() << "ParticleSource<DataTypes>::projectResponseT()";
    //    if (!this->mstate) return;
    //    if (m_lastparticles.getValue().empty()) return;
    //    //msg_info() << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
    //    double time = this->getContext()->getTime();
    //    if (time < d_start.getValue() || time > d_stop.getValue()) return;

    //    helper::ReadAccessor<Data<VecIndex> > _lastparticles = this->m_lastparticles; ///< lastparticles indices
    //    // constraint the last value
    //    for (unsigned int s=0; s<_lastparticles.size(); s++)
    //    {
    //        //HACK: TODO understand why these conditions can be reached
    //        if (_lastparticles[s] >= (unsigned int) this->mstate->getSize()) continue;

    //        res[_lastparticles[s]] = Deriv();
    //    }
    //}

    //using core::behavior::ProjectiveConstraintSet<DataTypes>::projectResponse;
    //void projectResponse(VecDeriv& dx)
    //{
    //    projectResponseT(dx);
    //}

    void projectResponse(const sofa::core::MechanicalParams* mparams, DataVecDeriv& dxData) override; ///< project dx to constrained space
        
    void projectVelocity(const sofa::core::MechanicalParams* mparams, DataVecDeriv&  vData) override; ///< project dx to constrained space (dx models a velocity) override    

    void projectPosition(const sofa::core::MechanicalParams* mparams, DataVecCoord& xData) override; ///< project x to constrained space (x models a position) override
    

    void projectJacobianMatrix(const sofa::core::MechanicalParams* /*mparams*/, DataMatrixDeriv& /* cData */) override
    {

    }

    void handleEvent(sofa::core::objectmodel::Event* event) override;

    void draw(const core::visual::VisualParams* vparams) override;

public:
    Data< Coord > d_translation; ///< translation applied to center(s)
    Data< Real > d_scale; ///< scale applied to center(s)
    Data< type::vector<Coord> > d_center; ///< Source center(s)
    Data< Coord > d_radius; ///< Source radius
    Data< Deriv > d_velocity; ///< Particle initial velocity
    Data< Real > d_delay; ///< Delay between particles creation
    Data< Real > d_start; ///< Source starting time
    Data< Real > d_stop; ///< Source stopping time
    Data< bool > d_addNoise; ///< Will add random value to the radius of new created particles

protected:    
    size_t m_numberParticles; ///< Number particles given by the initial particles size
    Real m_lastTime; ///< Last time particle have been computed
    Real m_maxdist;

    sofa::core::topology::TopologySubsetIndices m_lastparticles; ///< lastparticles indices
    VecCoord m_lastpos;

};

#if !defined(SOFA_COMPONENT_MISC_PARTICLESOURCE_CPP)
extern template class SOFA_SPH_FLUID_API ParticleSource<sofa::defaulttype::Vec3Types>;
extern template class SOFA_SPH_FLUID_API ParticleSource<sofa::defaulttype::Vec2Types>;
#endif

} // namespace sofa::component::misc


#endif

