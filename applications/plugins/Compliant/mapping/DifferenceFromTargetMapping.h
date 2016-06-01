#ifndef DIFFERENCEFROMTARGETMAPPING_H
#define DIFFERENCEFROMTARGETMAPPING_H

#include <Compliant/config.h>

#include "AssembledMapping.h"
#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace mapping
{


/**
 Maps a vec to its difference with a target:

 (p, target) -> p - target

 This is used in compliant constraints to obtain relative
 violation dofs, on which a compliance may be applied
 (ie conversion to a holonomic constraint)
*/
template <class TIn, class TOut >
class SOFA_Compliant_API DifferenceFromTargetMapping : public AssembledMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(DifferenceFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));

    typedef DifferenceFromTargetMapping Self;

    typedef typename TIn::Coord InCoord;
    typedef typename TIn::VecCoord InVecCoord;

    Data< helper::vector<unsigned> > indices;         ///< indices of the parent points

    typedef helper::vector< InCoord > targets_type;
    Data< targets_type > targets;

    Data< bool > inverted;
    Data< SReal > d_showObjectScale; ///< drawing size
    Data< defaulttype::Vec4f > d_color; ///< drawing color


    DifferenceFromTargetMapping()
        : indices( initData(&indices, "indices", "indices of the parent points") )
        , targets( initData(&targets, "targets", "target positions which who computes deltas") )
        , inverted( initData(&inverted, false, "inverted", "target-p (rather than p-target)") )
        , d_showObjectScale(initData(&d_showObjectScale, SReal(0), "showObjectScale", "Scale for object display"))
        , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    {

        // backward compatibility with OffsetMapping, a previous identical mapping
        this->addAlias(&targets, "offsets");
    }

    enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

    virtual void init()
    {
        const helper::vector<unsigned>& ind = indices.getValue();
        if( ind.empty() ) this->toModel->resize( this->fromModel->getSize() );
        else this->toModel->resize( ind.size() );

        // if targets is empty, set it with actual positions
        targets_type& t = *targets.beginEdit();
        if( t.empty() )
        {
            helper::ReadAccessor<Data<InVecCoord> > x = this->fromModel->read(core::ConstVecCoordId::position());
            if( ind.empty() )
            {
                t.resize( this->fromModel->getSize() );
                for( size_t j = 0 ; j < t.size() ; ++j )
                    t[j] = x[j];
            }
            else
            {
                t.resize( ind.size() );
                for( size_t j = 0 ; j < ind.size() ; ++j )
                {
                    const unsigned k = ind[j];
                    t[j] = x[k];
                }
            }
        }
        targets.endEdit();

        Inherit1::init();
    }

    virtual void apply(typename Self::out_pos_type& out,
                       const typename Self::in_pos_type& in )
    {
        const targets_type& t = targets.getValue();
        const helper::vector<unsigned>& ind = indices.getValue();

        if( ind.empty() )
        {
            if( out.size()!=in.size() ) this->toModel->resize( this->fromModel->getSize() );
            if( inverted.getValue() )
                for( size_t j = 0 ; j < in.size() ; ++j )
                {
                    out[j] = t[std::min(t.size()-1,j)] - in[j];
                }
            else
                for( size_t j = 0 ; j < in.size() ; ++j )
                {
                    out[j] = in[j] - t[std::min(t.size()-1,j)];
                }
        }
        else
        {
            if( out.size()!=in.size() ) this->toModel->resize( ind.size() );
            if( inverted.getValue() )
                for( size_t j = 0 ; j < ind.size() ; ++j )
                {
                    const unsigned k = ind[j];
                    out[j] = t[std::min(t.size()-1,j)] - in[k];
                }
            else
                for( size_t j = 0 ; j < ind.size() ; ++j )
                {
                    const unsigned k = ind[j];
                    out[j] = in[k] - t[std::min(t.size()-1,j)];
                }
        }
    }

    virtual void assemble( const typename Self::in_pos_type& in )
    {
        assert( Nout==Nin ); // supposing TIn==TOut

        const helper::vector<unsigned>& ind = indices.getValue();
        typename Self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

        if( ind.empty() )
        {
            J.resize( Nout * in.size(), Nin * in.size());
            J.setIdentity();
            if( inverted.getValue() ) J *= -1;
        }
        else
        {
            J.resize( Nout * ind.size(), Nin * in.size());
            J.reserve( Nout * ind.size() );

            const int value = inverted.getValue() ? -1 : 1;

            for( size_t j = 0 ; j < ind.size() ; ++j )
            {
                const unsigned k = ind[j];
                for( size_t w=0 ; w<Nout ; ++w )
                {
                    const size_t line = j*Nout+w;
                    const size_t col = k*Nout+w;
                    J.startVec( line );
                    J.insertBack( line, col ) = value;
                }
            }
        }
    }


    void draw(const core::visual::VisualParams* vparams)
    {
        if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

        typename core::behavior::MechanicalState<TIn>::ReadVecCoord pos = this->getFromModel()->readPositions();
        const targets_type& t = targets.getValue();
        const helper::vector<unsigned>& ind = indices.getValue();

        helper::vector< sofa::defaulttype::Vector3 > points;
        if( ind.empty() )
        {
            points.resize(2*pos.size());
            for( size_t j = 0 ; j < pos.size() ; ++j )
            {
                points[2*j] = t[std::min(t.size()-1,j)];
                points[2*j+1] = pos[j];
            }
        }
        else
        {
            points.resize(2*ind.size());
            for( size_t j = 0 ; j < ind.size() ; ++j )
            {
                const unsigned k = ind[j];
                points[2*j] = t[std::min(t.size()-1,j)];
                points[2*j+1] = pos[k];
            }
        }

        SReal scale = d_showObjectScale.getValue();
        if( scale == 0 )
            vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
        else
            for (unsigned int i=0; i<points.size()/2; ++i)
                vparams->drawTool()->drawCylinder( points[2*i+1], points[2*i], scale, d_color.getValue() );
    }


};




}
}
}


#endif
