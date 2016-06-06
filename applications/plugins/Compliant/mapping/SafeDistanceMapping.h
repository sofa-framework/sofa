#ifndef SafeDistanceMAPPING_H
#define SafeDistanceMAPPING_H

#include <Compliant/config.h>

#include "AssembledMapping.h"
#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Maps two dofs to their Euclidian Distance

 (p1, p2) -> || p2.t - p1.t || - restLength
 (with .t the translation obtained by DataTypes::getCPos)

 The 'Safe' part is when the distance is too close to 0 i.e. the direction is undefined
 AND the restLength is null.
 In that case the output becomes a difference  (p1, p2) -> p2.t - p1.t
 @warning the output size can then be different.

 @author Matthieu Nesme
 @date 2016

*/
template <class TIn, class TOut >
class SOFA_Compliant_API SafeDistanceMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(SafeDistanceMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
    typedef SafeDistanceMapping self;
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
    typedef helper::vector< index_pair > pairs_type;

    Data< pairs_type > d_pairs;
    Data< helper::vector< SReal > > d_restLengths;

    Data< SReal > d_epsilonLength;

    Data< unsigned > d_geometricStiffness; ///< how to compute geometric stiffness (0->no GS, 1->exact GS, 2->stabilized GS)

    Data< SReal > d_showObjectScale; ///< drawing size
    Data< defaulttype::Vec4f > d_color; ///< drawing color

protected:

    typedef defaulttype::Vec<TIn::spatial_dimensions,SReal> Direction;

    helper::vector<Direction> m_directions;   ///< Unit vectors in the directions of the lines
    helper::vector<SReal> m_lengths, m_invlengths;       ///< inverse of current distances. Null represents the infinity (null distance)


	
    SafeDistanceMapping()
        : d_pairs( initData(&d_pairs, "pairs", "index pairs for computing deltas") )
        , d_restLengths( initData(&d_restLengths, "restLengths", "rest lengths") )
        , d_epsilonLength( initData(&d_epsilonLength, 1e-4, "epsilonLength", "Threshold to consider a length too close to 0") )
        , d_geometricStiffness( initData(&d_geometricStiffness, 2u, "geometricStiffness", "0 -> no GS, 1 -> exact GS, 2 -> stabilized GS (default)") )
        , d_showObjectScale(initData(&d_showObjectScale, SReal(-1), "showObjectScale", "Scale for object display"))
        , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    {}

    enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

public:

    virtual void init()
    {
        reinit();
        Inherit1::init();
    }

    virtual void reinit()
    {
        const pairs_type& pairs = d_pairs.getValue();
        helper::vector<SReal>& restLengths = *d_restLengths.beginEdit();

        typename core::behavior::MechanicalState<TIn>::ReadVecCoord pos = this->getFromModel()->readPositions();

        m_directions.resize(pairs.size());
        m_lengths.resize(pairs.size());
        m_invlengths.resize(pairs.size());

        // compute the current rest lengths
        if( restLengths.empty() )
        {
            restLengths.resize( pairs.size() );
            for(unsigned i=0; i<pairs.size(); i++ )
                restLengths[i] = (pos[pairs[i][0]] - pos[pairs[i][1]]).norm();
        }
        else if( restLengths.size() != pairs.size() ) // repeteting the last given restLength
        {
            SReal last = restLengths.back();
            restLengths.reserve( pairs.size() );
            for( size_t i=restLengths.size(), n=pairs.size() ; i<n ; ++i )
                restLengths.push_back(last);
        }

        d_restLengths.endEdit();
        Inherit1::reinit();
    }

	virtual void apply(typename self::out_pos_type& out, 
	                   const typename self::in_pos_type& in )  {
		assert( this->Nout == this->Nin );

        const pairs_type& pairs = d_pairs.getValue();
        const helper::vector<SReal>& restLengths = d_restLengths.getValue();
        const SReal& epsilon = d_epsilonLength.getValue();

        size_t size = 0;

        for( unsigned i = 0, n = pairs.size(); i < n; ++i)
        {
            Direction& gap = m_directions[i];

             gap = in[pairs[i][1]] - in[pairs[i][0]]; // (only for position)
//            computeCoordPositionDifference( gap, in[links[i][0]], in[links[i][1]] ); // todo for more complex types such as Rigids

             m_lengths[i] = gap.norm();


             // normalize
             if( m_lengths[i] > epsilon )
             {
                 m_invlengths[i] = 1/m_lengths[i] ;
                 gap *= m_invlengths[i];

                 size++;
             }
             else
             {
                 m_invlengths[i] = 0;

                 if( restLengths[i] != 0 )
                 {
                    // arbritary vector mapping all directions
                     SReal p = 1.0f/std::sqrt((SReal)TIn::spatial_dimensions);
                     for( unsigned j=0;j<TIn::spatial_dimensions;++j)
                         gap[j]=p;

                     size++;
                 }
                 else
                 {
                     size += TIn::spatial_dimensions; // for the difference
                 }
             }



        }

        this->getToModel()->resize( size );

        for( unsigned i = 0, n = pairs.size(), s=0 ; i < n; ++i)
        {

            Direction& gap = m_directions[i];

            if( m_invlengths[i] != 0 || restLengths[i] )
            {
                out[s] = m_lengths[i]-restLengths[i];
                s++;
            }
            else
            {
                for(unsigned j=0;j<TIn::spatial_dimensions;++j)
                    out[s+j] = gap[j];
                s += TIn::spatial_dimensions;
            }
        }

	}

    virtual void assemble( const typename self::in_pos_type& in )
    {
        size_t size = this->getToModel()->getSize();

        const pairs_type& pairs = d_pairs.getValue();
        const helper::vector<SReal>& restLengths = d_restLengths.getValue();

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

        J.resize( size, Nin * in.size() );
        J.reserve( pairs.size() * TIn::spatial_dimensions );

        for( unsigned i = 0, n = pairs.size(), s=0 ; i < n; ++i)
        {
            Direction& gap = m_directions[i];


            if( m_invlengths[i] != 0 || restLengths[i] )
            {
                J.startVec(s);

                // insert in increasing column order
                if( pairs[i][1]<pairs[i][0])
                {
                    for(unsigned k=0; k<TIn::spatial_dimensions; k++ )
                        J.insertBack( s, pairs[i][1]*Nin+k ) = gap[k];
                    for(unsigned k=0; k<TIn::spatial_dimensions; k++ )
                        J.insertBack( s, pairs[i][0]*Nin+k ) = -gap[k];
                }
                else
                {
                    for(unsigned k=0; k<TIn::spatial_dimensions; k++ )
                        J.insertBack( s, pairs[i][0]*Nin+k ) = -gap[k];
                    for(unsigned k=0; k<TIn::spatial_dimensions; k++ )
                        J.insertBack( s, pairs[i][1]*Nin+k ) = gap[k];
                }

                s++;
            }
            else
            {
                for(unsigned j=0; j<TIn::spatial_dimensions; j++ )
                {
                    J.startVec(s+j);

                    if( pairs[i][1]==pairs[i][0] ) continue;

                    if( pairs[i][1]<pairs[i][0])
                    {
                        J.insertBack(s+j, pairs[i][1] * Nin + j ) = 1;
                        J.insertBack(s+j, pairs[i][0] * Nin + j ) = -1;
                    }
                    else
                    {
                        J.insertBack(s+j, pairs[i][0] * Nin + j ) = -1;
                        J.insertBack(s+j, pairs[i][1] * Nin + j ) = 1;
                    }
                }

                s += TIn::spatial_dimensions;
            }
        }

        J.finalize();
	}


    virtual void assemble_geometric( const typename self::in_pos_type& in, const typename self::out_force_type& out )
    {
        typename self::geometric_type& K = this->geometric;
        const unsigned& geometricStiffness = d_geometricStiffness.getValue();
        if( !geometricStiffness ) { K.resize(0,0); return; }


        const pairs_type& pairs = d_pairs.getValue();
        const helper::vector<SReal>& restLengths = d_restLengths.getValue();

        K.resizeBlocks(in.size(),in.size());
        for(size_t i=0; i<pairs.size(); i++)
        {
            if( !m_invlengths[i] ) continue;

            // force in compression (>0) can lead to negative eigen values in geometric stiffness
            // this results in a undefinite implicit matrix that causes instabilies
            // if stabilized GS (geometricStiffness==2) -> keep only force in extension
            if( out[i][0] < 0 || geometricStiffness==1 )
            {
                sofa::defaulttype::Mat<Nin,Nin,SReal> b;  // = (I - uu^T)

                for(unsigned j=0; j<TIn::spatial_dimensions; j++)
                {
                    for(unsigned k=0; k<TIn::spatial_dimensions; k++)
                    {
                        if( j==k )
                            b[j][k] = 1.f - m_directions[i][j]*m_directions[i][k];
                        else
                            b[j][k] =     - m_directions[i][j]*m_directions[i][k];
                    }
                }
                b *= out[i][0] * m_invlengths[i];  // (I - uu^T)*f/l

                // Note that 'links' is not sorted so the matrix can not be filled-up in order
                K.addBlock(pairs[i][0],pairs[i][0],b);
                K.addBlock(pairs[i][0],pairs[i][1],-b);
                K.addBlock(pairs[i][1],pairs[i][0],-b);
                K.addBlock(pairs[i][1],pairs[i][1],b);
            }
        }
        K.compress();
    }

    void draw(const core::visual::VisualParams* vparams)
    {

#ifndef SOFA_NO_OPENGL
        if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

        SReal scale = d_showObjectScale.getValue();

        if( scale < 0 ) return;

        glEnable(GL_LIGHTING);

        typename core::behavior::MechanicalState<TIn>::ReadVecCoord pos = this->getFromModel()->readPositions();
        const pairs_type& p = d_pairs.getValue();

        if( !scale )
        {
            helper::vector< defaulttype::Vector3 > points(p.size()*2);
            for(unsigned i=0; i<p.size(); i++ )
            {
                points[i*2  ] = defaulttype::Vector3( TIn::getCPos(pos[p[i][0]]) );
                points[i*2+1] = defaulttype::Vector3( TIn::getCPos(pos[p[i][1]]) );
            }
            vparams->drawTool()->drawLines ( points, 1, d_color.getValue() );
        }
        else
        {
            for(unsigned i=0; i<p.size(); i++ )
            {
                defaulttype::Vector3 p0 = defaulttype::Vector3( TIn::getCPos(pos[p[i][0]]) );
                defaulttype::Vector3 p1 = defaulttype::Vector3( TIn::getCPos(pos[p[i][1]]) );
                vparams->drawTool()->drawCylinder( p0, p1, d_showObjectScale.getValue(), d_color.getValue() );
            }
        }
#endif /* SOFA_NO_OPENGL */
    }

    virtual void updateForceMask()
    {
        const pairs_type& p = d_pairs.getValue();

        for( size_t i = 0, iend = p.size(); i < iend; ++i )
        {
            if( this->maskTo->getEntry(i) )
            {
                const index_pair& indices = p[i];
                this->maskFrom->insertEntry(indices[0]);
                this->maskFrom->insertEntry(indices[1]);
            }
        }
    }
	
};



//////////////////////




}
}
}


#endif
