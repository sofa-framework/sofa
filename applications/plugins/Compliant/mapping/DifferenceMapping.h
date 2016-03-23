#ifndef DIFFERENCEMAPPING_H
#define DIFFERENCEMAPPING_H

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
 Maps two dofs to their spatial position difference:

 (p1, p2) -> p2.t - p1.t
 (with .t the translation obtained by DataTypes::getCPos)

 This is used to obtain relative dofs
 on which a stiffness/compliance may be applied
*/
template <class TIn, class TOut >
class SOFA_Compliant_API DifferenceMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(DifferenceMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
    typedef DifferenceMapping self;
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
    typedef helper::vector< index_pair > pairs_type;

	Data< pairs_type > pairs;
    Data< SReal > d_showObjectScale; ///< drawing size
    Data< defaulttype::Vec4f > d_color; ///< drawing color



	
	DifferenceMapping() 
        : pairs( initData(&pairs, "pairs", "index pairs for computing deltas") )
        , d_showObjectScale(initData(&d_showObjectScale, SReal(0), "showObjectScale", "Scale for object display"))
        , d_color(initData(&d_color, defaulttype::Vec4f(1,1,0,1), "showColor", "Color for object display"))
    {}

	enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

    virtual void init()
    {
        this->getToModel()->resize( pairs.getValue().size() );
        AssembledMapping<TIn, TOut>::init();
    }

    virtual void reinit()
    {
        this->getToModel()->resize( pairs.getValue().size() );
        AssembledMapping<TIn, TOut>::reinit();
    }

	virtual void apply(typename self::out_pos_type& out, 
	                   const typename self::in_pos_type& in )  {
		assert( this->Nout == this->Nin );

        const pairs_type& p = pairs.getValue();
        assert( !p.empty() );

        for( unsigned j = 0, m = p.size(); j < m; ++j)
        {
            out[j] = TIn::getCPos( in[p[j][1]] ) - TIn::getCPos( in[p[j][0]] );
        }
	}

	virtual void assemble( const typename self::in_pos_type& in ) {
		// jacobian matrix assembly
        const pairs_type& p = pairs.getValue();
		assert( !p.empty() );

		typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;

		J.resize( Nout * p.size(), Nin * in.size());
		J.setZero();

		for(unsigned k = 0, n = p.size(); k < n; ++k) {
			
            for(unsigned i = 0; i < Nout; ++i) {

                if(p[k][1] == p[k][0]) continue;

                unsigned row = k * Nout + i;
                J.startVec( row );

                // needs to be inserted in the right order in the eigen matrix
                if( p[k][1] < p[k][0] )
                {
                    J.insertBack(row, p[k][1] * Nin + i ) = 1;
                    J.insertBack(row, p[k][0] * Nin + i ) = -1;
                }
                else
                {
                    J.insertBack(row, p[k][0] * Nin + i ) = -1;
                    J.insertBack(row, p[k][1] * Nin + i ) = 1;
                }
			}
		}
        J.finalize();
	}

    void draw(const core::visual::VisualParams* vparams)
    {

#ifndef SOFA_NO_OPENGL
        if( !vparams->displayFlags().getShowMechanicalMappings() ) return;

        glEnable(GL_LIGHTING);

        typename core::behavior::MechanicalState<TIn>::ReadVecCoord pos = this->getFromModel()->readPositions();
        const pairs_type& p = pairs.getValue();

        if( d_showObjectScale.getValue() == 0 )
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
        const pairs_type& p = pairs.getValue();

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




/**
 Multi-maps two vec dofs to their spatial position difference:

 (p1, p2) -> p2.t - p1.t
 (with .t the translation obtained by DataTypes::getCPos)

 This is used to obtain relative dofs
 on which a stiffness/compliance may be applied
*/

    template <class TIn, class TOut >
    class SOFA_Compliant_API DifferenceMultiMapping : public AssembledMultiMapping<TIn, TOut>
    {
        typedef DifferenceMultiMapping self;

    public:
        SOFA_CLASS(SOFA_TEMPLATE2(DifferenceMultiMapping,TIn,TOut), SOFA_TEMPLATE2(core::MultiMapping,TIn,TOut));

        typedef AssembledMultiMapping<TIn, TOut> Inherit;
        typedef TIn In;
        typedef TOut Out;
        typedef typename Out::VecCoord OutVecCoord;
        typedef typename Out::VecDeriv OutVecDeriv;
        typedef typename Out::Coord OutCoord;
        typedef typename Out::Deriv OutDeriv;
        typedef typename Out::MatrixDeriv OutMatrixDeriv;
        typedef typename Out::Real Real;
        typedef typename In::Deriv InDeriv;
        typedef typename In::MatrixDeriv InMatrixDeriv;
        typedef typename In::Coord InCoord;
        typedef typename In::VecCoord InVecCoord;
        typedef typename In::VecDeriv InVecDeriv;
        typedef linearsolver::EigenSparseMatrix<TIn,TOut>    SparseMatrixEigen;

        typedef typename helper::vector <const InVecCoord*> vecConstInVecCoord;
        typedef typename helper::vector<OutVecCoord*> vecOutVecCoord;

        enum {Nin = In::deriv_total_size, Nout = Out::deriv_total_size };

        virtual void init()
        {
            if(!pairs.getValue().size() && this->getFromModels()[0]->getSize()==this->getFromModels()[1]->getSize()) // if no pair is defined-> map all dofs
            {
                helper::WriteOnlyAccessor<Data<pairs_type> > p(pairs);
                p.resize(this->getFromModels()[0]->getSize());
                for( unsigned j = 0; j < p.size(); ++j) p[j]=index_pair(j,j);
            }
            this->getToModels()[0]->resize( pairs.getValue().size() );
            AssembledMultiMapping<TIn, TOut>::init();
        }

        virtual void reinit()
        {
            if(!pairs.getValue().size() && this->getFromModels()[0]->getSize()==this->getFromModels()[1]->getSize()) // if no pair is defined-> map all dofs
            {
                helper::WriteOnlyAccessor<Data<pairs_type> > p(pairs);
                p.resize(this->getFromModels()[0]->getSize());
                for( unsigned j = 0; j < p.size(); ++j) p[j]=index_pair(j,j);
            }
            this->getToModels()[0]->resize( pairs.getValue().size() );
            AssembledMultiMapping<TIn, TOut>::reinit();
        }

        virtual void apply(typename self::out_pos_type& out,
                           const helper::vector<typename self::in_pos_type>& in)  {
            // macro_trace;
            assert( in.size() == 2 );

            assert( this->Nout == this->Nin );

            const pairs_type& p = pairs.getValue();
            assert( !p.empty() );

            for( unsigned j = 0, m = p.size(); j < m; ++j) {
                out[j] = TIn::getCPos( in[1] [p[j][1]] ) - TIn::getCPos( in[0] [p[j][0]] );
            }

        }


        typedef defaulttype::Vec<2, unsigned> index_pair;
        typedef helper::vector< index_pair > pairs_type;

        Data< pairs_type > pairs;

    protected:

        DifferenceMultiMapping()
            : pairs( initData(&pairs, "pairs", "index pairs for computing deltas") ) {

        }

        void assemble(const helper::vector<typename self::in_pos_type>& in ) {

            const pairs_type& p = pairs.getValue();
            assert( !p.empty() );

            for(unsigned i = 0, n = in.size(); i < n; ++i) {

                typename Inherit::jacobian_type::CompressedMatrix& J = this->jacobian(i).compressedMatrix;

                J.resize( Nout * p.size(), Nin * in[i].size());
                J.setZero();

                Real sign = (i == 0) ? -1 : 1;

                for(unsigned k = 0, n = p.size(); k < n; ++k) {
                    write_block(J, k, p[k][i], sign);
                }

                J.finalize();
            }
        }



        // write sign * identity in jacobian(obj)
        void write_block(typename Inherit::jacobian_type::CompressedMatrix& J,
                         unsigned row, unsigned col,
                         SReal sign) {
            assert( Nout == Nin );

            // for each coordinate in matrix block
            for( unsigned i = 0, n = Nout; i < n; ++i ) {
                unsigned r = row * Nout + i;
                unsigned c = col * Nin + i;

                J.startVec(r);
                J.insertBack(r, c) = sign;
            }
        }


        virtual void updateForceMask()
        {
            const pairs_type& p = pairs.getValue();

            for( size_t i = 0, iend = p.size(); i < iend; ++i )
            {
                if( this->maskTo[0]->getEntry(i) )
                {
                    const index_pair& indices = p[i];
                    this->maskFrom[0]->insertEntry(indices[0]);
                    this->maskFrom[1]->insertEntry(indices[1]);
                }
            }
        }

    };


}
}
}


#endif
