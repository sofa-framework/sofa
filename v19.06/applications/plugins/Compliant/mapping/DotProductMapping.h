#ifndef DotProductMAPPING_H
#define DotProductMAPPING_H

#include <Compliant/config.h>

#include "ConstantAssembledMapping.h"
#include "AssembledMultiMapping.h"

namespace sofa
{
	
namespace component
{

namespace mapping
{


/**
 Maps two dofs to their DotProduct:

 (p1, p2) -> p1.p2

 @author Matthieu Nesme
 @date 2016

*/
template <class TIn, class TOut >
class SOFA_Compliant_API DotProductMapping : public AssembledMapping<TIn, TOut>
{
  public:
    SOFA_CLASS(SOFA_TEMPLATE2(DotProductMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMapping,TIn,TOut));
	
    typedef DotProductMapping<TIn,TOut> self;
	
	typedef defaulttype::Vec<2, unsigned> index_pair;
    typedef helper::vector< index_pair > pairs_type;

    Data< pairs_type > pairs; ///< index pairs for computing deltas

	
    DotProductMapping()
        : pairs( initData(&pairs, "pairs", "index pairs for computing deltas") )
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

        const pairs_type& p = pairs.getValue();
        assert( out.size() == p.size() );

        for( size_t j = 0, m = p.size(); j < m; ++j)
        {
            out[j][0] = in[p[j][0]] * in[p[j][1]];
        }
	}

	virtual void assemble( const typename self::in_pos_type& in ) {

		// jacobian matrix assembly
        const pairs_type& p = pairs.getValue();
		assert( !p.empty() );

        typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
        this->jacobian.resizeBlocks( p.size(), in.size() );
        J.reserve(p.size()*2*Nin);

        for(size_t k = 0, n = p.size(); k < n; ++k)
        {
            J.startVec( k );

            // needs to be inserted in the right order in the eigen matrix
            if( p[k][1] < p[k][0] )
            {
                for(size_t i = 0; i < Nin; ++i)
                    J.insertBack(k, p[k][1] * Nin + i ) = in[p[k][0]][i];
                 for(size_t i = 0; i < Nin; ++i)
                    J.insertBack(k, p[k][0] * Nin + i ) = in[p[k][1]][i];
            }
            else
            {
                for(size_t i = 0; i < Nin; ++i)
                    J.insertBack(k, p[k][0] * Nin + i ) = in[p[k][1]][i];
                for(size_t i = 0; i < Nin; ++i)
                    J.insertBack(k, p[k][1] * Nin + i ) = in[p[k][0]][i];
            }
		}
        J.finalize();
	}


    virtual void assemble_geometric( const typename self::in_pos_type& in, const typename self::out_force_type& out )
    {
        const pairs_type& p = pairs.getValue();

        typename self::geometric_type& K = this->geometric;
        K.resizeBlocks( in.size(), in.size() );
        K.compressedMatrix.reserve(p.size()*2*Nin);

        for(size_t i = 0, n = p.size(); i < n; ++i)
        {
            const typename TOut::Real& childForce = out[i][0];


            // for vec3
            //      | 0 0 0 1 0 0 |
            //      | 0 0 0 0 1 0 |
            // dJ = | 0 0 0 0 0 1 |
            //      | 1 0 0 0 0 0 |
            //      | 0 1 0 0 0 0 |
            //      | 0 0 1 0 0 0 |

            for(unsigned j=0; j<Nin; j++)
            {
                size_t u = p[i][0]*Nin+j;
                size_t v = p[i][1]*Nin+j;

                K.add(u,v,childForce);
                K.add(v,u,childForce);
            }
        }
        K.compress();
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
 Multi-maps two vec dofs to their Dot Product:

 (p1, p2) -> p1 . p2

 @author Matthieu Nesme
 @date 2016

*/

    template <class TIn, class TOut >
    class SOFA_Compliant_API DotProductMultiMapping : public AssembledMultiMapping<TIn, TOut>
    {
        typedef DotProductMultiMapping self;

    public:
        SOFA_CLASS(SOFA_TEMPLATE2(DotProductMultiMapping,TIn,TOut), SOFA_TEMPLATE2(AssembledMultiMapping,TIn,TOut));

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
            reinit();
            AssembledMultiMapping<TIn, TOut>::init();            
        }

        virtual void reinit()
        {
            if(!pairs.getValue().size() && this->getFromModels()[0]->getSize()==this->getFromModels()[1]->getSize()) // if no pair is defined-> map all dofs
            {
                helper::WriteOnlyAccessor<Data<pairs_type> > p(pairs);
                p.resize(this->getFromModels()[0]->getSize());
                for( unsigned j = 0; j < p.size(); ++j) p[j]=pair(index_pair(0,j),index_pair(1,j));
            }
            this->getToModels()[0]->resize( pairs.getValue().size() );
            AssembledMultiMapping<TIn, TOut>::reinit();
        }

        virtual void apply(typename self::out_pos_type& out,
                           const helper::vector<typename self::in_pos_type>& in)  {

            const pairs_type& p = pairs.getValue();
            assert( out.size() == p.size() );

            for( unsigned j = 0, m = p.size(); j < m; ++j) {

                const InCoord& u = in[p[j][0][0]][p[j][0][1]];
                const InCoord& v = in[p[j][1][0]][p[j][1][1]];

                out[j] = u * v;
            }

        }


        typedef defaulttype::Vec<2, unsigned> index_pair;
        typedef defaulttype::Vec<2, index_pair> pair;
        typedef helper::vector< pair > pairs_type;
        Data< pairs_type > pairs; ///< index pairs for computing deltas

    protected:

        DotProductMultiMapping()
            : pairs( initData(&pairs, "pairs", "index pairs for computing deltas") )
        {
        }

        void assemble(const helper::vector<typename self::in_pos_type>& in )
        {
            const pairs_type& p = pairs.getValue();
            assert( !p.empty() );

            for(unsigned i = 0, n = in.size(); i < n; ++i)
            {
                this->jacobian(i).resizeBlocks( p.size(), in[i].size() );
                this->jacobian(i).compressedMatrix.reserve( p.size()*Nin );
            }


            for( unsigned j = 0, m = p.size(); j < m; ++j)
            {
                const unsigned& in_u = p[j][0][0];
                const unsigned& in_v = p[j][1][0];
                const unsigned& index_u = p[j][0][1];
                const unsigned& index_v = p[j][1][1];

                const InCoord& u = in[in_u][index_u];
                const InCoord& v = in[in_v][index_v];

                for(size_t i = 0; i < Nin; ++i)
                {
                    this->jacobian(in_u).add( j, index_u*Nin+i, v[i] );
                    this->jacobian(in_v).add( j, index_v*Nin+i, u[i] );
                }
            }

            for(unsigned i = 0, n = in.size(); i < n; ++i)
                this->jacobian(i).compress();
        }

        virtual void assemble_geometric( const helper::vector<typename self::const_in_coord_type>& in, const typename self::const_out_deriv_type& out)
        {
            typename self::geometric_type& K = this->geometric;

            size_t size = in[0].size();
            helper::vector<size_t> cumulatedSize(in.size());
            cumulatedSize[0] = 0;
            for( size_t i=1 ; i<in.size() ; ++i )
            {
                size += in[i].size();
                cumulatedSize[i] = cumulatedSize[i-1]+in[i-1].size();
            }

            const pairs_type& p = pairs.getValue();

            K.resizeBlocks( size, size );
            K.compressedMatrix.reserve(p.size()*2*Nin);

            for(size_t i = 0, n = p.size(); i < n; ++i)
            {
                const typename TOut::Real& childForce = out[i][0];

                const unsigned& in_u = p[i][0][0];
                const unsigned& in_v = p[i][1][0];
                const unsigned& index_u = p[i][0][1];
                const unsigned& index_v = p[i][1][1];


                // for vec3
                //      | 0 0 0 1 0 0 |
                //      | 0 0 0 0 1 0 |
                // dJ = | 0 0 0 0 0 1 |
                //      | 1 0 0 0 0 0 |
                //      | 0 1 0 0 0 0 |
                //      | 0 0 1 0 0 0 |

                for(unsigned j=0; j<Nin; j++)
                {
                    size_t u = (cumulatedSize[in_u] + index_u)*Nin+j;
                    size_t v = (cumulatedSize[in_v] + index_v)*Nin+j;

                    K.add(u,v,childForce);
                    K.add(v,u,childForce);
                }
            }
            K.compress();
        }



        virtual void updateForceMask()
        {
            const pairs_type& p = pairs.getValue();

            for( size_t i = 0, iend = p.size(); i < iend; ++i )
            {
                if( this->maskTo[0]->getEntry(i) )
                {
                    const pair& pp = p[i];
                    this->maskFrom[pp[0][0]]->insertEntry(pp[0][1]);
                    this->maskFrom[pp[1][0]]->insertEntry(pp[1][1]);
                }
            }
        }

    };





//////////////////////


    /**
     Maps a dof to its dot product with a target:

     p -> p . t

     @author Matthieu Nesme
     @date 2016

    */
    template <class TIn, class TOut >
    class SOFA_Compliant_API DotProductFromTargetMapping : public ConstantAssembledMapping<TIn, TOut>
    {
      public:
        SOFA_CLASS(SOFA_TEMPLATE2(DotProductFromTargetMapping,TIn,TOut), SOFA_TEMPLATE2(ConstantAssembledMapping,TIn,TOut));

        typedef DotProductFromTargetMapping<TIn,TOut> self;

        typedef typename Inherit1::InVecCoord InVecCoord;

        Data< helper::vector< unsigned > > d_indices; ///< indices of the dofs used to compute a dot product
        Data< InVecCoord > d_targets; ///< targets to compute the dot products with


        DotProductFromTargetMapping()
            : d_indices( initData(&d_indices, "indices", "indices of the dofs used to compute a dot product") )
            , d_targets( initData(&d_targets, "targets", "targets to compute the dot products with") )
        {}

        enum {Nin = TIn::deriv_total_size, Nout = TOut::deriv_total_size };

        virtual void init()
        {
            reinit();
            AssembledMapping<TIn, TOut>::init();
        }

        virtual void reinit()
        {
            size_t size = d_indices.getValue().size();
            this->getToModel()->resize( size );

            // if targets is not big enough, duplicate last component
            InVecCoord& targets = *d_targets.beginEdit();
            assert( !targets.empty() );
            targets.reserve(size);
            for( size_t i=targets.size() ; i<size ; ++i )
                targets.push_back(targets.back());
            d_targets.endEdit();

            AssembledMapping<TIn, TOut>::reinit();
        }

        virtual void apply(typename self::out_pos_type& out,
                           const typename self::in_pos_type& in )  {

            const helper::vector< unsigned >& indices = d_indices.getValue();
            const InVecCoord& targets = d_targets.getValue();
            assert( out.size() == indices.size() );

            for( size_t j = 0, m = indices.size(); j < m; ++j)
            {
                out[j][0] = in[indices[j]] * targets[j];
            }
        }

        virtual void assemble( const typename self::in_pos_type& in ) {

            // jacobian matrix assembly
            const helper::vector< unsigned >& indices = d_indices.getValue();
            const InVecCoord& targets = d_targets.getValue();

            typename self::jacobian_type::CompressedMatrix& J = this->jacobian.compressedMatrix;
            this->jacobian.resizeBlocks( indices.size(), in.size() );
            J.reserve(indices.size()*Nin);

            for(size_t k = 0, n = indices.size(); k < n; ++k)
            {
                J.startVec( k );

                for(size_t i = 0; i < Nin; ++i)
                    J.insertBack(k, indices[k] * Nin + i ) = targets[k][i];
            }
            J.finalize();
        }


        virtual void updateForceMask()
        {
            const helper::vector< unsigned >& indices = d_indices.getValue();

            for( size_t i = 0, iend = indices.size(); i < iend; ++i )
            {
                if( this->maskTo->getEntry(i) )
                {
                    this->maskFrom->insertEntry(indices[i]);
                }
            }
        }

    };




}
}
}


#endif
