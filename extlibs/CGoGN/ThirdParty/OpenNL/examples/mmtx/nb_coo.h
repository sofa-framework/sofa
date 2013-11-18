#ifndef CNC_NB_COO_H
#define CNC_NB_COO_H


#include <stdio.h>
#include <stdlib.h>
#include <NL/nl_private.h>

extern "C"{
#include "mmio.h"
}

template<typename IndexType>
struct matrix_shape
{
    typedef IndexType index_type;
    IndexType num_rows, num_cols, num_nonzeros;
};

// COOrdinate matrix (aka IJV or Triplet format)
template <typename IndexType, typename ValueType>
struct coo_matrix : public matrix_shape<IndexType> 
{
    typedef IndexType index_type;
    typedef ValueType value_type;

    IndexType * I;  //row indices
    IndexType * J;  //column indices
    ValueType * V;  //nonzero values
};

template <class IndexType,class ValueType>
coo_matrix<IndexType,ValueType> read_coo_matrix(const char * mm_filename)
{
    coo_matrix<IndexType,ValueType> coo;

    FILE * fid;
    MM_typecode matcode;
    
    fid = fopen(mm_filename, "r");

    if (fid == NULL){
        printf("Unable to open file %s\n", mm_filename);
        exit(1);
    }

    if (mm_read_banner(fid, &matcode) != 0){
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }

    if (!mm_is_valid(matcode)){
        printf("Invalid Matrix Market file.\n");
        exit(1);
    }

    if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode) ) ){
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        printf("Only sparse real-valued or pattern coordinate matrices are supported\n");
        exit(1);
    }

    int num_rows, num_cols, num_nonzeros;
    if ( mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
            exit(1);

    coo.num_rows     = (IndexType) num_rows;
    coo.num_cols     = (IndexType) num_cols;
    coo.num_nonzeros = (IndexType) num_nonzeros;

    coo.I = NL_NEW_ARRAY(IndexType ,coo.num_nonzeros);
    coo.J = NL_NEW_ARRAY(IndexType ,coo.num_nonzeros);
    coo.V = NL_NEW_ARRAY(ValueType ,coo.num_nonzeros);

    printf("Reading sparse matrix from file (%s):",mm_filename);
    fflush(stdout);

    if (mm_is_pattern(matcode)){
        // pattern matrix defines sparsity pattern, but not values
        for( IndexType i = 0; i < coo.num_nonzeros; i++ ){
           if (fscanf(fid, " %d %d \n", &(coo.I[i]), &(coo.J[i])) >=2 ) {
	        coo.I[i]--;      //adjust from 1-based to 0-based indexing
                coo.J[i]--;
                coo.V[i] = 1.0;  //use value 1.0 for all nonzero entries 
           } 
	}
    } else if (mm_is_real(matcode) || mm_is_integer(matcode)){
        for( IndexType i = 0; i < coo.num_nonzeros; i++ ){
            IndexType I,J;
            double V;  // always read in a double and convert later if necessary
            
           if (fscanf(fid, " %d %d %lf \n", &I, &J, &V) >= 3) {	
	        coo.I[i] = (IndexType) I - 1; 
            	coo.J[i] = (IndexType) J - 1;
            	coo.V[i] = (ValueType)  V;
           } 
        }
    } else {
        printf("Unrecognized data type\n");
        exit(1);
    }

    fclose(fid);
    printf(" done\n");

    if( mm_is_symmetric(matcode) ){ //duplicate off diagonal entries
        IndexType off_diagonals = 0;
        for( IndexType i = 0; i < coo.num_nonzeros; i++ ){
            if( coo.I[i] != coo.J[i] )
                off_diagonals++;
        }

        IndexType true_nonzeros = 2*off_diagonals + (coo.num_nonzeros - off_diagonals);

        IndexType* new_I = NL_NEW_ARRAY(IndexType,true_nonzeros);
        IndexType* new_J = NL_NEW_ARRAY(IndexType,true_nonzeros);
        ValueType * new_V = NL_NEW_ARRAY(ValueType,true_nonzeros);

        IndexType ptr = 0;
        for( IndexType i = 0; i < coo.num_nonzeros; i++ ){
            if( coo.I[i] != coo.J[i] ){
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
                new_J[ptr] = coo.I[i];  new_I[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
            } else {
                new_I[ptr] = coo.I[i];  new_J[ptr] = coo.J[i];  new_V[ptr] = coo.V[i];
                ptr++;
            }
        }       
         NL_DELETE_ARRAY(coo.I); NL_DELETE_ARRAY(coo.J); NL_DELETE_ARRAY(coo.V);
         coo.I = new_I;  coo.J = new_J; coo.V = new_V;      
         coo.num_nonzeros = true_nonzeros;
    } //end symmetric case

    return coo;
}


#endif
