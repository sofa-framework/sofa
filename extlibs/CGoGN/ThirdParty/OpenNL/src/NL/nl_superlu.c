/*
 *  Copyright (c) 2004-2010, Bruno Levy
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Bruno Levy
 *
 *     levy@loria.fr
 *
 *     ALICE Project
 *     LORIA, INRIA Lorraine, 
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX 
 *     FRANCE
 *
 */

#include <NL/nl_superlu.h>
#include <NL/nl_context.h>

/************************************************************************/
/* SuperLU wrapper */

#ifdef NL_USE_SUPERLU

/* SuperLU includes */
#include <slu_cdefs.h>
#include <supermatrix.h>


typedef struct {
	superlu_options_t options ;
	SuperLUStat_t stat ;
	SuperMatrix L ;
	SuperMatrix U ;
	NLint* perm_c ;
	NLint* perm_r;
} superlu_context ;



NLboolean nlFactorize_SUPERLU() {

	/* OpenNL Context */
	NLSparseMatrix* M   = &(nlCurrentContext->M) ;
	NLuint          n   = nlCurrentContext->n ;
	NLuint          nnz = nlSparseMatrixNNZ(M) ; /* Number of Non-Zero coeffs */

	superlu_context* context = (superlu_context*)(nlCurrentContext->direct_solver_context) ;
	if(context == NULL) {
		nlCurrentContext->direct_solver_context = malloc(sizeof(superlu_context)) ;
		context = (superlu_context*)(nlCurrentContext->direct_solver_context) ;
	}

	/* SUPERLU variables */
	NLint info ;
	SuperMatrix A, AC ;

	/* Temporary variables */
	NLRowColumn* Ci = NULL ;
	NLuint       i,j,count ;

	/* Sanity checks */
	nl_assert(!(M->storage & NL_MATRIX_STORE_SYMMETRIC)) ;
	nl_assert(M->storage & NL_MATRIX_STORE_ROWS) ;
	nl_assert(M->m == M->n) ;

	set_default_options(&(context->options)) ;

	switch(nlCurrentContext->solver) {
	case NL_SUPERLU_EXT: {
		context->options.ColPerm = NATURAL ;
	} break ;
	case NL_PERM_SUPERLU_EXT: {
		context->options.ColPerm = COLAMD ;
	} break ;
	case NL_SYMMETRIC_SUPERLU_EXT: {
		context->options.ColPerm = MMD_AT_PLUS_A ;
		context->options.SymmetricMode = YES ;
	} break ;
	default: {
		nl_assert_not_reached ;
	} break ;
	}

	StatInit(&(context->stat)) ;

	/*
	 * Step 1: convert matrix M into SUPERLU compressed column representation
	 * ----------------------------------------------------------------------
	 */

	NLint*    xa   = NL_NEW_ARRAY(NLint, n+1) ;
	NLdouble* a    = NL_NEW_ARRAY(NLdouble, nnz) ;
	NLint*    asub = NL_NEW_ARRAY(NLint, nnz) ;

	count = 0 ;
	for(i = 0; i < n; i++) {
		Ci = &(M->row[i]) ;
		xa[i] = count ;
		for(j = 0; j < Ci->size; j++) {
			a[count]    = Ci->coeff[j].value ;
			asub[count] = Ci->coeff[j].index ;
			count++ ;
		}
	}
	xa[n] = nnz ;

	dCreate_CompCol_Matrix(
		&A, n, n, nnz, a, asub, xa,
		SLU_NR, /* Row wise     */
		SLU_D,  /* doubles         */
		SLU_GE  /* general storage */
	);

	/*
	 * Step 2: factorize matrix
	 * ------------------------
	 */

	context->perm_c = NL_NEW_ARRAY(NLint, n) ;
	context->perm_r = NL_NEW_ARRAY(NLint, n) ;
	NLint* etree    = NL_NEW_ARRAY(NLint, n) ;

	get_perm_c(context->options.ColPerm, &A, context->perm_c) ;
	sp_preorder(&(context->options), &A, context->perm_c, etree, &AC) ;

	int panel_size = sp_ienv(1) ;
	int relax = sp_ienv(2) ;

	dgstrf(&(context->options),
		   &AC,
		   relax,
		   panel_size,
		   etree,
		   NULL,
		   0,
		   context->perm_c,
		   context->perm_r,
		   &(context->L),
		   &(context->U),
		   &(context->stat),
		   &info) ;

	/*
	 * Step 3: cleanup
	 * ---------------
	 */

	NL_DELETE_ARRAY(xa) ;
	NL_DELETE_ARRAY(a) ;
	NL_DELETE_ARRAY(asub) ;
	NL_DELETE_ARRAY(etree) ;
	Destroy_SuperMatrix_Store(&A);
	Destroy_CompCol_Permuted(&AC);
	StatFree(&(context->stat));

	return NL_TRUE ;
}

NLboolean nlSolve_SUPERLU() {

	/* OpenNL Context */
	NLdouble* b = nlCurrentContext->b ;
	NLdouble* x = nlCurrentContext->x ;
	NLuint    n = nlCurrentContext->n ;

	superlu_context* context = (superlu_context*)(nlCurrentContext->direct_solver_context) ;
	nl_assert(context != NULL) ;

	/* SUPERLU variables */
	SuperMatrix B ;
	DNformat *vals = NULL ; /* access to result */
	double *rvals  = NULL ; /* access to result */

	/* Temporary variables */
	NLuint i ;
	NLint info ;

	StatInit(&(context->stat)) ;

	/*
	 * Step 1: convert right-hand side into SUPERLU representation
	 * -----------------------------------------------------------
	 */

	dCreate_Dense_Matrix(
		&B, n, 1, b, n,
		SLU_DN, /* Fortran-type column-wise storage */
		SLU_D,  /* doubles                          */
		SLU_GE  /* general storage                  */
	);

	/*
	 * Step 2: solve
	 * -------------
	 */

	dgstrs(NOTRANS,
		   &(context->L),
		   &(context->U),
		   context->perm_c,
		   context->perm_r,
		   &B,
		   &(context->stat),
		   &info) ;

	/*
	 * Step 3: get the solution
	 * ------------------------
	 */

	vals = (DNformat*)B.Store;
	rvals = (double*)(vals->nzval);
	for(i = 0; i <  n; i++)
		x[i] = rvals[i];

	/*
	 * Step 4: cleanup
	 * ---------------
	 */

	Destroy_SuperMatrix_Store(&B);
	StatFree(&(context->stat));

	return NL_TRUE ;
}

void nlClear_SUPERLU() {

	superlu_context* context = (superlu_context*)(nlCurrentContext->direct_solver_context) ;
	if(context != NULL) {
		Destroy_SuperNode_Matrix(&(context->L)) ;
		Destroy_CompCol_Matrix(&(context->U)) ;
		NL_DELETE_ARRAY(context->perm_c) ;
		NL_DELETE_ARRAY(context->perm_r) ;
	}
}

#else

NLboolean nlFactorize_SUPERLU() {
	nl_assert_not_reached ;
	return NL_FALSE;
}

NLboolean nlSolve_SUPERLU() {
    nl_assert_not_reached ;
    return NL_FALSE;
}

void nlClear_SUPERLU() {
	nl_assert_not_reached ;
}

#endif

