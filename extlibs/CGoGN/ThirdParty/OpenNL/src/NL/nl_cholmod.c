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

#include <NL/nl_cholmod.h>
#include <NL/nl_context.h>

/************************************************************************/
/* CHOLMOD wrapper */

#ifdef NL_USE_CHOLMOD

/* CHOLMOD includes */
#include <cholmod.h>



typedef struct {
	cholmod_common c ;
	cholmod_factor* cL ; // factor
	cholmod_dense* cb ;  // right-hand side
} cholmod_context ;



NLboolean nlFactorize_CHOLMOD() {

	NLSparseMatrix* M;
	NLuint          n;
	NLuint          nnz;
	cholmod_sparse* cA;
	NLRowColumn* Ci;
	cholmod_context* context;
	NLuint       i,j,count ;
	int* colptr;
	int* rowind;
	double* a;

	/* OpenNL Context */
	M   = &(nlCurrentContext->M) ;
	n   = nlCurrentContext->n ;
	nnz = nlSparseMatrixNNZ(M) ; /* Number of Non-Zero coeffs */

	context = (cholmod_context*)(nlCurrentContext->direct_solver_context) ;
	if(context == NULL) {
		nlCurrentContext->direct_solver_context = malloc(sizeof(cholmod_context)) ;
		context = (cholmod_context*)(nlCurrentContext->direct_solver_context) ;
	}

	/* CHOLMOD variables */
	cA = NULL ;

	/* Temporary variables */
	Ci = NULL ;
	

	/* Sanity checks */
	nl_assert(M->storage & NL_MATRIX_STORE_COLUMNS) ;
	nl_assert(M->m == M->n) ;

	cholmod_start(&(context->c)) ;

	/*
	 * Step 1: convert matrix M into CHOLMOD compressed column representation
	 * ----------------------------------------------------------------------
	 */

	cA = cholmod_allocate_sparse(n, n, nnz, NL_FALSE, NL_TRUE, -1, CHOLMOD_REAL, &(context->c)) ;
	colptr = (int*)(cA->p) ;
	rowind = (int*)(cA->i) ;
	a = (double*)(cA->x) ;

	count = 0 ;
	for(i = 0; i < n; i++) {
		Ci = &(M->column[i]);
		colptr[i] = count ;
		for(j = 0; j < Ci->size; j++) {
			a[count]      = Ci->coeff[j].value ;
			rowind[count] = Ci->coeff[j].index ;
			count++ ;
		}
	}
	colptr[n] = nnz ;

	// pre-allocate CHOLMOD right-hand side vector
	context->cb = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, &(context->c)) ;

	/*
	 * Step 2: factorize matrix
	 * ------------------------
	 */

	context->cL = cholmod_analyze(cA, &(context->c)) ;
	cholmod_factorize(cA, context->cL, &(context->c)) ;

	/*
	 * Step 3: cleanup
	 * ---------------
	 */

	cholmod_free_sparse(&cA, &(context->c)) ;

	return NL_TRUE ;
}

NLboolean nlSolve_CHOLMOD() {

	cholmod_dense* cx;
	NLuint i ;
	NLdouble* b;
	NLdouble* x;
	NLuint    n;
	/* Temporary variables */
	double* cbx;
	double* cxx;
	cholmod_context* context;

	/* OpenNL Context */
	b = nlCurrentContext->b ;
	x = nlCurrentContext->x ;
	n = nlCurrentContext->n ;

	context = (cholmod_context*)(nlCurrentContext->direct_solver_context) ;
	nl_assert(context != NULL) ;

	/* CHOLMOD variables */
	cx = NULL ;

	/*
	 * Step 1: convert right-hand side into CHOLMOD representation
	 * -----------------------------------------------------------
	 */

	cbx = (double*)(context->cb->x) ;
	for(i = 0; i < n; i++)
		cbx[i] = b[i] ;

	/*
	 * Step 2: solve
	 * -------------
	 */

	cx = cholmod_solve(CHOLMOD_A, context->cL, context->cb, &(context->c)) ;

	/*
	 * Step 3: get the solution
	 * ------------------------
	 */

	cxx = (double*)(cx->x) ;
	for(i = 0; i < n; i++)
		x[i] = cxx[i] ;

	/*
	 * Step 4: cleanup
	 * ---------------
	 */

	cholmod_free_dense(&cx, &(context->c)) ;

	return NL_TRUE ;
}

void nlClear_CHOLMOD() {

	cholmod_context* context = (cholmod_context*)(nlCurrentContext->direct_solver_context) ;
	if(context != NULL) {
		/* free CHOLMOD structures */
		cholmod_free_factor(&(context->cL), &(context->c)) ;
		cholmod_free_dense(&(context->cb), &(context->c)) ;
		cholmod_finish(&(context->c)) ;
	}
}

#else

NLboolean nlFactorize_CHOLMOD() {
	nl_assert_not_reached ;
	return NL_FALSE ;
}

NLboolean nlSolve_CHOLMOD() {
	nl_assert_not_reached ;
	return NL_FALSE ;
}

void nlClear_CHOLMOD() {
	nl_assert_not_reached ;
}

#endif
