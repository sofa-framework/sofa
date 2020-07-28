# include <stdlib.h>
# include <limits.h>
# include <math.h>
# include <stdio.h>

# include "csparse.h"

cs *cs_add ( const cs *A, const cs *B, double alpha, double beta )
/*
  Purpose:

    CS_ADD computes C = alpha*A + beta*B for sparse A and B.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int p, j, nz = 0, anz, *Cp, *Ci, *Bp, m, n, bnz, *w, values ;
    double *x, *Bx, *Cx ;
    cs *C ;
    if (!A || !B) return (NULL) ;	/* check inputs */
    m = A->m ; anz = A->p [A->n] ;
    n = B->n ; Bp = B->p ; Bx = B->x ; bnz = Bp [n] ;
    w = cs_calloc (m, sizeof (int)) ;
    values = (A->x != NULL) && (Bx != NULL) ;
    x = values ? cs_malloc (m, sizeof (double)) : NULL ;
    C = cs_spalloc (m, n, anz + bnz, values, 0) ;
    if (!C || !w || (values && !x)) return (cs_done (C, w, x, 0)) ;
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (j = 0 ; j < n ; j++)
    {
	Cp [j] = nz ;			/* column j of C starts here */
	nz = cs_scatter (A, j, alpha, w, x, j+1, C, nz) ;   /* alpha*A(:,j)*/
	nz = cs_scatter (B, j, beta, w, x, j+1, C, nz) ;    /* beta*B(:,j) */
	if (values) for (p = Cp [j] ; p < nz ; p++) Cx [p] = x [Ci [p]] ;
    }
    Cp [n] = nz ;			/* finalize the last column of C */
    cs_sprealloc (C, 0) ;		/* remove extra space from C */
    return (cs_done (C, w, x, 1)) ;	/* success; free workspace, return C */
}
static int cs_wclear (int mark, int lemax, int *w, int n)
/*
  Purpose:

    CS_WCLEAR clears W.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int k ;
    if (mark < 2 || (mark + lemax < 0))
    {
	for (k = 0 ; k < n ; k++) if (w [k] != 0) w [k] = 1 ;
	mark = 2 ;
    }
    return (mark) ;	/* at this point, w [0..n-1] < mark holds */
}

/* keep off-diagonal entries; drop diagonal entries */
static int cs_diag (int i, int j, double aij, void * other) 
{
  return (i != j); 
  (void)aij; (void)other; // unused parameters
}

/* p = amd(A+A') if symmetric is true, or amd(A'A) otherwise */
int *cs_amd ( const cs *A, int order )  
/*
  Purpose:

    CS_AMD carries out the approximate minimum degree algorithm.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.

  Parameters:

    Input, int ORDER:
    -1:natural, 
    0:Cholesky,  
    1:LU, 
    2:QR
*/
{
    cs *C, *A2, *AT ;
    int *Cp, *Ci, *last, *ww, *len, *nv, *next, *P, *head, *elen, *degree, *w,
	*hhead, *ATp, *ATi, d, dk, dext, lemax = 0, e, elenk, eln, i, j, k, k1,
	k2, k3, jlast, ln, dense, nzmax, mindeg = 0, nvi, nvj, nvk, mark, wnvi,
	ok, cnz, nel = 0, p, p1, p2, p3, p4, pj, pk, pk1, pk2, pn, q, n, m ;
    unsigned int h ;
    /* --- Construct matrix C ----------------------------------------------- */
    if (!A || order < 0) return (NULL) ;    /* check inputs; quick return */
    AT = cs_transpose (A, 0) ;		    /* compute A' */
    if (!AT) return (NULL) ;
    m = A->m ; n = A->n ;
    dense = (int)CS_MAX (16, 10 * sqrt ((double) n)) ;   /* find dense threshold */
    dense = CS_MIN (n-2, dense) ;
    if (order == 0 && n == m)
    {
	C = cs_add (A, AT, 0, 0) ;	    /* C = A+A' */
    }
    else if (order == 1)
    {
	ATp = AT->p ;			    /* drop dense columns from AT */
	ATi = AT->i ;
	for (p2 = 0, j = 0 ; j < m ; j++)
	{
	    p = ATp [j] ;		    /* column j of AT starts here */
	    ATp [j] = p2 ;		    /* new column j starts here */
	    if (ATp [j+1] - p > dense) continue ;   /* skip dense col j */
	    for ( ; p < ATp [j+1] ; p++) ATi [p2++] = ATi [p] ;
	}
	ATp [m] = p2 ;			    /* finalize AT */
	A2 = cs_transpose (AT, 0) ;	    /* A2 = AT' */
	C = A2 ? cs_multiply (AT, A2) : NULL ;	/* C=A'*A with no dense rows */
	cs_spfree (A2) ;
    }
    else
    {
	C = cs_multiply (AT, A) ;	    /* C=A'*A */
    }
    cs_spfree (AT) ;
    if (!C) return (NULL) ;
    P = cs_malloc (n+1, sizeof (int)) ;	    /* allocate result */
    ww = cs_malloc (8*(n+1), sizeof (int)) ;/* get workspace */
    len  = ww           ; nv     = ww +   (n+1) ; next   = ww + 2*(n+1) ;
    head = ww + 3*(n+1) ; elen   = ww + 4*(n+1) ; degree = ww + 5*(n+1) ;
    w    = ww + 6*(n+1) ; hhead  = ww + 7*(n+1) ;
    last = P ;				    /* use P as workspace for last */
    cs_fkeep (C, &cs_diag, NULL) ;	    /* drop diagonal entries */
    Cp = C->p ;
    cnz = Cp [n] ;
    if (!cs_sprealloc (C, cnz+cnz/5+2*n)) return (cs_idone (P, C, ww, 0)) ;
    /* --- Initialize quotient graph ---------------------------------------- */
    for (k = 0 ; k < n ; k++) len [k] = Cp [k+1] - Cp [k] ;
    len [n] = 0 ;
    nzmax = C->nzmax ;
    Ci = C->i ;
    for (i = 0 ; i <= n ; i++)
    {
	head [i] = -1 ;			    /* degree list i is empty */
	last [i] = -1 ;
	next [i] = -1 ;
	hhead [i] = -1 ;		    /* hash list i is empty */
	nv [i] = 1 ;			    /* node i is just one node */
	w [i] = 1 ;			    /* node i is alive */
	elen [i] = 0 ;			    /* Ek of node i is empty */
	degree [i] = len [i] ;		    /* degree of node i */
    }
    mark = cs_wclear (0, 0, w, n) ;	    /* clear w */
    elen [n] = -2 ;			    /* n is a dead element */
    Cp [n] = -1 ;			    /* n is a root of assembly tree */
    w [n] = 0 ;				    /* n is a dead element */
    /* --- Initialize degree lists ------------------------------------------ */
    for (i = 0 ; i < n ; i++)
    {
	d = degree [i] ;
	if (d == 0)			    /* node i is empty */
	{
	    elen [i] = -2 ;		    /* element i is dead */
	    nel++ ;
	    Cp [i] = -1 ;		    /* i is a root of assemby tree */
	    w [i] = 0 ;
	}
	else if (d > dense)		    /* node i is dense */
	{
	    nv [i] = 0 ;		    /* absorb i into element n */
	    elen [i] = -1 ;		    /* node i is dead */
	    nel++ ;
	    Cp [i] = CS_FLIP (n) ;
	    nv [n]++ ;
	}
	else
	{
	    if (head [d] != -1) last [head [d]] = i ;
	    next [i] = head [d] ;	    /* put node i in degree list d */
	    head [d] = i ;
	}
    }
    while (nel < n)			    /* while (selecting pivots) do */
    {
	/* --- Select node of minimum approximate degree -------------------- */
	for (k = -1 ; mindeg < n && (k = head [mindeg]) == -1 ; mindeg++) ;
	if (next [k] != -1) last [next [k]] = -1 ;
	head [mindeg] = next [k] ;	    /* remove k from degree list */
	elenk = elen [k] ;		    /* elenk = |Ek| */
	nvk = nv [k] ;			    /* # of nodes k represents */
	nel += nvk ;			    /* nv[k] nodes of A eliminated */
	/* --- Garbage collection ------------------------------------------- */
	if (elenk > 0 && cnz + mindeg >= nzmax)
	{
	    for (j = 0 ; j < n ; j++)
	    {
		if ((p = Cp [j]) >= 0)	    /* j is a live node or element */
		{
		    Cp [j] = Ci [p] ;	    /* save first entry of object */
		    Ci [p] = CS_FLIP (j) ;  /* first entry is now CS_FLIP(j) */
		}
	    }
	    for (q = 0, p = 0 ; p < cnz ; ) /* scan all of memory */
	    {
		if ((j = CS_FLIP (Ci [p++])) >= 0)  /* found object j */
		{
		    Ci [q] = Cp [j] ;	    /* restore first entry of object */
		    Cp [j] = q++ ;	    /* new pointer to object j */
		    for (k3 = 0 ; k3 < len [j]-1 ; k3++) Ci [q++] = Ci [p++] ;
		}
	    }
	    cnz = q ;			    /* Ci [cnz...nzmax-1] now free */
	}
	/* --- Construct new element ---------------------------------------- */
	dk = 0 ;
	nv [k] = -nvk ;			    /* flag k as in Lk */
	p = Cp [k] ;
	pk1 = (elenk == 0) ? p : cnz ;	    /* do in place if elen[k] == 0 */
	pk2 = pk1 ;
	for (k1 = 1 ; k1 <= elenk + 1 ; k1++)
	{
	    if (k1 > elenk)
	    {
		e = k ;			    /* search the nodes in k */
		pj = p ;		    /* list of nodes starts at Ci[pj]*/
		ln = len [k] - elenk ;	    /* length of list of nodes in k */
	    }
	    else
	    {
		e = Ci [p++] ;		    /* search the nodes in e */
		pj = Cp [e] ;
		ln = len [e] ;		    /* length of list of nodes in e */
	    }
	    for (k2 = 1 ; k2 <= ln ; k2++)
	    {
		i = Ci [pj++] ;
		if ((nvi = nv [i]) <= 0) continue ; /* node i dead, or seen */
		dk += nvi ;		    /* degree[Lk] += size of node i */
		nv [i] = -nvi ;		    /* negate nv[i] to denote i in Lk*/
		Ci [pk2++] = i ;	    /* place i in Lk */
		if (next [i] != -1) last [next [i]] = last [i] ;
		if (last [i] != -1)	    /* remove i from degree list */
		{
		    next [last [i]] = next [i] ;
		}
		else
		{
		    head [degree [i]] = next [i] ;
		}
	    }
	    if (e != k)
	    {
		Cp [e] = CS_FLIP (k) ;	    /* absorb e into k */
		w [e] = 0 ;		    /* e is now a dead element */
	    }
	}
	if (elenk != 0) cnz = pk2 ;	    /* Ci [cnz...nzmax] is free */
	degree [k] = dk ;		    /* external degree of k - |Lk\i| */
	Cp [k] = pk1 ;			    /* element k is in Ci[pk1..pk2-1] */
	len [k] = pk2 - pk1 ;
	elen [k] = -2 ;			    /* k is now an element */
	/* --- Find set differences ----------------------------------------- */
	mark = cs_wclear (mark, lemax, w, n) ;	/* clear w if necessary */
	for (pk = pk1 ; pk < pk2 ; pk++)    /* scan 1: find |Le\Lk| */
	{
	    i = Ci [pk] ;
	    if ((eln = elen [i]) <= 0) continue ;/* skip if elen[i] empty */
	    nvi = -nv [i] ;			 /* nv [i] was negated */
	    wnvi = mark - nvi ;
	    for (p = Cp [i] ; p <= Cp [i] + eln - 1 ; p++)  /* scan Ei */
	    {
		e = Ci [p] ;
		if (w [e] >= mark)
		{
		    w [e] -= nvi ;	    /* decrement |Le\Lk| */
		}
		else if (w [e] != 0)	    /* ensure e is a live element */
		{
		    w [e] = degree [e] + wnvi ;	/* 1st time e seen in scan 1 */
		}
	    }
	}
	/* --- Degree update ------------------------------------------------ */
	for (pk = pk1 ; pk < pk2 ; pk++)    /* scan2: degree update */
	{
	    i = Ci [pk] ;		    /* consider node i in Lk */
	    p1 = Cp [i] ;
	    p2 = p1 + elen [i] - 1 ;
	    pn = p1 ;
	    for (h = 0, d = 0, p = p1 ; p <= p2 ; p++)    /* scan Ei */
	    {
		e = Ci [p] ;
		if (w [e] != 0)		    /* e is an unabsorbed element */
		{
		    dext = w [e] - mark ;   /* dext = |Le\Lk| */
		    if (dext > 0)
		    {
			d += dext ;	    /* sum up the set differences */
			Ci [pn++] = e ;	    /* keep e in Ei */
			h += e ;	    /* compute the hash of node i */
		    }
		    else
		    {
			Cp [e] = CS_FLIP (k) ;	/* aggressive absorb. e->k */
			w [e] = 0 ;		/* e is a dead element */
		    }
		}
	    }
	    elen [i] = pn - p1 + 1 ;	    /* elen[i] = |Ei| */
	    p3 = pn ;
	    p4 = p1 + len [i] ;
	    for (p = p2 + 1 ; p < p4 ; p++) /* prune edges in Ai */
	    {
		j = Ci [p] ;
		if ((nvj = nv [j]) <= 0) continue ; /* node j dead or in Lk */
		d += nvj ;		    /* degree(i) += |j| */
		Ci [pn++] = j ;		    /* place j in node list of i */
		h += j ;		    /* compute hash for node i */
	    }
	    if (d == 0)			    /* check for mass elimination */
	    {
		Cp [i] = CS_FLIP (k) ;	    /* absorb i into k */
		nvi = -nv [i] ;
		dk -= nvi ;		    /* |Lk| -= |i| */
		nvk += nvi ;		    /* |k| += nv[i] */
		nel += nvi ;
		nv [i] = 0 ;
		elen [i] = -1 ;		    /* node i is dead */
	    }
	    else
	    {
		degree [i] = CS_MIN (degree [i], d) ;	/* update degree(i) */
		Ci [pn] = Ci [p3] ;	    /* move first node to end */
		Ci [p3] = Ci [p1] ;	    /* move 1st el. to end of Ei */
		Ci [p1] = k ;		    /* add k as 1st element in of Ei */
		len [i] = pn - p1 + 1 ;	    /* new len of adj. list of node i */
		h %= n ;		    /* finalize hash of i */
		next [i] = hhead [h] ;	    /* place i in hash bucket */
		hhead [h] = i ;
		last [i] = h ;		    /* save hash of i in last[i] */
	    }
	}				    /* scan2 is done */
	degree [k] = dk ;		    /* finalize |Lk| */
	lemax = CS_MAX (lemax, dk) ;
	mark = cs_wclear (mark+lemax, lemax, w, n) ;	/* clear w */
	/* --- Supernode detection ------------------------------------------ */
	for (pk = pk1 ; pk < pk2 ; pk++)
	{
	    i = Ci [pk] ;
	    if (nv [i] >= 0) continue ;		/* skip if i is dead */
	    h = last [i] ;			/* scan hash bucket of node i */
	    i = hhead [h] ;
	    hhead [h] = -1 ;			/* hash bucket will be empty */
	    for ( ; i != -1 && next [i] != -1 ; i = next [i], mark++)
	    {
		ln = len [i] ;
		eln = elen [i] ;
		for (p = Cp[i]+1 ; p <= Cp[i]+ln-1 ; p++) w [Ci [p]] = mark ;
		jlast = i ;
		for (j = next [i] ; j != -1 ; )	/* compare i with all j */
		{
		    ok = (len [j] == ln) && (elen [j] == eln) ;
		    for (p = Cp [j] + 1 ; ok && p <= Cp [j] + ln - 1 ; p++)
		    {
			if (w [Ci [p]] != mark) ok = 0 ;    /* compare i and j*/
		    }
		    if (ok)			/* i and j are identical */
		    {
			Cp [j] = CS_FLIP (i) ;	/* absorb j into i */
			nv [i] += nv [j] ;
			nv [j] = 0 ;
			elen [j] = -1 ;		/* node j is dead */
			j = next [j] ;		/* delete j from hash bucket */
			next [jlast] = j ;
		    }
		    else
		    {
			jlast = j ;		/* j and i are different */
			j = next [j] ;
		    }
		}
	    }
	}
	/* --- Finalize new element------------------------------------------ */
	for (p = pk1, pk = pk1 ; pk < pk2 ; pk++)   /* finalize Lk */
	{
	    i = Ci [pk] ;
	    if ((nvi = -nv [i]) <= 0) continue ;/* skip if i is dead */
	    nv [i] = nvi ;			/* restore nv[i] */
	    d = degree [i] + dk - nvi ;		/* compute external degree(i) */
	    d = CS_MIN (d, n - nel - nvi) ;
	    if (head [d] != -1) last [head [d]] = i ;
	    next [i] = head [d] ;		/* put i back in degree list */
	    last [i] = -1 ;
	    head [d] = i ;
	    mindeg = CS_MIN (mindeg, d) ;	/* find new minimum degree */
	    degree [i] = d ;
	    Ci [p++] = i ;			/* place i in Lk */
	}
	nv [k] = nvk ;			    /* # nodes absorbed into k */
	if ((len [k] = p-pk1) == 0)	    /* length of adj list of element k*/
	{
	    Cp [k] = -1 ;		    /* k is a root of the tree */
	    w [k] = 0 ;			    /* k is now a dead element */
	}
	if (elenk != 0) cnz = p ;	    /* free unused space in Lk */
    }
    /* --- Postordering ----------------------------------------------------- */
    for (i = 0 ; i < n ; i++) Cp [i] = CS_FLIP (Cp [i]) ;/* fix assembly tree */
    for (j = 0 ; j <= n ; j++) head [j] = -1 ;
    for (j = n ; j >= 0 ; j--)		    /* place unordered nodes in lists */
    {
	if (nv [j] > 0) continue ;	    /* skip if j is an element */
	next [j] = head [Cp [j]] ;	    /* place j in list of its parent */
	head [Cp [j]] = j ;
    }
    for (e = n ; e >= 0 ; e--)		    /* place elements in lists */
    {
	if (nv [e] <= 0) continue ;	    /* skip unless e is an element */
	if (Cp [e] != -1)
	{
	    next [e] = head [Cp [e]] ;	    /* place e in list of its parent */
	    head [Cp [e]] = e ;
	}
    }
    for (k = 0, i = 0 ; i <= n ; i++)	    /* postorder the assembly tree */
    {
	if (Cp [i] == -1) k = cs_tdfs (i, k, head, next, P, w) ;
    }
    return (cs_idone (P, C, ww, 1)) ;
}

/* compute nonzero pattern of L(k,:) */
static
int cs_ereach (const cs *A, int k, const int *parent, int *s, int *w,
    double *x, int top)
{
    int i, p, len, *Ap = A->p, *Ai = A->i ;
    double *Ax = A->x ;
    for (p = Ap [k] ; p < Ap [k+1] ; p++)	/* get pattern of L(k,:) */
    {
	i = Ai [p] ;		    /* A(i,k) is nonzero */
	if (i > k) continue ;	    /* only use upper triangular part of A */
	x [i] = Ax [p] ;	    /* x(i) = A(i,k) */
	for (len = 0 ; w [i] != k ; i = parent [i]) /* traverse up etree */
	{
	    s [len++] = i ;	    /* L(k,i) is nonzero */
	    w [i] = k ;		    /* mark i as visited */
	}
	while (len > 0) s [--top] = s [--len] ; /* push path onto stack */
    }
    return (top) ;		    /* s [top..n-1] contains pattern of L(k,:)*/
}

/* L = chol (A, [Pinv parent cp]), Pinv is optional */
csn *cs_chol (const cs *A, const css *S)
{
    double d, lki, *Lx, *x ;
    int top, i, p, k, n, *Li, *Lp, *cp, *Pinv, *w, *s, *c, *parent ;
    cs *L, *C, *E ;
    csn *N ;
    if (!A || !S || !S->cp || !S->parent) return (NULL) ;   /* check inputs */
    n = A->n ;
    N = cs_calloc (1, sizeof (csn)) ;
    w = cs_malloc (3*n, sizeof (int)) ; s = w + n, c = w + 2*n ;
    x = cs_malloc (n, sizeof (double)) ;
    cp = S->cp ; Pinv = S->Pinv ; parent = S->parent ;
    C = Pinv ? cs_symperm (A, Pinv, 1) : ((cs *) A) ;
    E = Pinv ? C : NULL ;
    if (!N || !w || !x || !C) return (cs_ndone (N, E, w, x, 0)) ;
    N->L = L = cs_spalloc (n, n, cp [n], 1, 0) ;
    if (!L) return (cs_ndone (N, E, w, x, 0)) ;
    Lp = L->p ; Li = L->i ; Lx = L->x ;
    for (k = 0 ; k < n ; k++)
    {
	/* --- Nonzero pattern of L(k,:) ------------------------------------ */
	Lp [k] = c [k] = cp [k] ;   /* column k of L starts here */
	x [k] = 0 ;		    /* x (0:k) is now zero */
	w [k] = k ;		    /* mark node k as visited */
	top = cs_ereach (C, k, parent, s, w, x, n) ;   /* find row k of L*/
	d = x [k] ;		    /* d = C(k,k) */
	x [k] = 0 ;		    /* clear workspace for k+1st iteration */
	/* --- Triangular solve --------------------------------------------- */
	for ( ; top < n ; top++)    /* solve L(0:k-1,0:k-1) * x = C(:,k) */
	{
	    i = s [top] ;	    /* s [top..n-1] is pattern of L(k,:) */
	    lki = x [i] / Lx [Lp [i]] ; /* L(k,i) = x (i) / L(i,i) */
	    x [i] = 0 ;		    /* clear workspace for k+1st iteration */
	    for (p = Lp [i] + 1 ; p < c [i] ; p++)
	    {
		x [Li [p]] -= Lx [p] * lki ;
	    }
	    d -= lki * lki ;	    /* d = d - L(k,i)*L(k,i) */
	    p = c [i]++ ;
	    Li [p] = k ;	    /* store L(k,i) in column i */
	    Lx [p] = lki ;
	}
	/* --- Compute L(k,k) ----------------------------------------------- */
	if (d <= 0) return (cs_ndone (N, E, w, x, 0)) ; /* not pos def */
	p = c [k]++ ;
	Li [p] = k ;		    /* store L(k,k) = sqrt (d) in column k */
	Lx [p] = sqrt (d) ;
    }
    Lp [n] = cp [n] ;		    /* finalize L */
    return (cs_ndone (N, E, w, x, 1)) ; /* success: free E,w,x; return N */
}


/* x=A\b where A is symmetric positive definite; b overwritten with solution */
int cs_cholsol (const cs *A, double *b, int order)
{
    double *x ;
    css *S ;
    csn *N ;
    int n, ok ;
    if (!A || !b) return (0) ;		/* check inputs */
    n = A->n ;
    S = cs_schol (A, order) ;		/* ordering and symbolic analysis */
    N = cs_chol (A, S) ;		/* numeric Cholesky factorization */
    x = cs_malloc (n, sizeof (double)) ;
    ok = (S && N && x) ;
    if (ok)
    {
	cs_ipvec (n, S->Pinv, b, x) ;	/* x = P*b */
	cs_lsolve (N->L, x) ;		/* x = L\x */
	cs_ltsolve (N->L, x) ;		/* x = L'\x */
	cs_pvec (n, S->Pinv, x, b) ;	/* b = P'*x */
    }
    cs_free (x) ;
    cs_sfree (S) ;
    cs_nfree (N) ;
    return (ok) ;
}

/* process edge (j,i) of the matrix */
static void cs_cedge (int j, int i, const int *first, int *maxfirst, int *delta,
    int *prevleaf, int *ancestor)
{
    int q, s, sparent, jprev ;
    if (i <= j || first [j] <= maxfirst [i]) return ;
    maxfirst [i] = first [j] ;	/* update max first[j] seen so far */
    jprev = prevleaf [i] ;	/* j is a leaf of the ith subtree */
    delta [j]++ ;		/* A(i,j) is in the skeleton matrix */
    if (jprev != -1)
    {
	/* q = least common ancestor of jprev and j */
	for (q = jprev ; q != ancestor [q] ; q = ancestor [q]) ;
	for (s = jprev ; s != q ; s = sparent)
	{
	    sparent = ancestor [s] ;	/* path compression */
	    ancestor [s] = q ;
	}
	delta [q]-- ;		/* decrement to account for overlap in q */
    }
    prevleaf [i] = j ;		/* j is now previous leaf of ith subtree */
}

/* colcount = column counts of LL'=A or LL'=A'A, given parent & post ordering */
int *cs_counts (const cs *A, const int *parent, const int *post, int ata)
{
    int i, j, k, p, n, m, ii, s, *ATp, *ATi, *maxfirst, *prevleaf, *ancestor,
	*head = NULL, *next = NULL, *colcount, *w, *first, *delta ;
    cs *AT ;
    if (!A || !parent || !post) return (NULL) ;	    /* check inputs */
    m = A->m ; n = A->n ;
    s = 4*n + (ata ? (n+m+1) : 0) ;
    w = cs_malloc (s, sizeof (int)) ; first = w+3*n ;	/* get workspace */
    ancestor = w ; maxfirst = w+n ; prevleaf = w+2*n ;
    delta = colcount = cs_malloc (n, sizeof (int)) ;	/* allocate result */
    AT = cs_transpose (A, 0) ;
    if (!AT || !colcount || !w) return (cs_idone (colcount, AT, w, 1)) ;
    for (k = 0 ; k < s ; k++) w [k] = -1 ;	/* clear workspace w [0..s-1] */
    for (k = 0 ; k < n ; k++)			/* find first [j] */
    {
	j = post [k] ;
	delta [j] = (first [j] == -1) ? 1 : 0 ;  /* delta[j]=1 if j is a leaf */
	for ( ; j != -1 && first [j] == -1 ; j = parent [j]) first [j] = k ;
    }
    ATp = AT->p ; ATi = AT->i ;
    if (ata)
    {
	head = w+4*n ; next = w+5*n+1 ;
	for (k = 0 ; k < n ; k++) w [post [k]] = k ;    /* invert post */
	for (i = 0 ; i < m ; i++)
	{
	    k = n ;		    /* k = least postordered column in row i */
	    for (p = ATp [i] ; p < ATp [i+1] ; p++) k = CS_MIN (k, w [ATi [p]]);
	    next [i] = head [k] ;   /* place row i in link list k */
	    head [k] = i ;
	}
    }
    for (i = 0 ; i < n ; i++) ancestor [i] = i ; /* each node in its own set */
    for (k = 0 ; k < n ; k++)
    {
	j = post [k] ;		/* j is the kth node in postordered etree */
	if (parent [j] != -1) delta [parent [j]]-- ;	/* j is not a root */
	if (ata)
	{
	    for (ii = head [k] ; ii != -1 ; ii = next [ii])
	    {
		for (p = ATp [ii] ; p < ATp [ii+1] ; p++)
		    cs_cedge (j, ATi [p], first, maxfirst, delta, prevleaf,
			    ancestor) ;
	    }
	}
	else
	{
	    for (p = ATp [j] ; p < ATp [j+1] ; p++)
		cs_cedge (j, ATi [p], first, maxfirst, delta, prevleaf,
			    ancestor) ;
	}
	if (parent [j] != -1) ancestor [j] = parent [j] ;
    }
    for (j = 0 ; j < n ; j++)		/* sum up delta's of each child */
    {
	if (parent [j] != -1) colcount [parent [j]] += colcount [j] ;
    }
    return (cs_idone (colcount, AT, w, 1)) ;	/* success: free workspace */
}

/* p [0..n] = cumulative sum of c [0..n-1], and then copy p [0..n-1] into c */
int cs_cumsum (int *p, int *c, int n)
{
    int i, nz = 0 ;
    if (!p || !c) return (-1) ;	    /* check inputs */
    for (i = 0 ; i < n ; i++)
    {
	p [i] = nz ;
	nz += c [i] ;
	c [i] = p [i] ;
    }
    p [n] = nz ;
    return (nz) ;		    /* return sum (c [0..n-1]) */
}

/* depth-first-search of the graph of a matrix, starting at node j */
int cs_dfs (int j, cs *L, int top, int *xi, int *pstack, const int *Pinv)
{
    int i, p, p2, done, jnew, head = 0, *Lp, *Li ;
    if (!L || !xi || !pstack) return (-1) ;
    Lp = L->p ; Li = L->i ;
    xi [0] = j ;		/* initialize the recursion stack */
    while (head >= 0)
    {
	j = xi [head] ;		/* get j from the top of the recursion stack */
	jnew = Pinv ? (Pinv [j]) : j ;
	if (!CS_MARKED(Lp,j))
	{
	    CS_MARK (Lp,j) ;	    /* mark node j as visited */
	    pstack [head] = (jnew < 0) ? 0 : CS_UNFLIP (Lp [jnew]) ;
	}
	done = 1 ;		    /* node j done if no unvisited neighbors */
	p2 = (jnew < 0) ? 0 : CS_UNFLIP (Lp [jnew+1]) ;
	for (p = pstack [head] ; p < p2 ; p++)  /* examine all neighbors of j */
	{
	    i = Li [p] ;	    /* consider neighbor node i */
	    if (CS_MARKED (Lp,i)) continue ;	/* skip visited node i */
	    pstack [head] = p ;	    /* pause depth-first search of node j */
	    xi [++head] = i ;	    /* start dfs at node i */
	    done = 0 ;		    /* node j is not done */
	    break ;		    /* break, to start dfs (i) */
	}
	if (done)		/* depth-first search at node j is done */
	{
	    head-- ;		/* remove j from the recursion stack */
	    xi [--top] = j ;	/* and place in the output stack */
	}
    }
    return (top) ;
}

/* breadth-first search for coarse decomposition (C0,C1,R1 or R0,R3,C3) */
static int cs_bfs (const cs *A, int n, int *wi, int *wj, int *queue,
    const int *imatch, const int *jmatch, int mark)
{
    int *Ap, *Ai, head = 0, tail = 0, j, i, p, j2 ;
    cs *C ;
    for (j = 0 ; j < n ; j++)		/* place all unmatched nodes in queue */
    {
	if (imatch [j] >= 0) continue ;	/* skip j if matched */
	wj [j] = 0 ;			/* j in set C0 (R0 if transpose) */
	queue [tail++] = j ;		/* place unmatched col j in queue */
    }
    if (tail == 0) return (1) ;		/* quick return if no unmatched nodes */
    C = (mark == 1) ? ((cs *) A) : cs_transpose (A, 0) ;
    if (!C) return (0) ;		/* bfs of C=A' to find R0,R3,C3 */
    Ap = C->p ; Ai = C->i ;
    while (head < tail)			/* while queue is not empty */
    {
	j = queue [head++] ;		/* get the head of the queue */
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    if (wi [i] >= 0) continue ;	/* skip if i is marked */
	    wi [i] = mark ;		/* i in set R1 (C3 if transpose) */
	    j2 = jmatch [i] ;		/* traverse alternating path to j2 */
	    if (wj [j2] >= 0) continue ;/* skip j2 if it is marked */
	    wj [j2] = mark ;		/* j2 in set C1 (R3 if transpose) */
	    queue [tail++] = j2 ;	/* add j2 to queue */
	}
    }
    if (mark != 1) cs_spfree (C) ;	/* free A' if it was created */
    return (1) ;
}

/* collect matched rows and columns into P and Q */
static void cs_matched (int m, const int *wi, const int *jmatch, int *P, int *Q,
    int *cc, int *rr, int set, int mark)
{
    int kc = cc [set], i ;
    int kr = rr [set-1] ;
    for (i = 0 ; i < m ; i++)
    {
	if (wi [i] != mark) continue ;	    /* skip if i is not in R set */
	P [kr++] = i ;
	Q [kc++] = jmatch [i] ;
    }
    cc [set+1] = kc ;
    rr [set] = kr ;
}


static void cs_unmatched (int m, const int *wi, int *P, int *rr, int set)
/* 
  Purpose:

    CS_UNMATCHED collects unmatched rows into the permutation vector P.
*/
{
    int i, kr = rr [set] ;
    for (i = 0 ; i < m ; i++) if (wi [i] == 0) P [kr++] = i ;
    rr [set+1] = kr ;
}

/* return 1 if row i is in R2 */
static int cs_rprune (int i, int j, double aij, void *other)
{
    int *rr = (int *) other ;
    return (i >= rr [1] && i < rr [2]) ;
	(void)j; (void)aij; // unused parameters
}

/* Given A, find coarse dmperm */
csd *cs_dmperm (const cs *A)
{
    int m, n, i, j, k, p, cnz, nc, *jmatch, *imatch, *wi, *wj, *Pinv, *Cp, *Ci,
	*Ps, *Rs, nb1, nb2, *P, *Q, *cc, *rr, *R, *S, ok ;
    cs *C ;
    csd *D, *scc ;
    /* --- Maximum matching ------------------------------------------------- */
    if (!A) return (NULL) ;			/* check inputs */
    m = A->m ; n = A->n ;
    D = cs_dalloc (m, n) ;			/* allocate result */
    if (!D) return (NULL) ;
    P = D->P ; Q = D->Q ; R = D->R ; S = D->S ; cc = D->cc ; rr = D->rr ;
    jmatch = cs_maxtrans (A) ;			/* max transversal */
    imatch = jmatch + m ;			/* imatch = inverse of jmatch */
    if (!jmatch) return (cs_ddone (D, NULL, jmatch, 0)) ;
    /* --- Coarse decomposition --------------------------------------------- */
    wi = R ; wj = S ;				/* use R and S as workspace */
    for (j = 0 ; j < n ; j++) wj [j] = -1 ;	/* unmark all cols for bfs */
    for (i = 0 ; i < m ; i++) wi [i] = -1 ;	/* unmark all rows for bfs */
    cs_bfs (A, n, wi, wj, Q, imatch, jmatch, 1) ;	/* find C0, C1, R1 */
    ok = cs_bfs (A, m, wj, wi, P, jmatch, imatch, 3) ;	/* find R0, R3, C3 */
    if (!ok) return (cs_ddone (D, NULL, jmatch, 0)) ;
    cs_unmatched (n, wj, Q, cc, 0) ;			/* unmatched set C0 */
    cs_matched (m, wi, jmatch, P, Q, cc, rr, 1, 1) ;	/* set R1 and C1 */
    cs_matched (m, wi, jmatch, P, Q, cc, rr, 2, -1) ;	/* set R2 and C2 */
    cs_matched (m, wi, jmatch, P, Q, cc, rr, 3, 3) ;	/* set R3 and C3 */
    cs_unmatched (m, wi, P, rr, 3) ;			/* unmatched set R0 */
    cs_free (jmatch) ;
    /* --- Fine decomposition ----------------------------------------------- */
    Pinv = cs_pinv (P, m) ;	    /* Pinv=P' */
    if (!Pinv) return (cs_ddone (D, NULL, NULL, 0)) ;
    C = cs_permute (A, Pinv, Q, 0) ;/* C=A(P,Q) (it will hold A(R2,C2)) */
    cs_free (Pinv) ;
    if (!C) return (cs_ddone (D, NULL, NULL, 0)) ;
    Cp = C->p ; Ci = C->i ;
    nc = cc [3] - cc [2] ;	    /* delete cols C0, C1, and C3 from C */
    if (cc [2] > 0) for (j = cc [2] ; j <= cc [3] ; j++) Cp [j-cc[2]] = Cp [j] ;
    C->n = nc ;
    if (rr [2] - rr [1] < m)	    /* delete rows R0, R1, and R3 from C */
    {
	cs_fkeep (C, cs_rprune, rr) ;
	cnz = Cp [nc] ;
	if (rr [1] > 0) for (p = 0 ; p < cnz ; p++) Ci [p] -= rr [1] ;
    }
    C->m = nc ;
    scc = cs_scc (C) ;		    /* find strongly-connected components of C*/
    if (!scc) return (cs_ddone (D, C, NULL, 0)) ;
    /* --- Combine coarse and fine decompositions --------------------------- */
    Ps = scc->P ;		    /* C(Ps,Ps) is the permuted matrix */
    Rs = scc->R ;		    /* kth block is Rs[k]..Rs[k+1]-1 */
    nb1 = scc->nb  ;		    /* # of blocks of A(*/
    for (k = 0 ; k < nc ; k++) wj [k] = Q [Ps [k] + cc [2]] ;	/* combine */
    for (k = 0 ; k < nc ; k++) Q [k + cc [2]] = wj [k] ;
    for (k = 0 ; k < nc ; k++) wi [k] = P [Ps [k] + rr [1]] ;
    for (k = 0 ; k < nc ; k++) P [k + rr [1]] = wi [k] ;
    nb2 = 0 ;			    /* create the fine block partitions */
    R [0] = 0 ;
    S [0] = 0 ;
    if (cc [2] > 0) nb2++ ;	    /* leading coarse block A (R1, [C0 C1]) */
    for (k = 0 ; k < nb1 ; k++)	    /* coarse block A (R2,C2) */
    {
	R [nb2] = Rs [k] + rr [1] ; /* A (R2,C2) splits into nb1 fine blocks */
	S [nb2] = Rs [k] + cc [2] ;
	nb2++ ;
    }
    if (rr [2] < m)
    {
	R [nb2] = rr [2] ;	    /* trailing coarse block A ([R3 R0], C3) */
	S [nb2] = cc [3] ;
	nb2++ ;
    }
    R [nb2] = m ;
    S [nb2] = n ;
    D->nb = nb2 ;
    cs_dfree (scc) ;
    return (cs_ddone (D, C, NULL, 1)) ;
}

static int cs_tol (int i, int j, double aij, void *tol)
{
    return (fabs (aij) > *((double *) tol)) ;
	(void)i; (void)j; // unused parameters
}
int cs_droptol (cs *A, double tol)
{
    return (cs_fkeep (A, &cs_tol, &tol)) ;    /* keep all large entries */
}

static int cs_nonzero (int i, int j, double aij, void *other)
{
    return (aij != 0) ;
	(void)i; (void)j; (void)other; // unused parameters
}
int cs_dropzeros (cs *A)
{
    return (cs_fkeep (A, &cs_nonzero, NULL)) ;	/* keep all nonzero entries */
}
int cs_dupl (cs *A)
/*
  Purpose:

    CS_DUPL removes duplicate entries from A.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int i, j, p, q, nz = 0, n, m, *Ap, *Ai, *w ;
    double *Ax ;
    if (!A) return (0) ;			/* check inputs */
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    w = cs_malloc (m, sizeof (int)) ;		/* get workspace */
    if (!w) return (0) ;			/* out of memory */
    for (i = 0 ; i < m ; i++) w [i] = -1 ;	/* row i not yet seen */
    for (j = 0 ; j < n ; j++)
    {
	q = nz ;				/* column j will start at q */
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;			/* A(i,j) is nonzero */
	    if (w [i] >= q)
	    {
		Ax [w [i]] += Ax [p] ;		/* A(i,j) is a duplicate */
	    }
	    else
	    {
		w [i] = nz ;			/* record where row i occurs */
		Ai [nz] = i ;			/* keep A(i,j) */
		Ax [nz++] = Ax [p] ;
	    }
	}
	Ap [j] = q ;				/* record start of column j */
    }
    Ap [n] = nz ;				/* finalize A */
    cs_free (w) ;				/* free workspace */
    return (cs_sprealloc (A, 0)) ;		/* remove extra space from A */
}

/* add an entry to a triplet matrix; return 1 if ok, 0 otherwise */
int cs_entry (cs *T, int i, int j, double x)
{
    if (!T || (T->nz >= T->nzmax && !cs_sprealloc (T, 2*(T->nzmax)))) return(0);
    if (T->x) T->x [T->nz] = x ;
    T->i [T->nz] = i ;
    T->p [T->nz++] = j ;
    T->m = CS_MAX (T->m, i+1) ;
    T->n = CS_MAX (T->n, j+1) ;
    return (1) ;
}

/* compute the etree of A (using triu(A), or A'A without forming A'A */
int *cs_etree (const cs *A, int ata)
{
    int i, k, p, m, n, inext, *Ap, *Ai, *w, *parent, *ancestor, *prev ;
    if (!A) return (NULL) ;		    /* check inputs */
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ;
    parent = cs_malloc (n, sizeof (int)) ;
    w = cs_malloc (n + (ata ? m : 0), sizeof (int)) ;
    ancestor = w ; prev = w + n ;
    if (!w || !parent) return (cs_idone (parent, NULL, w, 0)) ;
    if (ata) for (i = 0 ; i < m ; i++) prev [i] = -1 ;
    for (k = 0 ; k < n ; k++)
    {
	parent [k] = -1 ;		    /* node k has no parent yet */
	ancestor [k] = -1 ;		    /* nor does k have an ancestor */
	for (p = Ap [k] ; p < Ap [k+1] ; p++)
	{
	    i = ata ? (prev [Ai [p]]) : (Ai [p]) ;
	    for ( ; i != -1 && i < k ; i = inext)   /* traverse from i to k */
	    {
		inext = ancestor [i] ;		    /* inext = ancestor of i */
		ancestor [i] = k ;		    /* path compression */
		if (inext == -1) parent [i] = k ;   /* no anc., parent is k */
	    }
	    if (ata) prev [Ai [p]] = k ;
	}
    }
    return (cs_idone (parent, NULL, w, 1)) ;
}

/* drop entries for which fkeep(A(i,j)) is false; return nz if OK, else -1 */
int cs_fkeep (cs *A, int (*fkeep) (int, int, double, void *), void *other)
{
    int j, p, nz = 0, n, *Ap, *Ai ;
    double *Ax ;
    if (!A || !fkeep) return (-1) ;	    /* check inputs */
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    for (j = 0 ; j < n ; j++)
    {
	p = Ap [j] ;			    /* get current location of col j */
	Ap [j] = nz ;			    /* record new location of col j */
	for ( ; p < Ap [j+1] ; p++)
	{
	    if (fkeep (Ai [p], j, Ax ? Ax [p] : 1, other))
	    {
		if (Ax) Ax [nz] = Ax [p] ;  /* keep A(i,j) */
		Ai [nz++] = Ai [p] ;
	    }
	}
    }
    return (Ap [n] = nz) ;		    /* finalize A and return nnz(A) */
}

/* y = A*x+y */
int cs_gaxpy (const cs *A, const double *x, double *y)
{
    int p, j, n, *Ap, *Ai ;
    double *Ax ;
    if (!A || !x || !y) return (0) ;	    /* check inputs */
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    for (j = 0 ; j < n ; j++)
    {
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    y [Ai [p]] += Ax [p] * x [j] ;
	}
    }
    return (1) ;
}

/* apply the ith Householder vector to x */
int cs_happly (const cs *V, int i, double beta, double *x)
{
    int p, *Vp, *Vi ;
    double *Vx, tau = 0 ;
    if (!V || !x) return (0) ;		    /* check inputs */
    Vp = V->p ; Vi = V->i ; Vx = V->x ;
    for (p = Vp [i] ; p < Vp [i+1] ; p++)   /* tau = v'*x */
    {
	tau += Vx [p] * x [Vi [p]] ;
    }
    tau *= beta ;			    /* tau = beta*(v'*x) */
    for (p = Vp [i] ; p < Vp [i+1] ; p++)   /* x = x - v*tau */
    {
	x [Vi [p]] -= Vx [p] * tau ;
    }
    return (1) ;
}

/* create a Householder reflection [v,beta,s]=house(x), overwrite x with v,
 * where (I-beta*v*v')*x = s*x.  See Algo 5.1.1, Golub & Van Loan, 3rd ed. */
double cs_house (double *x, double *beta, int n)
{
    double s, sigma = 0 ;
    int i ;
    if (!x || !beta) return (-1) ;	    /* check inputs */
    for (i = 1 ; i < n ; i++) sigma += x [i] * x [i] ;
    if (sigma == 0)
    {
	s = fabs (x [0]) ;		    /* s = |x(0)| */
	(*beta) = (x [0] <= 0) ? 2 : 0 ;
	x [0] = 1 ;
    }
    else
    {
	s = sqrt (x [0] * x [0] + sigma) ;  /* s = norm (x) */
	x [0] = (x [0] <= 0) ? (x [0] - s) : (-sigma / (x [0] + s)) ;
	(*beta) = -1. / (s * x [0]) ;
    }
    return (s) ;
}

/* x(P) = b, for dense vectors x and b; P=NULL denotes identity */
int cs_ipvec (int n, const int *P, const double *b, double *x)
{
    int k ;
    if (!x || !b) return (0) ;				    /* check inputs */
    for (k = 0 ; k < n ; k++) x [P ? P [k] : k] = b [k] ;
    return (1) ;
}
cs *cs_load ( FILE *f )
/*
  Purpose:

    CS_LOAD loads a triplet matrix from a file.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int i, j ;
    double x ;
    cs *T ;
    if (!f) return (NULL) ;
    T = cs_spalloc (0, 0, 1, 1, 1) ;
    while (fscanf (f, "%d %d %lg\n", &i, &j, &x) == 3)
    {
	if (!cs_entry (T, i, j, x)) return (cs_spfree (T)) ;
    }
    return (T) ;
}
int cs_lsolve ( const cs *L, double *x )
/*
  Purpose:

    CS_LSOLVE solves L*x=b.

  Discussion:

    On input, X contains the right hand side, and on output, the solution.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int p, j, n, *Lp, *Li ;
    double *Lx ;
    if (!L || !x) return (0) ;				    /* check inputs */
    n = L->n ; Lp = L->p ; Li = L->i ; Lx = L->x ;
    for (j = 0 ; j < n ; j++)
    {
	x [j] /= Lx [Lp [j]] ;
	for (p = Lp [j]+1 ; p < Lp [j+1] ; p++)
	{
	    x [Li [p]] -= Lx [p] * x [j] ;
	}
    }
    return (1) ;
}
int cs_ltsolve ( const cs *L, double *x )
/*
  Purpose:

    CS_LTSOLVE solves L'*x=b.

  Discussion:

    On input, X contains the right hand side, and on output, the solution.

  Reference:

    Timothy Davis,
    Direct Methods for Sparse Linear Systems,
    SIAM, Philadelphia, 2006.
*/
{
    int p, j, n, *Lp, *Li ;
    double *Lx ;
    if (!L || !x) return (0) ;				    /* check inputs */
    n = L->n ; Lp = L->p ; Li = L->i ; Lx = L->x ;
    for (j = n-1 ; j >= 0 ; j--)
    {
	for (p = Lp [j]+1 ; p < Lp [j+1] ; p++)
	{
	    x [j] -= Lx [p] * x [Li [p]] ;
	}
	x [j] /= Lx [Lp [j]] ;
    }
    return (1) ;
}

/* [L,U,Pinv]=lu(A, [Q lnz unz]). lnz and unz can be guess */
csn *cs_lu (const cs *A, const css *S, double tol)
{
    cs *L, *U ;
    csn *N ;
    double pivot, *Lx, *Ux, *x,  a, t ;
    int *Lp, *Li, *Up, *Ui, *Pinv, *xi, *Q, n, ipiv, k, top, p, i, col, lnz,unz;
    if (!A || !S) return (NULL) ;		    /* check inputs */
    n = A->n ;
    Q = S->Q ; lnz = S->lnz ; unz = S->unz ;
    x = cs_malloc (n, sizeof (double)) ;
    xi = cs_malloc (2*n, sizeof (int)) ;
    N = cs_calloc (1, sizeof (csn)) ;
    if (!x || !xi || !N) return (cs_ndone (N, NULL, xi, x, 0)) ;
    N->L = L = cs_spalloc (n, n, lnz, 1, 0) ;	    /* initial L and U */
    N->U = U = cs_spalloc (n, n, unz, 1, 0) ;
    N->Pinv = Pinv = cs_malloc (n, sizeof (int)) ;
    if (!L || !U || !Pinv) return (cs_ndone (N, NULL, xi, x, 0)) ;
    Lp = L->p ; Up = U->p ;
    for (i = 0 ; i < n ; i++) x [i] = 0 ;	    /* clear workspace */
    for (i = 0 ; i < n ; i++) Pinv [i] = -1 ;	    /* no rows pivotal yet */
    for (k = 0 ; k <= n ; k++) Lp [k] = 0 ;	    /* no cols of L yet */
    lnz = unz = 0 ;
    for (k = 0 ; k < n ; k++)	    /* compute L(:,k) and U(:,k) */
    {
	/* --- Triangular solve --------------------------------------------- */
	Lp [k] = lnz ;		    /* L(:,k) starts here */
	Up [k] = unz ;		    /* U(:,k) starts here */
	if ((lnz + n > L->nzmax && !cs_sprealloc (L, 2*L->nzmax + n)) ||
	    (unz + n > U->nzmax && !cs_sprealloc (U, 2*U->nzmax + n)))
	{
	    return (cs_ndone (N, NULL, xi, x, 0)) ;
	}
	Li = L->i ; Lx = L->x ; Ui = U->i ; Ux = U->x ;
	col = Q ? (Q [k]) : k ;
	top = cs_splsolve (L, A, col, xi, x, Pinv) ; /* x = L\A(:,col) */
	/* --- Find pivot --------------------------------------------------- */
	ipiv = -1 ;
	a = -1 ;
	for (p = top ; p < n ; p++)
	{
	    i = xi [p] ;	    /* x(i) is nonzero */
	    if (Pinv [i] < 0)	    /* row i is not pivotal */
	    {
		if ((t = fabs (x [i])) > a)
		{
		    a = t ;	    /* largest pivot candidate so far */
		    ipiv = i ;
		}
	    }
	    else		    /* x(i) is the entry U(Pinv[i],k) */
	    {
		Ui [unz] = Pinv [i] ;
		Ux [unz++] = x [i] ;
	    }
	}
	if (ipiv == -1 || a <= 0) return (cs_ndone (N, NULL, xi, x, 0)) ;
	if (Pinv [col] < 0 && fabs (x [col]) >= a*tol) ipiv = col ;
	/* --- Divide by pivot ---------------------------------------------- */
	pivot = x [ipiv] ;	    /* the chosen pivot */
	Ui [unz] = k ;		    /* last entry in U(:,k) is U(k,k) */
	Ux [unz++] = pivot ;
	Pinv [ipiv] = k ;	    /* ipiv is the kth pivot row */
	Li [lnz] = ipiv ;	    /* first entry in L(:,k) is L(k,k) = 1 */
	Lx [lnz++] = 1 ;
	for (p = top ; p < n ; p++) /* L(k+1:n,k) = x / pivot */
	{
	    i = xi [p] ;
	    if (Pinv [i] < 0)	    /* x(i) is an entry in L(:,k) */
	    {
		Li [lnz] = i ;	    /* save unpermuted row in L */
		Lx [lnz++] = x [i] / pivot ;	/* scale pivot column */
	    }
	    x [i] = 0 ;		    /* x [0..n-1] = 0 for next k */
	}
    }
    /* --- Finalize L and U ------------------------------------------------- */
    Lp [n] = lnz ;
    Up [n] = unz ;
    Li = L->i ;			    /* fix row indices of L for final Pinv */
    for (p = 0 ; p < lnz ; p++) Li [p] = Pinv [Li [p]] ;
    cs_sprealloc (L, 0) ;	    /* remove extra space from L and U */
    cs_sprealloc (U, 0) ;
    return (cs_ndone (N, NULL, xi, x, 1)) ;	/* success */
}

/* x=A\b where A is unsymmetric; b overwritten with solution */
int cs_lusol (const cs *A, double *b, int order, double tol)
{
    double *x ;
    css *S ;
    csn *N ;
    int n, ok ;
    if (!A || !b) return (0) ;		/* check inputs */
    n = A->n ;
    S = cs_sqr (A, order, 0) ;		/* ordering and symbolic analysis */
    N = cs_lu (A, S, tol) ;		/* numeric LU factorization */
    x = cs_malloc (n, sizeof (double)) ;
    ok = (S && N && x) ;
    if (ok)
    {
	cs_ipvec (n, N->Pinv, b, x) ;	/* x = P*b */
	cs_lsolve (N->L, x) ;		/* x = L\x */
	cs_usolve (N->U, x) ;		/* x = U\x */
	cs_ipvec (n, S->Q, x, b) ;	/* b = Q*x */
    }
    cs_free (x) ;
    cs_sfree (S) ;
    cs_nfree (N) ;
    return (ok) ;
}

#ifdef MATLAB_MEX_FILE
#define malloc mxMalloc
#define free mxFree
#define realloc mxRealloc
#define calloc mxCalloc
#endif

/* wrapper for malloc */
void *cs_malloc (int n, size_t size)
{
    return (CS_OVERFLOW (n,size) ? NULL : malloc (CS_MAX (n,1) * size)) ;
}

/* wrapper for calloc */
void *cs_calloc (int n, size_t size)
{
    return (CS_OVERFLOW (n,size) ? NULL : calloc (CS_MAX (n,1), size)) ;
}

/* wrapper for free */
void *cs_free (void *p)
{
    if (p) free (p) ;	    /* free p if it is not already NULL */
    return (NULL) ;	    /* return NULL to simplify the use of cs_free */
}

/* wrapper for realloc */
void *cs_realloc (void *p, int n, size_t size, int *ok)
{
    void *p2 ;
    *ok = !CS_OVERFLOW (n,size) ;	    /* guard against int overflow */
    if (!(*ok)) return (p) ;		    /* p unchanged if n too large */
    p2 = realloc (p, CS_MAX (n,1) * size) ; /* realloc the block */
    *ok = (p2 != NULL) ;
    return ((*ok) ? p2 : p) ;		    /* return original p if failure */
}

/* find an augmenting path starting at column k and extend the match if found */
static void cs_augment (int k, const cs *A, int *jmatch, int *cheap, int *w,
	int *js, int *is, int *ps)
{
    int found = 0, p, i = -1, *Ap = A->p, *Ai = A->i, head = 0, j ;
    js [0] = k ;			/* start with just node k in jstack */
    while (head >= 0)
    {
	/* --- Start (or continue) depth-first-search at node j ------------- */
	j = js [head] ;			/* get j from top of jstack */
	if (w [j] != k)			/* 1st time j visited for kth path */
	{
	    w [j] = k ;			/* mark j as visited for kth path */
	    for (p = cheap [j] ; p < Ap [j+1] && !found ; p++)
	    {
		i = Ai [p] ;		/* try a cheap assignment (i,j) */
		found = (jmatch [i] == -1) ;
	    }
	    cheap [j] = p ;		/* start here next time j is traversed*/
	    if (found)
	    {
		is [head] = i ;		/* column j matched with row i */
		break ;			/* end of augmenting path */
	    }
	    ps [head] = Ap [j] ;	/* no cheap match: start dfs for j */
	}
	/* --- Depth-first-search of neighbors of j ------------------------- */
	for (p = ps [head] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;		/* consider row i */
	    if (w [jmatch [i]] == k) continue ;	/* skip jmatch [i] if marked */
	    ps [head] = p + 1 ;		/* pause dfs of node j */
	    is [head] = i ;		/* i will be matched with j if found */
	    js [++head] = jmatch [i] ;	/* start dfs at column jmatch [i] */
	    break ;
	}
	if (p == Ap [j+1]) head-- ;	/* node j is done; pop from stack */
    }					/* augment the match if path found: */
    if (found) for (p = head ; p >= 0 ; p--) jmatch [is [p]] = js [p] ;
}

/* find a maximum transveral */
int *cs_maxtrans (const cs *A)   /* returns jmatch [0..m-1]; imatch [0..n-1] */
{
    int i, j, k, n, m, p, n2 = 0, m2 = 0, *Ap, *jimatch, *w, *cheap, *js, *is,
	*ps, *Ai, *Cp, *jmatch, *imatch ;
    cs *C ;
    if (!A) return (NULL) ;			    /* check inputs */
    n = A->n ; m = A->m ; Ap = A->p ; Ai = A->i ;
    w = jimatch = cs_calloc (m+n, sizeof (int)) ;   /* allocate result */
    if (!jimatch) return (NULL) ;
    for (j = 0 ; j < n ; j++)		/* count non-empty rows and columns */
    {
	n2 += (Ap [j] < Ap [j+1]) ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++) w [Ai [p]] = 1 ;
    }
    for (i = 0 ; i < m ; i++) m2 += w [i] ;
    C = (m2 < n2) ? cs_transpose (A,0) : ((cs *) A) ; /* transpose if needed */
    if (!C) return (cs_idone (jimatch, (m2 < n2) ? C : NULL, NULL, 0)) ;
    n = C->n ; m = C->m ; Cp = C->p ;
    jmatch = (m2 < n2) ? jimatch + n : jimatch ;
    imatch = (m2 < n2) ? jimatch : jimatch + m ;
    w = cs_malloc (5*n, sizeof (int)) ;		    /* allocate workspace */
    if (!w) return (cs_idone (jimatch, (m2 < n2) ? C : NULL, w, 0)) ;
    cheap = w + n ; js = w + 2*n ; is = w + 3*n ; ps = w + 4*n ;
    for (j = 0 ; j < n ; j++) cheap [j] = Cp [j] ;  /* for cheap assignment */
    for (j = 0 ; j < n ; j++) w [j] = -1 ;	    /* all columns unflagged */
    for (i = 0 ; i < m ; i++) jmatch [i] = -1 ;	    /* nothing matched yet */
    for (k = 0 ; k < n ; k++) cs_augment (k, C, jmatch, cheap, w, js, is, ps) ;
    for (j = 0 ; j < n ; j++) imatch [j] = -1 ;	    /* find row match */
    for (i = 0 ; i < m ; i++) if (jmatch [i] >= 0) imatch [jmatch [i]] = i ;
    return (cs_idone (jimatch, (m2 < n2) ? C : NULL, w, 1)) ;
}

/* C = A*B */
cs *cs_multiply (const cs *A, const cs *B)
{
    int p, j, nz = 0, anz, *Cp, *Ci, *Bp, m, n, bnz, *w, values, *Bi ;
    double *x, *Bx, *Cx ;
    cs *C ;
    if (!A || !B) return (NULL) ;	/* check inputs */
    m = A->m ; anz = A->p [A->n] ;
    n = B->n ; Bp = B->p ; Bi = B->i ; Bx = B->x ; bnz = Bp [n] ;
    w = cs_calloc (m, sizeof (int)) ;
    values = (A->x != NULL) && (Bx != NULL) ;
    x = values ? cs_malloc (m, sizeof (double)) : NULL ;
    C = cs_spalloc (m, n, anz + bnz, values, 0) ;
    if (!C || !w || (values && !x)) return (cs_done (C, w, x, 0)) ;
    Cp = C->p ;
    for (j = 0 ; j < n ; j++)
    {
	if (nz + m > C->nzmax && !cs_sprealloc (C, 2*(C->nzmax)+m))
	{
	    return (cs_done (C, w, x, 0)) ;		/* out of memory */
	} 
	Ci = C->i ; Cx = C->x ;		/* C may have been reallocated */
	Cp [j] = nz ;			/* column j of C starts here */
	for (p = Bp [j] ; p < Bp [j+1] ; p++)
	{
	    nz = cs_scatter (A, Bi [p], Bx ? Bx [p] : 1, w, x, j+1, C, nz) ;
	}
	if (values) for (p = Cp [j] ; p < nz ; p++) Cx [p] = x [Ci [p]] ;
    }
    Cp [n] = nz ;			/* finalize the last column of C */
    cs_sprealloc (C, 0) ;		/* remove extra space from C */
    return (cs_done (C, w, x, 1)) ;	/* success; free workspace, return C */
}

/* 1-norm of a sparse matrix = max (sum (abs (A))), largest column sum */
double cs_norm (const cs *A)
{
    int p, j, n, *Ap ;
    double *Ax,  norm = 0, s ;
    if (!A || !A->x) return (-1) ;		/* check inputs */
    n = A->n ; Ap = A->p ; Ax = A->x ;
    for (j = 0 ; j < n ; j++)
    {
	for (s = 0, p = Ap [j] ; p < Ap [j+1] ; p++) s += fabs (Ax [p]) ;
	norm = CS_MAX (norm, s) ;
    }
    return (norm) ;
}

/* C = A(P,Q) where P and Q are permutations of 0..m-1 and 0..n-1. */
cs *cs_permute (const cs *A, const int *Pinv, const int *Q, int values)
{
    int p, j, k, nz = 0, m, n, *Ap, *Ai, *Cp, *Ci ;
    double *Cx, *Ax ;
    cs *C ;
    if (!A) return (NULL) ;		/* check inputs */
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    C = cs_spalloc (m, n, Ap [n], values && Ax != NULL, 0) ;
    if (!C) return (cs_done (C, NULL, NULL, 0)) ;   /* out of memory */
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (k = 0 ; k < n ; k++)
    {
	Cp [k] = nz ;			/* column k of C is column Q[k] of A */
	j = Q ? (Q [k]) : k ;
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    if (Cx) Cx [nz] = Ax [p] ;	/* row i of A is row Pinv[i] of C */
	    Ci [nz++] = Pinv ? (Pinv [Ai [p]]) : Ai [p] ;
	}
    }
    Cp [n] = nz ;			/* finalize the last column of C */
    return (cs_done (C, NULL, NULL, 1)) ;
}

/* Pinv = P', or P = Pinv' */
int *cs_pinv (int const *P, int n)
{
    int k, *Pinv ;
    if (!P) return (NULL) ;			/* P = NULL denotes identity */
    Pinv = cs_malloc (n, sizeof (int)) ;	/* allocate resuult */
    if (!Pinv) return (NULL) ;			/* out of memory */
    for (k = 0 ; k < n ; k++) Pinv [P [k]] = k ;/* invert the permutation */
    return (Pinv) ;				/* return result */
}

/* post order a forest */
int *cs_post (int n, const int *parent)
{
    int j, k = 0, *post, *w, *head, *next, *stack ;
    if (!parent) return (NULL) ;			/* check inputs */
    post = cs_malloc (n, sizeof (int)) ;		/* allocate result */
    w = cs_malloc (3*n, sizeof (int)) ;			/* 3*n workspace */
    head = w ; next = w + n ; stack = w + 2*n ;
    if (!w || !post) return (cs_idone (post, NULL, w, 0)) ;
    for (j = 0 ; j < n ; j++) head [j] = -1 ;		/* empty link lists */
    for (j = n-1 ; j >= 0 ; j--)	    /* traverse nodes in reverse order*/
    {
	if (parent [j] == -1) continue ;    /* j is a root */
	next [j] = head [parent [j]] ;	    /* add j to list of its parent */
	head [parent [j]] = j ;
    }
    for (j = 0 ; j < n ; j++)
    {
	if (parent [j] != -1) continue ;    /* skip j if it is not a root */
	k = cs_tdfs (j, k, head, next, post, stack) ;
    }
    return (cs_idone (post, NULL, w, 1)) ;  /* success; free w, return post */
}

/* print a sparse matrix */
int cs_print (const cs *A, int brief)
{
    int p, j, m, n, nzmax, nz, *Ap, *Ai ;
    double *Ax ;
    if (!A) { printf ("(null)\n") ; return (0) ; }
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    nzmax = A->nzmax ; nz = A->nz ;
    printf ("CSparse Version %d.%d.%d, %s.  %s\n", CS_VER, CS_SUBVER,
	CS_SUBSUB, CS_DATE, CS_COPYRIGHT) ;
    if (nz < 0)
    {
	printf ("%d-by-%d, nzmax: %d nnz: %d, 1-norm: %g\n", m, n, nzmax,
		Ap [n], cs_norm (A)) ;
	for (j = 0 ; j < n ; j++)
	{
	    printf ("    col %d : locations %d to %d\n", j, Ap [j], Ap [j+1]-1);
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		printf ("      %d : %g\n", Ai [p], Ax ? Ax [p] : 1) ;
		if (brief && p > 20) { printf ("  ...\n") ; return (1) ; }
	    }
	}
    }
    else
    {
	printf ("triplet: %d-by-%d, nzmax: %d nnz: %d\n", m, n, nzmax, nz) ;
	for (p = 0 ; p < nz ; p++)
	{
	    printf ("    %d %d : %g\n", Ai [p], Ap [p], Ax ? Ax [p] : 1) ;
	    if (brief && p > 20) { printf ("  ...\n") ; return (1) ; }
	}
    }
    return (1) ;
}

/* x = b(P), for dense vectors x and b; P=NULL denotes identity */
int cs_pvec (int n, const int *P, const double *b, double *x)
{
    int k ;
    if (!x || !b) return (0) ;				    /* check inputs */
    for (k = 0 ; k < n ; k++) x [k] = b [P ? P [k] : k] ;
    return (1) ;
}

/* sparse QR factorization [V,beta,p,R] = qr (A) */
csn *cs_qr (const cs *A, const css *S)
{
    double *Rx, *Vx, *Ax, *Beta, *x ;
    int i, k, p, m, n, vnz, p1, top, m2, len, col, rnz, *s, *leftmost, *Ap,
	*Ai, *parent, *Rp, *Ri, *Vp, *Vi, *w, *Pinv, *Q ;
    cs *R, *V ;
    csn *N ;
    if (!A || !S || !S->parent || !S->Pinv) return (NULL) ; /* check inputs */
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    Q = S->Q ; parent = S->parent ; Pinv = S->Pinv ; m2 = S->m2 ;
    vnz = S->lnz ; rnz = S->unz ;
    leftmost = Pinv + m + n ;
    w = cs_malloc (m2+n, sizeof (int)) ;
    x = cs_malloc (m2, sizeof (double)) ;
    N = cs_calloc (1, sizeof (csn)) ;
    if (!w || !x || !N) return (cs_ndone (N, NULL, w, x, 0)) ;
    s = w + m2 ;				/* size n */
    for (k = 0 ; k < m2 ; k++) x [k] = 0 ;	/* clear workspace x */
    N->L = V = cs_spalloc (m2, n, vnz, 1, 0) ;	/* allocate V */
    N->U = R = cs_spalloc (m2, n, rnz, 1, 0) ;	/* allocate R, m2-by-n */
    N->B = Beta = cs_malloc (n, sizeof (double)) ;
    if (!R || !V || !Beta) return (cs_ndone (N, NULL, w, x, 0)) ;
    Rp = R->p ; Ri = R->i ; Rx = R->x ;
    Vp = V->p ; Vi = V->i ; Vx = V->x ;
    for (i = 0 ; i < m2 ; i++) w [i] = -1 ;	/* clear w, to mark nodes */
    rnz = 0 ; vnz = 0 ;
    for (k = 0 ; k < n ; k++)		    /* compute V and R */
    {
	Rp [k] = rnz ;			    /* R(:,k) starts here */
	Vp [k] = p1 = vnz ;		    /* V(:,k) starts here */
	w [k] = k ;			    /* add V(k,k) to pattern of V */
	Vi [vnz++] = k ;
	top = n ;
	col = Q ? Q [k] : k ;
	for (p = Ap [col] ; p < Ap [col+1] ; p++)   /* find R(:,k) pattern */
	{
	    i = leftmost [Ai [p]] ;	    /* i = min(find(A(i,Q))) */
	    for (len = 0 ; w [i] != k ; i = parent [i])	/* traverse up to k */
	    {
		s [len++] = i ;
		w [i] = k ;
	    }
	    while (len > 0) s [--top] = s [--len] ; /* push path on stack */
	    i = Pinv [Ai [p]] ;		    /* i = permuted row of A(:,col) */
	    x [i] = Ax [p] ;		    /* x (i) = A(.,col) */
	    if (i > k && w [i] < k)	    /* pattern of V(:,k) = x (k+1:m) */
	    {
		Vi [vnz++] = i ;	    /* add i to pattern of V(:,k) */
		w [i] = k ;
	    }
	}
	for (p = top ; p < n ; p++) /* for each i in pattern of R(:,k) */
	{
	    i = s [p] ;			    /* R(i,k) is nonzero */
	    cs_happly (V, i, Beta [i], x) ; /* apply (V(i),Beta(i)) to x */
	    Ri [rnz] = i ;		    /* R(i,k) = x(i) */
	    Rx [rnz++] = x [i] ;
	    x [i] = 0 ;
	    if (parent [i] == k) vnz = cs_scatter (V, i, 0, w, NULL, k, V, vnz);
	}
	for (p = p1 ; p < vnz ; p++)	    /* gather V(:,k) = x */
	{
	    Vx [p] = x [Vi [p]] ;
	    x [Vi [p]] = 0 ;
	}
	Ri [rnz] = k ;			   /* R(k,k) = norm (x) */
	Rx [rnz++] = cs_house (Vx+p1, Beta+k, vnz-p1) ;	/* [v,beta]=house(x) */
    }
    Rp [n] = rnz ;			    /* finalize R */
    Vp [n] = vnz ;			    /* finalize V */
    return (cs_ndone (N, NULL, w, x, 1)) ;  /* success */
}

/* x=A\b where A can be rectangular; b overwritten with solution */
int cs_qrsol (const cs *A, double *b, int order)
{
    double *x ;
    css *S ;
    csn *N ;
    cs *AT = NULL ;
    int k, m, n, ok ;
    if (!A || !b) return (0) ;		/* check inputs */
    n = A->n ;
    m = A->m ;
    if (m >= n)
    {
	S = cs_sqr (A, order, 1) ;	/* ordering and symbolic analysis */
	N = cs_qr (A, S) ;		/* numeric QR factorization */
	x = cs_calloc (S ? S->m2 : 1, sizeof (double)) ;
	ok = (S && N && x) ;
	if (ok)
	{
	    cs_ipvec (m, S->Pinv, b, x) ;   /* x(0:m-1) = P*b(0:m-1) */
	    for (k = 0 ; k < n ; k++)	    /* apply Householder refl. to x */
	    {
		cs_happly (N->L, k, N->B [k], x) ;
	    }
	    cs_usolve (N->U, x) ;	    /* x = R\x */
	    cs_ipvec (n, S->Q, x, b) ;	    /* b(0:n-1) = Q*x (permutation) */
	}
    }
    else
    {
	AT = cs_transpose (A, 1) ;	/* Ax=b is underdetermined */
	S = cs_sqr (AT, order, 1) ;	/* ordering and symbolic analysis */
	N = cs_qr (AT, S) ;		/* numeric QR factorization of A' */
	x = cs_calloc (S ? S->m2 : 1, sizeof (double)) ;
	ok = (AT && S && N && x) ;
	if (ok)
	{
	    cs_pvec (m, S->Q, b, x) ;	    /* x(0:m-1) = Q'*b (permutation) */
	    cs_utsolve (N->U, x) ;	    /* x = R'\x */
	    for (k = m-1 ; k >= 0 ; k--)    /* apply Householder refl. to x */
	    {
		cs_happly (N->L, k, N->B [k], x) ;
	    }
	    cs_pvec (n, S->Pinv, x, b) ;    /* b (0:n-1) = P'*x */
	}
    }
    cs_free (x) ;
    cs_sfree (S) ;
    cs_nfree (N) ;
    cs_spfree (AT) ;
    return (ok) ;
}

/* xi [top...n-1] = nodes reachable from graph of L*P' via nodes in B(:,k).
 * xi [n...2n-1] used as workspace */
int cs_reach (cs *L, const cs *B, int k, int *xi, const int *Pinv)
{
    int p, n, top, *Bp, *Bi, *Lp ;
    if (!L || !B || !xi) return (-1) ;
    n = L->n ; Bp = B->p ; Bi = B->i ; Lp = L->p ;
    top = n ;
    for (p = Bp [k] ; p < Bp [k+1] ; p++)
    {
	if (!CS_MARKED (Lp, Bi [p]))	/* start a dfs at unmarked node i */
	{
	    top = cs_dfs (Bi [p], L, top, xi, xi+n, Pinv) ;
	}
    }
    for (p = top ; p < n ; p++) CS_MARK (Lp, xi [p]) ;	/* restore L */
    return (top) ;
}

/* x = x + beta * A(:,j), where x is a dense vector and A(:,j) is sparse */
int cs_scatter (const cs *A, int j, double beta, int *w, double *x, int mark,
    cs *C, int nz)
{
    int i, p, *Ap, *Ai, *Ci ;
    double *Ax ;
    if (!A || !w || !C) return (-1) ;		/* ensure inputs are valid */
    Ap = A->p ; Ai = A->i ; Ax = A->x ; Ci = C->i ;
    for (p = Ap [j] ; p < Ap [j+1] ; p++)
    {
	i = Ai [p] ;				/* A(i,j) is nonzero */
	if (w [i] < mark)
	{
	    w [i] = mark ;			/* i is new entry in column j */
	    Ci [nz++] = i ;			/* add i to pattern of C(:,j) */
	    if (x) x [i] = beta * Ax [p] ;	/* x(i) = beta*A(i,j) */
	}
	else if (x) x [i] += beta * Ax [p] ;	/* i exists in C(:,j) already */
    }
    return (nz) ;
}

/* find the strongly connected components of a square matrix */
csd *cs_scc (cs *A)	/* matrix A temporarily modified, then restored */
{
    int n, i, k, b = 0, top, *xi, *pstack, *P, *R, *Ap, *ATp ;
    cs *AT ;
    csd *D ;
    if (!A) return (NULL) ;
    n = A->n ; Ap = A->p ;
    D = cs_dalloc (n, 0) ;
    AT = cs_transpose (A, 0) ;		    /* AT = A' */
    xi = cs_malloc (2*n, sizeof (int)) ;    /* allocate workspace */
    pstack = xi + n ;
    if (!D || !AT || !xi) return (cs_ddone (D, AT, xi, 0)) ;
    P = D->P ; R = D->R ; ATp = AT->p ;
    top = n ;
    for (i = 0 ; i < n ; i++)	/* first dfs(A) to find finish times (xi) */
    {
	if (!CS_MARKED (Ap,i)) top = cs_dfs (i, A, top, xi, pstack, NULL) ;
    }
    for (i = 0 ; i < n ; i++) CS_MARK (Ap, i) ;	/* restore A; unmark all nodes*/
    top = n ;
    b = n ;
    for (k = 0 ; k < n ; k++)	/* dfs(A') to find strongly connnected comp. */
    {
	i = xi [k] ;		/* get i in reverse order of finish times */
	if (CS_MARKED (ATp,i)) continue ;  /* skip node i if already ordered */
	R [b--] = top ;		/* node i is the start of a component in P */
	top = cs_dfs (i, AT, top, P, pstack, NULL) ;
    }
    R [b] = 0 ;			/* first block starts at zero; shift R up */
    for (k = b ; k <= n ; k++) R [k-b] = R [k] ;
    D->nb = R [n+1] = b = n-b ;	/* b = # of strongly connected components */
    return (cs_ddone (D, AT, xi, 1)) ;
}

/* ordering and symbolic analysis for a Cholesky factorization */
css *cs_schol (const cs *A, int order)
{
    int n, *c, *post, *P ;
    cs *C ;
    css *S ;
    if (!A) return (NULL) ;		    /* check inputs */
    n = A->n ;
    S = cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
    if (!S) return (NULL) ;		    /* out of memory */
    P = cs_amd (A, order) ;		    /* P = amd(A+A'), or natural */
    S->Pinv = cs_pinv (P, n) ;		    /* find inverse permutation */
    cs_free (P) ;
    if (order >= 0 && !S->Pinv) return (cs_sfree (S)) ;
    C = cs_symperm (A, S->Pinv, 0) ;	    /* C = spones(triu(A(P,P))) */
    S->parent = cs_etree (C, 0) ;	    /* find etree of C */
    post = cs_post (n, S->parent) ;	    /* postorder the etree */
    c = cs_counts (C, S->parent, post, 0) ; /* find column counts of chol(C) */
    cs_free (post) ;
    cs_spfree (C) ;
    S->cp = cs_malloc (n+1, sizeof (int)) ; /* find column pointers for L */
    S->unz = S->lnz = cs_cumsum (S->cp, c, n) ;
    cs_free (c) ;
    return ((S->lnz >= 0) ? S : cs_sfree (S)) ;
}

/* solve Lx=b(:,k), leaving pattern in xi[top..n-1], values scattered in x. */
int cs_splsolve (cs *L, const cs *B, int k, int *xi, double *x, const int *Pinv)
{
    int j, jnew, p, px, top, n, *Lp, *Li, *Bp, *Bi ;
    double *Lx, *Bx ;
    if (!L || !B || !xi || !x) return (-1) ;
    Lp = L->p ; Li = L->i ; Lx = L->x ; n = L->n ;
    Bp = B->p ; Bi = B->i ; Bx = B->x ;
    top = cs_reach (L, B, k, xi, Pinv) ;	/* xi[top..n-1]=Reach(B(:,k)) */
    for (p = top ; p < n ; p++) x [xi [p]] = 0 ;/* clear x */
    for (p = Bp [k] ; p < Bp [k+1] ; p++) x [Bi [p]] = Bx [p] ;	/* scatter B */
    for (px = top ; px < n ; px++)
    {
	j = xi [px] ;				/* x(j) is nonzero */
	jnew = Pinv ? (Pinv [j]) : j ;		/* j is column jnew of L */
	if (jnew < 0) continue ;		/* column jnew is empty */
	for (p = Lp [jnew]+1 ; p < Lp [jnew+1] ; p++)
	{
	    x [Li [p]] -= Lx [p] * x [j] ;	/* x(i) -= L(i,j) * x(j) */
	}
    }
    return (top) ;				/* return top of stack */
}

/* compute vnz, Pinv, leftmost, m2 from A and parent */
static int *cs_vcount (const cs *A, const int *parent, int *m2, int *vnz)
{
    int i, k, p, pa, n = A->n, m = A->m, *Ap = A->p, *Ai = A->i ;
    int *Pinv = cs_malloc (2*m+n, sizeof (int)), *leftmost = Pinv + m + n ;
    int *w = cs_malloc (m+3*n, sizeof (int)) ;
    int *next = w, *head = w + m, *tail = w + m + n, *nque = w + m + 2*n ;
    if (!Pinv || !w) return (cs_idone (Pinv, NULL, w, 0)) ;
    for (k = 0 ; k < n ; k++) head [k] = -1 ;	/* queue k is empty */
    for (k = 0 ; k < n ; k++) tail [k] = -1 ;
    for (k = 0 ; k < n ; k++) nque [k] = 0 ;
    for (i = 0 ; i < m ; i++) leftmost [i] = -1 ;
    for (k = n-1 ; k >= 0 ; k--)
    {
	for (p = Ap [k] ; p < Ap [k+1] ; p++)
	{
	    leftmost [Ai [p]] = k ;	    /* leftmost[i] = min(find(A(i,:)))*/
	}
    }
    for (i = m-1 ; i >= 0 ; i--)	    /* scan rows in reverse order */
    {
	Pinv [i] = -1 ;			    /* row i is not yet ordered */
	k = leftmost [i] ;
	if (k == -1) continue ;		    /* row i is empty */
	if (nque [k]++ == 0) tail [k] = i ; /* first row in queue k */
	next [i] = head [k] ;		    /* put i at head of queue k */
	head [k] = i ;
    }
    (*vnz) = 0 ;
    (*m2) = m ;
    for (k = 0 ; k < n ; k++)		    /* find row permutation and nnz(V)*/
    {
	i = head [k] ;			    /* remove row i from queue k */
	(*vnz)++ ;			    /* count V(k,k) as nonzero */
	if (i < 0) i = (*m2)++ ;	    /* add a fictitious row */
	Pinv [i] = k ;			    /* associate row i with V(:,k) */
	if (--nque [k] <= 0) continue ;	    /* skip if V(k+1:m,k) is empty */
	(*vnz) += nque [k] ;		    /* nque [k] = nnz (V(k+1:m,k)) */
	if ((pa = parent [k]) != -1)	    /* move all rows to parent of k */
	{
	    if (nque [pa] == 0) tail [pa] = tail [k] ;
	    next [tail [k]] = head [pa] ;
	    head [pa] = next [i] ;
	    nque [pa] += nque [k] ;
	}
    }
    for (i = 0 ; i < m ; i++) if (Pinv [i] < 0) Pinv [i] = k++ ;
    return (cs_idone (Pinv, NULL, w, 1)) ;
}

/* symbolic analysis for QR or LU */
css *cs_sqr (const cs *A, int order, int qr)
{
    int n, k, ok = 1, *post ;
    css *S ;
    if (!A) return (NULL) ;		    /* check inputs */
    n = A->n ;
    S = cs_calloc (1, sizeof (css)) ;	    /* allocate symbolic analysis */
    if (!S) return (NULL) ;		    /* out of memory */
    S->Q = cs_amd (A, order) ;		    /* fill-reducing ordering */
    if (order >= 0 && !S->Q) return (cs_sfree (S)) ;
    if (qr)				    /* QR symbolic analysis */
    {
	cs *C = (order >= 0) ? cs_permute (A, NULL, S->Q, 0) : ((cs *) A) ;
	S->parent = cs_etree (C, 1) ;	    /* etree of C'*C, where C=A(:,Q) */
	post = cs_post (n, S->parent) ;
	S->cp = cs_counts (C, S->parent, post, 1) ;  /* col counts chol(C'*C) */
	cs_free (post) ;
	ok = C && S->parent && S->cp ;
	if (ok) S->Pinv = cs_vcount (C, S->parent, &(S->m2), &(S->lnz)) ;
	ok = ok && S->Pinv ;
	if (ok) for (S->unz = 0, k = 0 ; k < n ; k++) S->unz += S->cp [k] ;
	if (order >= 0) cs_spfree (C) ;
    }
    else
    {
	S->unz = 4*(A->p [n]) + n ;	    /* for LU factorization only, */
	S->lnz = S->unz ;		    /* guess nnz(L) and nnz(U) */
    }
    return (ok ? S : cs_sfree (S)) ;
}

/* C = A(p,p) where A and C are symmetric the upper part stored, Pinv not P */
cs *cs_symperm (const cs *A, const int *Pinv, int values)
{
    int i, j, p, q, i2, j2, n, *Ap, *Ai, *Cp, *Ci, *w ;
    double *Cx, *Ax ;
    cs *C ;
    if (!A) return (NULL) ;
    n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    C = cs_spalloc (n, n, Ap [n], values && (Ax != NULL), 0) ;
    w = cs_calloc (n, sizeof (int)) ;
    if (!C || !w) return (cs_done (C, w, NULL, 0)) ;	/* out of memory */
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (j = 0 ; j < n ; j++)		/* count entries in each column of C */
    {
	j2 = Pinv ? Pinv [j] : j ;	/* column j of A is column j2 of C */
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    if (i > j) continue ;	/* skip lower triangular part of A */
	    i2 = Pinv ? Pinv [i] : i ;	/* row i of A is row i2 of C */
	    w [CS_MAX (i2, j2)]++ ;	/* column count of C */
	}
    }
    cs_cumsum (Cp, w, n) ;		/* compute column pointers of C */
    for (j = 0 ; j < n ; j++)
    {
	j2 = Pinv ? Pinv [j] : j ;	/* column j of A is column j2 of C */
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    i = Ai [p] ;
	    if (i > j) continue ;	/* skip lower triangular part of A*/
	    i2 = Pinv ? Pinv [i] : i ;	/* row i of A is row i2 of C */
	    Ci [q = w [CS_MAX (i2, j2)]++] = CS_MIN (i2, j2) ;
	    if (Cx) Cx [q] = Ax [p] ;
	}
    }
    return (cs_done (C, w, NULL, 1)) ;	/* success; free workspace, return C */
}

/* depth-first search and postorder of a tree rooted at node j */
int cs_tdfs (int j, int k, int *head, const int *next, int *post, int *stack)
{
    int i, p, top = 0 ;
    if (!head || !next || !post || !stack) return (-1) ;    /* check inputs */
    stack [0] = j ;		    /* place j on the stack */
    while (top >= 0)		    /* while (stack is not empty) */
    {
	p = stack [top] ;	    /* p = top of stack */
	i = head [p] ;		    /* i = youngest child of p */
	if (i == -1)
	{
	    top-- ;		    /* p has no unordered children left */
	    post [k++] = p ;	    /* node p is the kth postordered node */
	}
	else
	{
	    head [p] = next [i] ;   /* remove i from children of p */
	    stack [++top] = i ;	    /* start dfs on child node i */
	}
    }
    return (k) ;
}

/* C = A' */
cs *cs_transpose (const cs *A, int values)
{
    int p, q, j, *Cp, *Ci, n, m, *Ap, *Ai, *w ;
    double *Cx, *Ax ;
    cs *C ;
    if (!A) return (NULL) ;
    m = A->m ; n = A->n ; Ap = A->p ; Ai = A->i ; Ax = A->x ;
    C = cs_spalloc (n, m, Ap [n], values && Ax, 0) ;	   /* allocate result */
    w = cs_calloc (m, sizeof (int)) ;
    if (!C || !w) return (cs_done (C, w, NULL, 0)) ;	   /* out of memory */
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (p = 0 ; p < Ap [n] ; p++) w [Ai [p]]++ ;	   /* row counts */
    cs_cumsum (Cp, w, m) ;				   /* row pointers */
    for (j = 0 ; j < n ; j++)
    {
	for (p = Ap [j] ; p < Ap [j+1] ; p++)
	{
	    Ci [q = w [Ai [p]]++] = j ;	/* place A(i,j) as entry C(j,i) */
	    if (Cx) Cx [q] = Ax [p] ;
	}
    }
    return (cs_done (C, w, NULL, 1)) ;	/* success; free w and return C */
}

/* C = compressed-column form of a triplet matrix T */
cs *cs_triplet (const cs *T)
{
    int m, n, nz, p, k, *Cp, *Ci, *w, *Ti, *Tj ;
    double *Cx, *Tx ;
    cs *C ;
    if (!T) return (NULL) ;				/* check inputs */
    m = T->m ; n = T->n ; Ti = T->i ; Tj = T->p ; Tx = T->x ; nz = T->nz ;
    C = cs_spalloc (m, n, nz, Tx != NULL, 0) ;		/* allocate result */
    w = cs_calloc (n, sizeof (int)) ;			/* get workspace */
    if (!C || !w) return (cs_done (C, w, NULL, 0)) ;	/* out of memory */
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    for (k = 0 ; k < nz ; k++) w [Tj [k]]++ ;		/* column counts */
    cs_cumsum (Cp, w, n) ;				/* column pointers */
    for (k = 0 ; k < nz ; k++)
    {
	Ci [p = w [Tj [k]]++] = Ti [k] ;    /* A(i,j) is the pth entry in C */
	if (Cx) Cx [p] = Tx [k] ;
    }
    return (cs_done (C, w, NULL, 1)) ;	    /* success; free w and return C */
}

/* sparse Cholesky update/downdate, L*L' + sigma*w*w' (sigma = +1 or -1) */
int cs_updown (cs *L, int sigma, const cs *C, const int *parent)
{
    int p, f, j, n, *Lp, *Li, *Cp, *Ci ;
    double *Lx, *Cx, alpha, beta = 1, delta, gamma, w1, w2, *w,  beta2 = 1 ;
    if (!L || !C || !parent) return (0) ;
    Lp = L->p ; Li = L->i ; Lx = L->x ; n = L->n ;
    Cp = C->p ; Ci = C->i ; Cx = C->x ;
    if ((p = Cp [0]) >= Cp [1]) return (1) ;	    /* return if C empty */
    w = cs_malloc (n, sizeof (double)) ;
    if (!w) return (0) ;
    f = Ci [p] ;
    for ( ; p < Cp [1] ; p++) f = CS_MIN (f, Ci [p]) ;	/* f = min (find (C)) */
    for (j = f ; j != -1 ; j = parent [j]) w [j] = 0 ;	/* clear workspace w */
    for (p = Cp [0] ; p < Cp [1] ; p++) w [Ci [p]] = Cx [p] ; /* w = C */
    for (j = f ; j != -1 ; j = parent [j])	    /* walk path f up to root */
    {
	p = Lp [j] ;
	alpha = w [j] / Lx [p] ;		    /* alpha = w(j) / L(j,j) */
	beta2 = beta*beta + sigma*alpha*alpha ;
	if (beta2 <= 0) break ;			    /* not positive definite */
	beta2 = sqrt (beta2) ;
	delta = (sigma > 0) ? (beta / beta2) : (beta2 / beta) ;
	gamma = sigma * alpha / (beta2 * beta) ;
	Lx [p] = delta * Lx [p] + ((sigma > 0) ? (gamma * w [j]) : 0) ;
	beta = beta2 ;
	for (p++ ; p < Lp [j+1] ; p++)
	{
	    w1 = w [Li [p]] ;
	    w [Li [p]] = w2 = w1 - alpha * Lx [p] ;
	    Lx [p] = delta * Lx [p] + gamma * ((sigma > 0) ? w1 : w2) ;
	}
    }
    cs_free (w) ;
    return (beta2 > 0) ;
}

/* solve Ux=b where x and b are dense.  x=b on input, solution on output. */
int cs_usolve (const cs *U, double *x)
{
    int p, j, n, *Up, *Ui ;
    double *Ux ;
    if (!U || !x) return (0) ;				    /* check inputs */
    n = U->n ; Up = U->p ; Ui = U->i ; Ux = U->x ;
    for (j = n-1 ; j >= 0 ; j--)
    {
	x [j] /= Ux [Up [j+1]-1] ;
	for (p = Up [j] ; p < Up [j+1]-1 ; p++)
	{
	    x [Ui [p]] -= Ux [p] * x [j] ;
	}
    }
    return (1) ;
}

/* allocate a sparse matrix (triplet form or compressed-column form) */
cs *cs_spalloc (int m, int n, int nzmax, int values, int triplet)
{
    cs *A = cs_calloc (1, sizeof (cs)) ;    /* allocate the cs struct */
    if (!A) return (NULL) ;		    /* out of memory */
    A->m = m ;				    /* define dimensions and nzmax */
    A->n = n ;
    A->nzmax = nzmax = CS_MAX (nzmax, 1) ;
    A->nz = triplet ? 0 : -1 ;		    /* allocate triplet or comp.col */
    A->p = cs_malloc (triplet ? nzmax : n+1, sizeof (int)) ;
    A->i = cs_malloc (nzmax, sizeof (int)) ;
    A->x = values ? cs_malloc (nzmax, sizeof (double)) : NULL ;
    return ((!A->p || !A->i || (values && !A->x)) ? cs_spfree (A) : A) ;
}

/* change the max # of entries sparse matrix */
int cs_sprealloc (cs *A, int nzmax)
{
    int ok, oki, okj = 1, okx = 1 ;
    if (!A) return (0) ;
    nzmax = (nzmax <= 0) ? (A->p [A->n]) : nzmax ;
    A->i = cs_realloc (A->i, nzmax, sizeof (int), &oki) ;
    if (A->nz >= 0) A->p = cs_realloc (A->p, nzmax, sizeof (int), &okj) ;
    if (A->x) A->x = cs_realloc (A->x, nzmax, sizeof (double), &okx) ;
    ok = (oki && okj && okx) ;
    if (ok) A->nzmax = nzmax ;
    return (ok) ;
}

/* free a sparse matrix */
cs *cs_spfree (cs *A)
{
    if (!A) return (NULL) ;	/* do nothing if A already NULL */
    cs_free (A->p) ;
    cs_free (A->i) ;
    cs_free (A->x) ;
    return (cs_free (A)) ;	/* free the cs struct and return NULL */
}

/* free a numeric factorization */
csn *cs_nfree (csn *N)
{
    if (!N) return (NULL) ;	/* do nothing if N already NULL */
    cs_spfree (N->L) ;
    cs_spfree (N->U) ;
    cs_free (N->Pinv) ;
    cs_free (N->B) ;
    return (cs_free (N)) ;	/* free the csn struct and return NULL */
}

/* free a symbolic factorization */
css *cs_sfree (css *S)
{
    if (!S) return (NULL) ;	/* do nothing if S already NULL */
    cs_free (S->Pinv) ;
    cs_free (S->Q) ;
    cs_free (S->parent) ;
    cs_free (S->cp) ;
    return (cs_free (S)) ;	/* free the css struct and return NULL */
}

/* allocate a cs_dmperm or cs_scc result */
csd *cs_dalloc (int m, int n)
{
    csd *D ;
    D = cs_calloc (1, sizeof (csd)) ;
    if (!D) return (NULL) ;
    D->P = cs_malloc (m, sizeof (int)) ;
    D->R = cs_malloc (m+6, sizeof (int)) ;
    D->Q = cs_malloc (n, sizeof (int)) ;
    D->S = cs_malloc (n+6, sizeof (int)) ;
    return ((!D->P || !D->R || !D->Q || !D->S) ? cs_dfree (D) : D) ;
}

/* free a cs_dmperm or cs_scc result */
csd *cs_dfree (csd *D)
{
    if (!D) return (NULL) ;	/* do nothing if D already NULL */
    cs_free (D->P) ;
    cs_free (D->Q) ;
    cs_free (D->R) ;
    cs_free (D->S) ;
    return (cs_free (D)) ;
}

/* free workspace and return a sparse matrix result */
cs *cs_done (cs *C, void *w, void *x, int ok)
{
    cs_free (w) ;			/* free workspace */
    cs_free (x) ;
    return (ok ? C : cs_spfree (C)) ;	/* return result if OK, else free it */
}

/* free workspace and return int array result */
int *cs_idone (int *p, cs *C, void *w, int ok)
{
    cs_spfree (C) ;			/* free temporary matrix */
    cs_free (w) ;			/* free workspace */
    return (ok ? p : cs_free (p)) ;	/* return result if OK, else free it */
}

/* free workspace and return a numeric factorization (Cholesky, LU, or QR) */
csn *cs_ndone (csn *N, cs *C, void *w, void *x, int ok)
{
    cs_spfree (C) ;			/* free temporary matrix */
    cs_free (w) ;			/* free workspace */
    cs_free (x) ;
    return (ok ? N : cs_nfree (N)) ;	/* return result if OK, else free it */
}

/* free workspace and return a csd result */
csd *cs_ddone (csd *D, cs *C, void *w, int ok)
{
    cs_spfree (C) ;			/* free temporary matrix */
    cs_free (w) ;			/* free workspace */
    return (ok ? D : cs_dfree (D)) ;	/* return result if OK, else free it */
}

/* solve U'x=b where x and b are dense.  x=b on input, solution on output. */
int cs_utsolve (const cs *U, double *x)
{
    int p, j, n, *Up, *Ui ;
    double *Ux ;
    if (!U || !x) return (0) ;				    /* check inputs */
    n = U->n ; Up = U->p ; Ui = U->i ; Ux = U->x ;
    for (j = 0 ; j < n ; j++)
    {
	for (p = Up [j] ; p < Up [j+1]-1 ; p++)
	{
	    x [j] -= Ux [p] * x [Ui [p]] ;
	}
	x [j] /= Ux [p] ;
    }
    return (1) ;
}
