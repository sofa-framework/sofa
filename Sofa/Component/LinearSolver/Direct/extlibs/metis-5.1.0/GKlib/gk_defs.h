/*!
\file gk_defs.h
\brief This file contains various constants definitions

\date   Started 3/27/2007
\author George
\version\verbatim $Id: gk_defs.h 22039 2018-05-26 16:34:48Z karypis $ \endverbatim
*/

#ifndef _GK_DEFS_H_
#define _GK_DEFS_H_


#define LTERM                   (void **) 0     /* List terminator for GKfree() */

/* mopt_t types */
#define GK_MOPT_MARK            1
#define GK_MOPT_CORE            2
#define GK_MOPT_HEAP            3

#define HTABLE_EMPTY            -1
#define HTABLE_DELETED          -2
#define HTABLE_FIRST             1
#define HTABLE_NEXT              2

/* pdb corruption bit switches */
#define CRP_ALTLOCS    1
#define CRP_MISSINGCA  2
#define CRP_MISSINGBB  4
#define CRP_MULTICHAIN 8
#define CRP_MULTICA    16
#define CRP_MULTIBB    32

#define MAXLINELEN 300000

/* GKlib signals to standard signal mapping */
#define SIGMEM  SIGABRT
#define SIGERR  SIGTERM


/* CSR-related defines */
#define GK_CSR_ROW      1
#define GK_CSR_COL      2
#define GK_CSR_ROWCOL   3

#define GK_CSR_MAXTF    1
#define GK_CSR_SQRT     2
#define GK_CSR_POW25    3
#define GK_CSR_POW65    4
#define GK_CSR_POW75    5
#define GK_CSR_POW85    6
#define GK_CSR_LOG      7
#define GK_CSR_IDF      8
#define GK_CSR_IDF2     9
#define GK_CSR_MAXTF2   10

#define GK_CSR_DOTP     1
#define GK_CSR_COS      2
#define GK_CSR_JAC      3
#define GK_CSR_MIN      4
#define GK_CSR_AMIN     5

#define GK_CSR_FMT_AUTO         2
#define GK_CSR_FMT_CLUTO        1
#define GK_CSR_FMT_CSR          2
#define GK_CSR_FMT_METIS        3
#define GK_CSR_FMT_BINROW       4
#define GK_CSR_FMT_BINCOL       5
#define GK_CSR_FMT_IJV          6
#define GK_CSR_FMT_BIJV         7

#define GK_CSR_SYM_SUM          1
#define GK_CSR_SYM_MIN          2
#define GK_CSR_SYM_MAX          3
#define GK_CSR_SYM_AVG          4


#define GK_GRAPH_FMT_METIS      1
#define GK_GRAPH_FMT_IJV        2
#define GK_GRAPH_FMT_HIJV       3

#define GK_GRAPH_SYM_SUM        1
#define GK_GRAPH_SYM_MIN        2
#define GK_GRAPH_SYM_MAX        3
#define GK_GRAPH_SYM_AVG        4

#endif
