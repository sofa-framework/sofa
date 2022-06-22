/*!
\file  
\brief The top-level routines for  multilevel k-way partitioning that minimizes
       the edge cut.

\date   Started 7/28/1997
\author George  
\author Copyright 1997-2011, Regents of the University of Minnesota 
\version\verbatim $Id: kmetis.c 20398 2016-11-22 17:17:12Z karypis $ \endverbatim
*/

#include "metislib.h"


/*************************************************************************/
/*! This function is the entry point for MCKMETIS */
/*************************************************************************/
int METIS_PartGraphKway(idx_t *nvtxs, idx_t *ncon, idx_t *xadj, idx_t *adjncy, 
          idx_t *vwgt, idx_t *vsize, idx_t *adjwgt, idx_t *nparts, 
          real_t *tpwgts, real_t *ubvec, idx_t *options, idx_t *objval, 
          idx_t *part)
{
  int sigrval=0, renumber=0;
  graph_t *graph;
  ctrl_t *ctrl;

  /* set up malloc cleaning code and signal catchers */
  if (!gk_malloc_init()) 
    return METIS_ERROR_MEMORY;

  gk_sigtrap();

  if ((sigrval = gk_sigcatch()) != 0)
    goto SIGTHROW;

  /* set up the run parameters */
  ctrl = SetupCtrl(METIS_OP_KMETIS, options, *ncon, *nparts, tpwgts, ubvec);
  if (!ctrl) {
    gk_siguntrap();
    return METIS_ERROR_INPUT;
  }

  /* if required, change the numbering to 0 */
  if (ctrl->numflag == 1) {
    Change2CNumbering(*nvtxs, xadj, adjncy);
    renumber = 1;
  }

  /* set up the graph */
  graph = SetupGraph(ctrl, *nvtxs, *ncon, xadj, adjncy, vwgt, vsize, adjwgt);

  /* set up multipliers for making balance computations easier */
  SetupKWayBalMultipliers(ctrl, graph);

  /* set various run parameters that depend on the graph */
  ctrl->CoarsenTo = gk_max((*nvtxs)/(40*gk_log2(*nparts)), 30*(*nparts));
  ctrl->nIparts   = (ctrl->nIparts != -1 ? ctrl->nIparts : (ctrl->CoarsenTo == 30*(*nparts) ? 4 : 5));

  /* take care contiguity requests for disconnected graphs */
  if (ctrl->contig && !IsConnected(graph, 0)) 
    gk_errexit(SIGERR, "METIS Error: A contiguous partition is requested for a non-contiguous input graph.\n");
    
  /* allocate workspace memory */  
  AllocateWorkSpace(ctrl, graph);

  /* start the partitioning */
  IFSET(ctrl->dbglvl, METIS_DBG_TIME, InitTimers(ctrl));
  IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_startcputimer(ctrl->TotalTmr));

  if (ctrl->dbglvl&512) {
    *objval = BlockKWayPartitioning(ctrl, graph, part);
  }
  else {
    *objval = MlevelKWayPartitioning(ctrl, graph, part);
  }

  IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_stopcputimer(ctrl->TotalTmr));
  IFSET(ctrl->dbglvl, METIS_DBG_TIME, PrintTimers(ctrl));

  /* clean up */
  FreeCtrl(&ctrl);

SIGTHROW:
  /* if required, change the numbering back to 1 */
  if (renumber)
    Change2FNumbering(*nvtxs, xadj, adjncy, part);

  gk_siguntrap();
  gk_malloc_cleanup(0);

  return metis_rcode(sigrval);
}


/*************************************************************************/
/*! This function computes a k-way partitioning of a graph that minimizes
    the specified objective function.

    \param ctrl is the control structure
    \param graph is the graph to be partitioned
    \param part is the vector that on return will store the partitioning

    \returns the objective value of the partitoning. The partitioning 
             itself is stored in the part vector.
*/
/*************************************************************************/
idx_t MlevelKWayPartitioning(ctrl_t *ctrl, graph_t *graph, idx_t *part)
{
  idx_t i, j, objval=0, curobj=0, bestobj=0;
  real_t curbal=0.0, bestbal=0.0;
  graph_t *cgraph;
  int status;


  for (i=0; i<ctrl->ncuts; i++) {
    cgraph = CoarsenGraph(ctrl, graph);

    IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_startcputimer(ctrl->InitPartTmr));
    AllocateKWayPartitionMemory(ctrl, cgraph);

    /* Release the work space */
    FreeWorkSpace(ctrl);

    /* Compute the initial partitioning */
    InitKWayPartitioning(ctrl, cgraph);

    /* Re-allocate the work space */
    AllocateWorkSpace(ctrl, graph);
    AllocateRefinementWorkSpace(ctrl, graph->nedges, 2*cgraph->nedges);

    IFSET(ctrl->dbglvl, METIS_DBG_TIME, gk_stopcputimer(ctrl->InitPartTmr));
    IFSET(ctrl->dbglvl, METIS_DBG_IPART, 
        printf("Initial %"PRIDX"-way partitioning cut: %"PRIDX"\n", ctrl->nparts, objval));

    RefineKWay(ctrl, graph, cgraph);

    switch (ctrl->objtype) {
      case METIS_OBJTYPE_CUT:
        curobj = graph->mincut;
        break;

      case METIS_OBJTYPE_VOL:
        curobj = graph->minvol;
        break;

      default:
        gk_errexit(SIGERR, "Unknown objtype: %d\n", ctrl->objtype);
    }

    curbal = ComputeLoadImbalanceDiff(graph, ctrl->nparts, ctrl->pijbm, ctrl->ubfactors);

    if (i == 0 
        || (curbal <= 0.0005 && bestobj > curobj)
        || (bestbal > 0.0005 && curbal < bestbal)) {
      icopy(graph->nvtxs, graph->where, part);
      bestobj = curobj;
      bestbal = curbal;
    }

    FreeRData(graph);

    if (bestobj == 0)
      break;
  }

  FreeGraph(&graph);

  return bestobj;
}


/*************************************************************************/
/*! This function computes the initial k-way partitioning using PMETIS 
*/
/*************************************************************************/
void InitKWayPartitioning(ctrl_t *ctrl, graph_t *graph)
{
  idx_t i, ntrials, options[METIS_NOPTIONS], curobj=0, bestobj=0;
  idx_t *bestwhere=NULL;
  real_t *ubvec=NULL;
  int status;

  METIS_SetDefaultOptions(options);
  //options[METIS_OPTION_NITER]     = 10;
  options[METIS_OPTION_NITER]     = ctrl->niter;
  options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_NO2HOP]    = ctrl->no2hop;
  options[METIS_OPTION_ONDISK]    = ctrl->ondisk;
  options[METIS_OPTION_DROPEDGES] = ctrl->dropedges;
  //options[METIS_OPTION_DBGLVL]    = ctrl->dbglvl;

  ubvec = rmalloc(graph->ncon, "InitKWayPartitioning: ubvec");
  for (i=0; i<graph->ncon; i++) 
    ubvec[i] = (real_t)pow(ctrl->ubfactors[i], 1.0/log(ctrl->nparts));


  switch (ctrl->objtype) {
    case METIS_OBJTYPE_CUT:
    case METIS_OBJTYPE_VOL:
      options[METIS_OPTION_NCUTS] = ctrl->nIparts;
      status = METIS_PartGraphRecursive(&graph->nvtxs, &graph->ncon, 
                   graph->xadj, graph->adjncy, graph->vwgt, graph->vsize, 
                   graph->adjwgt, &ctrl->nparts, ctrl->tpwgts, ubvec, 
                   options, &curobj, graph->where);

      if (status != METIS_OK)
        gk_errexit(SIGERR, "Failed during initial partitioning\n");

      break;

#ifdef XXX /* This does not seem to help */
    case METIS_OBJTYPE_VOL:
      bestwhere = imalloc(graph->nvtxs, "InitKWayPartitioning: bestwhere");
      options[METIS_OPTION_NCUTS] = 2;

      ntrials = (ctrl->nIparts+1)/2;
      for (i=0; i<ntrials; i++) {
        status = METIS_PartGraphRecursive(&graph->nvtxs, &graph->ncon, 
                     graph->xadj, graph->adjncy, graph->vwgt, graph->vsize, 
                     graph->adjwgt, &ctrl->nparts, ctrl->tpwgts, ubvec, 
                     options, &curobj, graph->where);
        if (status != METIS_OK)
          gk_errexit(SIGERR, "Failed during initial partitioning\n");

        curobj = ComputeVolume(graph, graph->where);

        if (i == 0 || bestobj > curobj) {
          bestobj = curobj;
          if (i < ntrials-1)
            icopy(graph->nvtxs, graph->where, bestwhere);
        }

        if (bestobj == 0)
          break;
      }
      if (bestobj != curobj)
        icopy(graph->nvtxs, bestwhere, graph->where);

      break;
#endif

    default:
      gk_errexit(SIGERR, "Unknown objtype: %d\n", ctrl->objtype);
  }

  gk_free((void **)&ubvec, &bestwhere, LTERM);

}


/*************************************************************************/
/*! This function computes a k-way partitioning of a graph that minimizes
    the specified objective function.

    \param ctrl is the control structure
    \param graph is the graph to be partitioned
    \param part is the vector that on return will store the partitioning

    \returns the objective value of the partitoning. The partitioning 
             itself is stored in the part vector.
*/
/*************************************************************************/
idx_t BlockKWayPartitioning(ctrl_t *ctrl, graph_t *graph, idx_t *part)
{
  idx_t i, ii, j, nvtxs, objval=0;
  idx_t *vwgt;
  idx_t nparts, mynparts;
  idx_t *fpwgts, *cpwgts, *fpart, *perm;
  ipq_t *queue;

  WCOREPUSH;

  nvtxs = graph->nvtxs;
  vwgt  = graph->vwgt;

  nparts = ctrl->nparts;

  mynparts = gk_min(100*nparts, sqrt(nvtxs));

  for (i=0; i<nvtxs; i++)
    part[i] = i%nparts;
  irandArrayPermute(nvtxs, part, 4*nvtxs, 0);
  printf("Random cut: %d\n", (int)ComputeCut(graph, part));

  /* create the initial multi-section */
  mynparts = GrowMultisection(ctrl, graph, mynparts, part);

  /* balance using label-propagation and refine using a randomized greedy strategy */
  BalanceAndRefineLP(ctrl, graph, mynparts, part);

  /* determine the size of the fine partitions */
  fpwgts = iset(mynparts, 0, iwspacemalloc(ctrl, mynparts));
  for (i=0; i<nvtxs; i++)
    fpwgts[part[i]] += vwgt[i];

  /* create and initialize the queue that will determine
     where to put the next one */
  cpwgts = iset(nparts, 0, iwspacemalloc(ctrl, nparts));
  queue = ipqCreate(nparts);
  for (i=0; i<nparts; i++)
    ipqInsert(queue, i, 0);

  /* assign the fine partitions into the coarse partitions */
  fpart = iwspacemalloc(ctrl, mynparts);
  perm  = iwspacemalloc(ctrl, mynparts);
  irandArrayPermute(mynparts, perm, mynparts, 1);
  for (ii=0; ii<mynparts; ii++) {
    i = perm[ii];
    j = ipqSeeTopVal(queue);
    fpart[i] = j;
    cpwgts[j] += fpwgts[i];
    ipqUpdate(queue, j, -cpwgts[j]);
  }
  ipqDestroy(queue);

  for (i=0; i<nparts; i++) 
    printf("cpwgts[%d] = %d\n", (int)i, (int)cpwgts[i]);

  for (i=0; i<nvtxs; i++)
    part[i] = fpart[part[i]];

  WCOREPOP;

  return ComputeCut(graph, part);
}


/*************************************************************************/
/*! This function takes a graph and produces a bisection by using a region
    growing algorithm. The resulting bisection is refined using FM.
    The resulting partition is returned in graph->where.
*/
/*************************************************************************/
idx_t GrowMultisection(ctrl_t *ctrl, graph_t *graph, idx_t nparts, idx_t *where)
{
  idx_t i, j, k, l, nvtxs, nleft, first, last; 
  idx_t *xadj, *vwgt, *adjncy;
  idx_t *queue;
  idx_t tvwgt, maxpwgt, *pwgts;

  WCOREPUSH;

  nvtxs  = graph->nvtxs;
  xadj   = graph->xadj;
  vwgt   = graph->xadj;
  adjncy = graph->adjncy;

  queue = iwspacemalloc(ctrl, nvtxs);


  /* Select the seeds for the nparts-way BFS */
  for (nleft=0, i=0; i<nvtxs; i++) {
    if (xadj[i+1]-xadj[i] > 1) /* a seed's degree should be > 1 */
      where[nleft++] = i;
  }
  nparts = gk_min(nparts, nleft);
  for (i=0; i<nparts; i++) {
    j = irandInRange(nleft);
    queue[i] = where[j];
    where[j] = --nleft;
  }

  pwgts   = iset(nparts, 0, iwspacemalloc(ctrl, nparts));
  tvwgt   = isum(nvtxs, vwgt, 1);
  maxpwgt = (1.5*tvwgt)/nparts;

  iset(nvtxs, -1, where);
  for (i=0; i<nparts; i++) { 
    where[queue[i]] = i;
    pwgts[i] = vwgt[queue[i]];
  }

  first = 0; 
  last  = nparts;
  nleft = nvtxs-nparts;


  /* Start the BFS from queue to get a partition */
  while (first < last) { 
    i = queue[first++];
    l = where[i];
    if (pwgts[l] > maxpwgt)
      continue;

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (where[k] == -1) {
        if (pwgts[l]+vwgt[k] > maxpwgt)
          break;
        pwgts[l] += vwgt[k];
        where[k] = l;
        queue[last++] = k;
        nleft--;
      }
    }
  }
  
  /* Assign the unassigned vertices randomly to the nparts partitions */
  if (nleft > 0) { 
    for (i=0; i<nvtxs; i++) {
      if (where[i] == -1)
        where[i] = irandInRange(nparts);
    }
  }

  WCOREPOP;

  return nparts;
}


/*************************************************************************/
/*! This function balances the partitioning using label propagation. 
*/
/*************************************************************************/
void BalanceAndRefineLP(ctrl_t *ctrl, graph_t *graph, idx_t nparts, idx_t *where)
{
  idx_t ii, i, j, k, u, v, nvtxs, iter; 
  idx_t *xadj, *vwgt, *adjncy, *adjwgt;
  idx_t tvwgt, *pwgts, maxpwgt, minpwgt;
  idx_t *perm;
  idx_t from, to, nmoves, nnbrs, *nbrids, *nbrwgts, *nbrmrks;
  real_t ubfactor;

  WCOREPUSH;

  nvtxs  = graph->nvtxs;
  xadj   = graph->xadj;
  vwgt   = graph->vwgt;
  adjncy = graph->adjncy;
  adjwgt = graph->adjwgt;

  pwgts    = iset(nparts, 0, iwspacemalloc(ctrl, nparts));

  ubfactor = I2RUBFACTOR(ctrl->ufactor);
  tvwgt    = isum(nvtxs, vwgt, 1);
  maxpwgt  = (ubfactor*tvwgt)/nparts;
  minpwgt  = (1.0*tvwgt)/(ubfactor*nparts);

  for (i=0; i<nvtxs; i++)
    pwgts[where[i]] += vwgt[i];

  /* for randomly visiting the vertices */
  perm = iincset(nvtxs, 0, iwspacemalloc(ctrl, nvtxs));

  /* for keeping track of adjancent partitions */
  nbrids  = iwspacemalloc(ctrl, nparts);
  nbrwgts = iset(nparts, 0, iwspacemalloc(ctrl, nparts));
  nbrmrks = iset(nparts, -1, iwspacemalloc(ctrl, nparts));

  /* perform a fixed number of balancing LP iterations */
  if (ctrl->dbglvl&METIS_DBG_REFINE) 
    printf("BLP: nparts: %"PRIDX", min-max: [%"PRIDX", %"PRIDX"], bal: %7.4"PRREAL", cut: %9"PRIDX"\n",
        nparts, minpwgt, maxpwgt, 1.0*imax(nparts, pwgts, 1)*nparts/tvwgt, ComputeCut(graph, where));
  for (iter=0; iter<ctrl->niter; iter++) {
    if (imax(nparts, pwgts, 1)*nparts < ubfactor*tvwgt)
      break;

    irandArrayPermute(nvtxs, perm, nvtxs/8, 1);
    nmoves = 0;

    for (ii=0; ii<nvtxs; ii++) {
      u = perm[ii];

      from = where[u];
      if (pwgts[from] - vwgt[u] < minpwgt)
        continue;

      nnbrs = 0;
      for (j=xadj[u]; j<xadj[u+1]; j++) {
        v  = adjncy[j];
        to = where[v];

        if (pwgts[to] + vwgt[u] > maxpwgt)
          continue; /* skip if 'to' is overweight */

        if ((k = nbrmrks[to]) == -1) {
          nbrmrks[to] = k = nnbrs++;
          nbrids[k] = to;
        }
        nbrwgts[k] += xadj[v+1]-xadj[v];
      }
      if (nnbrs == 0)
        continue;

      to = nbrids[iargmax(nnbrs, nbrwgts, 1)];
      if (from != to) {
        where[u] = to;
        INC_DEC(pwgts[to], pwgts[from], vwgt[u]);
        nmoves++;
      }

      for (k=0; k<nnbrs; k++) {
        nbrmrks[nbrids[k]] = -1;
        nbrwgts[k] = 0;
      }

    }

    if (ctrl->dbglvl&METIS_DBG_REFINE) 
      printf("     nmoves: %8"PRIDX", bal: %7.4"PRREAL", cut: %9"PRIDX"\n",
          nmoves, 1.0*imax(nparts, pwgts, 1)*nparts/tvwgt, ComputeCut(graph, where));

    if (nmoves == 0)
      break;
  }

  /* perform a fixed number of refinement LP iterations */
  if (ctrl->dbglvl&METIS_DBG_REFINE) 
    printf("RLP: nparts: %"PRIDX", min-max: [%"PRIDX", %"PRIDX"], bal: %7.4"PRREAL", cut: %9"PRIDX"\n",
        nparts, minpwgt, maxpwgt, 1.0*imax(nparts, pwgts, 1)*nparts/tvwgt, ComputeCut(graph, where));
  for (iter=0; iter<ctrl->niter; iter++) {
    irandArrayPermute(nvtxs, perm, nvtxs/8, 1);
    nmoves = 0;

    for (ii=0; ii<nvtxs; ii++) {
      u = perm[ii];

      from = where[u];
      if (pwgts[from] - vwgt[u] < minpwgt)
        continue;

      nnbrs = 0;
      for (j=xadj[u]; j<xadj[u+1]; j++) {
        v  = adjncy[j];
        to = where[v];

        if (to != from && pwgts[to] + vwgt[u] > maxpwgt)
          continue; /* skip if 'to' is overweight */

        if ((k = nbrmrks[to]) == -1) {
          nbrmrks[to] = k = nnbrs++;
          nbrids[k] = to;
        }
        nbrwgts[k] += adjwgt[j];
      }
      if (nnbrs == 0)
        continue;

      to = nbrids[iargmax(nnbrs, nbrwgts, 1)];
      if (from != to) {
        where[u] = to;
        INC_DEC(pwgts[to], pwgts[from], vwgt[u]);
        nmoves++;
      }

      for (k=0; k<nnbrs; k++) {
        nbrmrks[nbrids[k]] = -1;
        nbrwgts[k] = 0;
      }

    }

    if (ctrl->dbglvl&METIS_DBG_REFINE) 
      printf("     nmoves: %8"PRIDX", bal: %7.4"PRREAL", cut: %9"PRIDX"\n",
          nmoves, 1.0*imax(nparts, pwgts, 1)*nparts/tvwgt, ComputeCut(graph, where));

    if (nmoves == 0)
      break;
  }

  WCOREPOP;
}
