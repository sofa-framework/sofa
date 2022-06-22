/*!
\file 
\brief Functions dealing with simulating cache behavior for performance
       modeling and analysis;

\date Started 4/13/18
\author George
\author Copyright 1997-2011, Regents of the University of Minnesota 
\version $Id: cache.c 21991 2018-04-16 03:08:12Z karypis $
*/

#include <GKlib.h>


/*************************************************************************/
/*! This function creates a cache 
 */
/*************************************************************************/
gk_cache_t *gk_cacheCreate(uint32_t nway, uint32_t lnbits, size_t cnbits)
{
  gk_cache_t *cache;

  cache = (gk_cache_t *)gk_malloc(sizeof(gk_cache_t), "gk_cacheCreate: cache");
  memset(cache, 0, sizeof(gk_cache_t));

  cache->nway   = nway;
  cache->lnbits = lnbits;
  cache->cnbits = cnbits;
  cache->csize  = 1<<cnbits;
  cache->cmask  = cache->csize-1;

  cache->latimes = gk_ui64smalloc(cache->csize*nway, 0, "gk_cacheCreate: latimes");
  cache->clines  = gk_zusmalloc(cache->csize*nway, 0, "gk_cacheCreate: clines");

  return cache;
}


/*************************************************************************/
/*! This function resets a cache 
 */
/*************************************************************************/
void gk_cacheReset(gk_cache_t *cache)
{
  cache->nhits   = 0;
  cache->nmisses = 0;

  gk_ui64set(cache->csize*cache->nway, 0, cache->latimes);
  gk_zuset(cache->csize*cache->nway, 0, cache->clines);

  return;
}


/*************************************************************************/
/*! This function destroys a cache.
 */
/*************************************************************************/
void gk_cacheDestroy(gk_cache_t **r_cache)
{
  gk_cache_t *cache = *r_cache;

  if (cache == NULL)
    return;

  gk_free((void **)&cache->clines, &cache->latimes, &cache, LTERM);

  *r_cache = NULL;
}


/*************************************************************************/
/*! This function simulates a load(ptr) operation.
 */
/*************************************************************************/
int gk_cacheLoad(gk_cache_t *cache, size_t addr)
{
  uint32_t i, nway=cache->nway;
  size_t lru=0;

  //printf("%16"PRIx64" ", (uint64_t)addr);
  addr = addr>>(cache->lnbits);
  //printf("%16"PRIx64" %16"PRIx64" %16"PRIx64" ", (uint64_t)addr, (uint64_t)addr&(cache->cmask), (uint64_t)cache->cmask);

  size_t *clines    = cache->clines  + (addr&(cache->cmask));
  uint64_t *latimes = cache->latimes + (addr&(cache->cmask));

  cache->clock++;
  for (i=0; i<nway; i++) { /* look for hits */
    if (clines[i] == addr) { 
      cache->nhits++;
      latimes[i] = cache->clock;
      goto DONE;
    }
  }

  for (i=0; i<nway; i++) { /* look for empty spots or the lru spot */
    if (clines[i] == 0) {
      lru = i;
      break;
    }
    else if (latimes[i] < latimes[lru]) {
      lru = i;
    }
  }

  /* initial fill or replace */
  cache->nmisses++;
  clines[lru]  = addr;
  latimes[lru] = cache->clock;

DONE:
  //printf(" %"PRIu64" %"PRIu64"\n", cache->nhits, cache->clock);
  return 1;
}


/*************************************************************************/
/*! This function returns the cache's hitrate
 */
/*************************************************************************/
double gk_cacheGetHitRate(gk_cache_t *cache)
{
  return ((double)cache->nhits)/((double)(cache->clock+1));
}

