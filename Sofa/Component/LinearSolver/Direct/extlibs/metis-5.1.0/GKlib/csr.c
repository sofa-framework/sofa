/*!
 * \file 
 *
 * \brief Various routines with dealing with CSR matrices
 *
 * \author George Karypis
 * \version\verbatim $Id: csr.c 21044 2017-05-24 22:50:32Z karypis $ \endverbatim
 */

#include <GKlib.h>

#define OMPMINOPS       50000

/*************************************************************************/
/*! Allocate memory for a CSR matrix and initializes it 
    \returns the allocated matrix. The various fields are set to NULL.
*/
/**************************************************************************/
gk_csr_t *gk_csr_Create()
{
  gk_csr_t *mat=NULL;

  if ((mat = (gk_csr_t *)gk_malloc(sizeof(gk_csr_t), "gk_csr_Create: mat")))
    gk_csr_Init(mat);

  return mat;
}


/*************************************************************************/
/*! Initializes the matrix 
    \param mat is the matrix to be initialized.
*/
/*************************************************************************/
void gk_csr_Init(gk_csr_t *mat)
{
  memset(mat, 0, sizeof(gk_csr_t));
  mat->nrows = mat->ncols = 0;
}


/*************************************************************************/
/*! Frees all the memory allocated for matrix.
    \param mat is the matrix to be freed.
*/
/*************************************************************************/
void gk_csr_Free(gk_csr_t **mat)
{
  if (*mat == NULL)
    return;
  gk_csr_FreeContents(*mat);
  gk_free((void **)mat, LTERM);
}


/*************************************************************************/
/*! Frees only the memory allocated for the matrix's different fields and
    sets them to NULL.
    \param mat is the matrix whose contents will be freed.
*/    
/*************************************************************************/
void gk_csr_FreeContents(gk_csr_t *mat)
{
  gk_free((void *)&mat->rowptr, &mat->rowind, &mat->rowval, 
      &mat->rowids, &mat->rlabels, &mat->rmap,
      &mat->colptr, &mat->colind, &mat->colval, 
      &mat->colids, &mat->clabels, &mat->cmap,
      &mat->rnorms, &mat->cnorms, &mat->rsums, &mat->csums, 
      &mat->rsizes, &mat->csizes, &mat->rvols, &mat->cvols, 
      &mat->rwgts, &mat->cwgts, 
          LTERM);
}


/*************************************************************************/
/*! Returns a copy of a matrix.
    \param mat is the matrix to be duplicated.
    \returns the newly created copy of the matrix.
*/
/**************************************************************************/
gk_csr_t *gk_csr_Dup(gk_csr_t *mat)
{
  gk_csr_t *nmat;

  nmat = gk_csr_Create();

  nmat->nrows  = mat->nrows;
  nmat->ncols  = mat->ncols;

  /* copy the row structure */
  if (mat->rowptr)
    nmat->rowptr = gk_zcopy(mat->nrows+1, mat->rowptr, 
                            gk_zmalloc(mat->nrows+1, "gk_csr_Dup: rowptr"));
  if (mat->rowids)
    nmat->rowids = gk_icopy(mat->nrows, mat->rowids, 
                            gk_imalloc(mat->nrows, "gk_csr_Dup: rowids"));
  if (mat->rlabels)
    nmat->rlabels = gk_icopy(mat->nrows, mat->rlabels, 
                            gk_imalloc(mat->nrows, "gk_csr_Dup: rlabels"));
  if (mat->rnorms)
    nmat->rnorms = gk_fcopy(mat->nrows, mat->rnorms, 
                            gk_fmalloc(mat->nrows, "gk_csr_Dup: rnorms"));
  if (mat->rsums)
    nmat->rsums = gk_fcopy(mat->nrows, mat->rsums, 
                            gk_fmalloc(mat->nrows, "gk_csr_Dup: rsums"));
  if (mat->rsizes)
    nmat->rsizes = gk_fcopy(mat->nrows, mat->rsizes, 
                            gk_fmalloc(mat->nrows, "gk_csr_Dup: rsizes"));
  if (mat->rvols)
    nmat->rvols = gk_fcopy(mat->nrows, mat->rvols, 
                            gk_fmalloc(mat->nrows, "gk_csr_Dup: rvols"));
  if (mat->rwgts)
    nmat->rwgts = gk_fcopy(mat->nrows, mat->rwgts, 
                            gk_fmalloc(mat->nrows, "gk_csr_Dup: rwgts"));
  if (mat->rowind)
    nmat->rowind = gk_icopy(mat->rowptr[mat->nrows], mat->rowind, 
                            gk_imalloc(mat->rowptr[mat->nrows], "gk_csr_Dup: rowind"));
  if (mat->rowval)
    nmat->rowval = gk_fcopy(mat->rowptr[mat->nrows], mat->rowval, 
                            gk_fmalloc(mat->rowptr[mat->nrows], "gk_csr_Dup: rowval"));

  /* copy the col structure */
  if (mat->colptr)
    nmat->colptr = gk_zcopy(mat->ncols+1, mat->colptr, 
                            gk_zmalloc(mat->ncols+1, "gk_csr_Dup: colptr"));
  if (mat->colids)
    nmat->colids = gk_icopy(mat->ncols, mat->colids, 
                            gk_imalloc(mat->ncols, "gk_csr_Dup: colids"));
  if (mat->clabels)
    nmat->clabels = gk_icopy(mat->ncols, mat->clabels, 
                            gk_imalloc(mat->ncols, "gk_csr_Dup: clabels"));
  if (mat->cnorms)
    nmat->cnorms = gk_fcopy(mat->ncols, mat->cnorms, 
                            gk_fmalloc(mat->ncols, "gk_csr_Dup: cnorms"));
  if (mat->csums)
    nmat->csums = gk_fcopy(mat->ncols, mat->csums, 
                            gk_fmalloc(mat->ncols, "gk_csr_Dup: csums"));
  if (mat->csizes)
    nmat->csizes = gk_fcopy(mat->ncols, mat->csizes, 
                            gk_fmalloc(mat->ncols, "gk_csr_Dup: csizes"));
  if (mat->cvols)
    nmat->cvols = gk_fcopy(mat->ncols, mat->cvols, 
                            gk_fmalloc(mat->ncols, "gk_csr_Dup: cvols"));
  if (mat->cwgts)
    nmat->cwgts = gk_fcopy(mat->ncols, mat->cwgts, 
                            gk_fmalloc(mat->ncols, "gk_csr_Dup: cwgts"));
  if (mat->colind)
    nmat->colind = gk_icopy(mat->colptr[mat->ncols], mat->colind, 
                            gk_imalloc(mat->colptr[mat->ncols], "gk_csr_Dup: colind"));
  if (mat->colval)
    nmat->colval = gk_fcopy(mat->colptr[mat->ncols], mat->colval, 
                            gk_fmalloc(mat->colptr[mat->ncols], "gk_csr_Dup: colval"));

  return nmat;
}


/*************************************************************************/
/*! Returns a submatrix containint a set of consecutive rows.
    \param mat is the original matrix.
    \param rstart is the starting row.
    \param nrows is the number of rows from rstart to extract.
    \returns the row structure of the newly created submatrix.
*/
/**************************************************************************/
gk_csr_t *gk_csr_ExtractSubmatrix(gk_csr_t *mat, int rstart, int nrows)
{
  ssize_t i;
  gk_csr_t *nmat;

  if (rstart+nrows > mat->nrows)
    return NULL;

  nmat = gk_csr_Create();

  nmat->nrows  = nrows;
  nmat->ncols  = mat->ncols;

  /* copy the row structure */
  if (mat->rowptr)
    nmat->rowptr = gk_zcopy(nrows+1, mat->rowptr+rstart, 
                              gk_zmalloc(nrows+1, "gk_csr_ExtractSubmatrix: rowptr"));
  for (i=nrows; i>=0; i--)
    nmat->rowptr[i] -= nmat->rowptr[0];
  ASSERT(nmat->rowptr[0] == 0);

  if (mat->rowids)
    nmat->rowids = gk_icopy(nrows, mat->rowids+rstart, 
                            gk_imalloc(nrows, "gk_csr_ExtractSubmatrix: rowids"));
  if (mat->rnorms)
    nmat->rnorms = gk_fcopy(nrows, mat->rnorms+rstart, 
                            gk_fmalloc(nrows, "gk_csr_ExtractSubmatrix: rnorms"));

  if (mat->rsums)
    nmat->rsums = gk_fcopy(nrows, mat->rsums+rstart, 
                            gk_fmalloc(nrows, "gk_csr_ExtractSubmatrix: rsums"));

  ASSERT(nmat->rowptr[nrows] == mat->rowptr[rstart+nrows]-mat->rowptr[rstart]);
  if (mat->rowind)
    nmat->rowind = gk_icopy(mat->rowptr[rstart+nrows]-mat->rowptr[rstart], 
                            mat->rowind+mat->rowptr[rstart], 
                            gk_imalloc(mat->rowptr[rstart+nrows]-mat->rowptr[rstart],
                                       "gk_csr_ExtractSubmatrix: rowind"));
  if (mat->rowval)
    nmat->rowval = gk_fcopy(mat->rowptr[rstart+nrows]-mat->rowptr[rstart], 
                            mat->rowval+mat->rowptr[rstart], 
                            gk_fmalloc(mat->rowptr[rstart+nrows]-mat->rowptr[rstart],
                                       "gk_csr_ExtractSubmatrix: rowval"));

  return nmat;
}


/*************************************************************************/
/*! Returns a submatrix containing a certain set of rows.
    \param mat is the original matrix.
    \param nrows is the number of rows to extract.
    \param rind is the set of row numbers to extract.
    \returns the row structure of the newly created submatrix.
*/
/**************************************************************************/
gk_csr_t *gk_csr_ExtractRows(gk_csr_t *mat, int nrows, int *rind)
{
  ssize_t i, ii, j, nnz;
  gk_csr_t *nmat;

  nmat = gk_csr_Create();

  nmat->nrows = nrows;
  nmat->ncols = mat->ncols;

  for (nnz=0, i=0; i<nrows; i++)  
    nnz += mat->rowptr[rind[i]+1]-mat->rowptr[rind[i]];

  nmat->rowptr = gk_zmalloc(nmat->nrows+1, "gk_csr_ExtractPartition: rowptr");
  nmat->rowind = gk_imalloc(nnz, "gk_csr_ExtractPartition: rowind");
  nmat->rowval = gk_fmalloc(nnz, "gk_csr_ExtractPartition: rowval");

  nmat->rowptr[0] = 0;
  for (nnz=0, j=0, ii=0; ii<nrows; ii++) {
    i = rind[ii];
    gk_icopy(mat->rowptr[i+1]-mat->rowptr[i], mat->rowind+mat->rowptr[i], nmat->rowind+nnz);
    gk_fcopy(mat->rowptr[i+1]-mat->rowptr[i], mat->rowval+mat->rowptr[i], nmat->rowval+nnz);
    nnz += mat->rowptr[i+1]-mat->rowptr[i];
    nmat->rowptr[++j] = nnz;
  }
  ASSERT(j == nmat->nrows);

  return nmat;
}


/*************************************************************************/
/*! Returns a submatrix corresponding to a specified partitioning of rows.
    \param mat is the original matrix.
    \param part is the partitioning vector of the rows.
    \param pid is the partition ID that will be extracted.
    \returns the row structure of the newly created submatrix.
*/
/**************************************************************************/
gk_csr_t *gk_csr_ExtractPartition(gk_csr_t *mat, int *part, int pid)
{
  ssize_t i, j, nnz;
  gk_csr_t *nmat;

  nmat = gk_csr_Create();

  nmat->nrows = 0;
  nmat->ncols = mat->ncols;

  for (nnz=0, i=0; i<mat->nrows; i++) {
    if (part[i] == pid) {
      nmat->nrows++;
      nnz += mat->rowptr[i+1]-mat->rowptr[i];
    }
  }

  nmat->rowptr = gk_zmalloc(nmat->nrows+1, "gk_csr_ExtractPartition: rowptr");
  nmat->rowind = gk_imalloc(nnz, "gk_csr_ExtractPartition: rowind");
  nmat->rowval = gk_fmalloc(nnz, "gk_csr_ExtractPartition: rowval");

  nmat->rowptr[0] = 0;
  for (nnz=0, j=0, i=0; i<mat->nrows; i++) {
    if (part[i] == pid) {
      gk_icopy(mat->rowptr[i+1]-mat->rowptr[i], mat->rowind+mat->rowptr[i], nmat->rowind+nnz);
      gk_fcopy(mat->rowptr[i+1]-mat->rowptr[i], mat->rowval+mat->rowptr[i], nmat->rowval+nnz);
      nnz += mat->rowptr[i+1]-mat->rowptr[i];
      nmat->rowptr[++j] = nnz;
    }
  }
  ASSERT(j == nmat->nrows);

  return nmat;
}


/*************************************************************************/
/*! Splits the matrix into multiple sub-matrices based on the provided
    color array.
    \param mat is the original matrix.
    \param color is an array of size equal to the number of non-zeros
           in the matrix (row-wise structure). The matrix is split into
           as many parts as the number of colors. For meaningfull results,
           the colors should be numbered consecutively starting from 0.
    \returns an array of matrices for each supplied color number.
*/
/**************************************************************************/
gk_csr_t **gk_csr_Split(gk_csr_t *mat, int *color)
{
  ssize_t i, j;
  int nrows, ncolors;
  ssize_t *rowptr;
  int *rowind;
  float *rowval;
  gk_csr_t **smats;

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  ncolors = gk_imax(rowptr[nrows], color, 1)+1;

  smats = (gk_csr_t **)gk_malloc(sizeof(gk_csr_t *)*ncolors, "gk_csr_Split: smats");
  for (i=0; i<ncolors; i++) {
    smats[i] = gk_csr_Create();
    smats[i]->nrows  = mat->nrows;
    smats[i]->ncols  = mat->ncols;
    smats[i]->rowptr = gk_zsmalloc(nrows+1, 0, "gk_csr_Split: smats[i]->rowptr"); 
  }

  for (i=0; i<nrows; i++) {
    for (j=rowptr[i]; j<rowptr[i+1]; j++) 
      smats[color[j]]->rowptr[i]++;
  }
  for (i=0; i<ncolors; i++) 
    MAKECSR(j, nrows, smats[i]->rowptr);

  for (i=0; i<ncolors; i++) {
    smats[i]->rowind = gk_imalloc(smats[i]->rowptr[nrows], "gk_csr_Split: smats[i]->rowind"); 
    smats[i]->rowval = gk_fmalloc(smats[i]->rowptr[nrows], "gk_csr_Split: smats[i]->rowval"); 
  }

  for (i=0; i<nrows; i++) {
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      smats[color[j]]->rowind[smats[color[j]]->rowptr[i]] = rowind[j];
      smats[color[j]]->rowval[smats[color[j]]->rowptr[i]] = rowval[j];
      smats[color[j]]->rowptr[i]++;
    }
  }

  for (i=0; i<ncolors; i++) 
    SHIFTCSR(j, nrows, smats[i]->rowptr);

  return smats;
}


/**************************************************************************/
/*! Determines the format of the CSR matrix based on the extension.
    \param filename is the name of the file.
    \param the user-supplied format.
    \returns the type. The extension of the file directly maps to the
           name of the format.
*/
/**************************************************************************/
int gk_csr_DetermineFormat(char *filename, int format)
{
  if (format != GK_CSR_FMT_AUTO)
    return format;

  format = GK_CSR_FMT_CSR;
  char *extension = gk_getextname(filename);

  if (!strcmp(extension, "csr"))
    format = GK_CSR_FMT_CSR;
  else if (!strcmp(extension, "ijv"))
    format = GK_CSR_FMT_IJV;
  else if (!strcmp(extension, "cluto"))
    format = GK_CSR_FMT_CLUTO;
  else if (!strcmp(extension, "metis"))
    format = GK_CSR_FMT_METIS;
  else if (!strcmp(extension, "binrow"))
    format = GK_CSR_FMT_BINROW;
  else if (!strcmp(extension, "bincol"))
    format = GK_CSR_FMT_BINCOL;
  else if (!strcmp(extension, "bijv"))
    format = GK_CSR_FMT_BIJV;

  gk_free((void **)&extension, LTERM);

  return format;
}


/**************************************************************************/
/*! Reads a CSR matrix from the supplied file and stores it the matrix's 
    forward structure.
    \param filename is the file that stores the data.
    \param format is either GK_CSR_FMT_METIS, GK_CSR_FMT_CLUTO, 
           GK_CSR_FMT_CSR, GK_CSR_FMT_BINROW, GK_CSR_FMT_BINCOL 
           specifying the type of the input format. 
           The GK_CSR_FMT_CSR does not contain a header
           line, whereas the GK_CSR_FMT_BINROW is a binary format written 
           by gk_csr_Write() using the same format specifier.
    \param readvals is either 1 or 0, indicating if the CSR file contains
           values or it does not. It only applies when GK_CSR_FMT_CSR is
           used.
    \param numbering is either 1 or 0, indicating if the numbering of the 
           indices start from 1 or 0, respectively. If they start from 1, 
           they are automatically decreamented during input so that they
           will start from 0. It only applies when GK_CSR_FMT_CSR is
           used.
    \returns the matrix that was read.
*/
/**************************************************************************/
gk_csr_t *gk_csr_Read(char *filename, int format, int readvals, int numbering)
{
  ssize_t i, k, l;
  size_t nfields, nrows, ncols, nnz, fmt, ncon;
  size_t lnlen;
  ssize_t *rowptr;
  int *rowind, *iinds, *jinds, ival;
  float *rowval=NULL, *vals, fval;
  int readsizes, readwgts;
  char *line=NULL, *head, *tail, fmtstr[256];
  FILE *fpin;
  gk_csr_t *mat=NULL;

  format = gk_csr_DetermineFormat(filename, format);

  if (!gk_fexists(filename)) 
    gk_errexit(SIGERR, "File %s does not exist!\n", filename);

  switch (format) {
    case GK_CSR_FMT_BINROW:
      mat = gk_csr_Create();

      fpin = gk_fopen(filename, "rb", "gk_csr_Read: fpin");
      if (fread(&(mat->nrows), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the nrows from file %s!\n", filename);
      if (fread(&(mat->ncols), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the ncols from file %s!\n", filename);
      mat->rowptr = gk_zmalloc(mat->nrows+1, "gk_csr_Read: rowptr");
      if (fread(mat->rowptr, sizeof(ssize_t), mat->nrows+1, fpin) != mat->nrows+1)
        gk_errexit(SIGERR, "Failed to read the rowptr from file %s!\n", filename);
      mat->rowind = gk_imalloc(mat->rowptr[mat->nrows], "gk_csr_Read: rowind");
      if (fread(mat->rowind, sizeof(int32_t), mat->rowptr[mat->nrows], fpin) != mat->rowptr[mat->nrows])
        gk_errexit(SIGERR, "Failed to read the rowind from file %s!\n", filename);
      if (readvals == 1) {
        mat->rowval = gk_fmalloc(mat->rowptr[mat->nrows], "gk_csr_Read: rowval");
        if (fread(mat->rowval, sizeof(float), mat->rowptr[mat->nrows], fpin) != mat->rowptr[mat->nrows])
          gk_errexit(SIGERR, "Failed to read the rowval from file %s!\n", filename);
      }

      gk_fclose(fpin);
      return mat;

      break;

    case GK_CSR_FMT_BINCOL:
      mat = gk_csr_Create();

      fpin = gk_fopen(filename, "rb", "gk_csr_Read: fpin");
      if (fread(&(mat->nrows), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the nrows from file %s!\n", filename);
      if (fread(&(mat->ncols), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the ncols from file %s!\n", filename);
      mat->colptr = gk_zmalloc(mat->ncols+1, "gk_csr_Read: colptr");
      if (fread(mat->colptr, sizeof(ssize_t), mat->ncols+1, fpin) != mat->ncols+1)
        gk_errexit(SIGERR, "Failed to read the colptr from file %s!\n", filename);
      mat->colind = gk_imalloc(mat->colptr[mat->ncols], "gk_csr_Read: colind");
      if (fread(mat->colind, sizeof(int32_t), mat->colptr[mat->ncols], fpin) != mat->colptr[mat->ncols])
        gk_errexit(SIGERR, "Failed to read the colind from file %s!\n", filename);
      if (readvals) {
        mat->colval = gk_fmalloc(mat->colptr[mat->ncols], "gk_csr_Read: colval");
        if (fread(mat->colval, sizeof(float), mat->colptr[mat->ncols], fpin) != mat->colptr[mat->ncols])
          gk_errexit(SIGERR, "Failed to read the colval from file %s!\n", filename);
      }

      gk_fclose(fpin);
      return mat;

      break;


    case GK_CSR_FMT_IJV:
      gk_getfilestats(filename, &nrows, &nnz, NULL, NULL);

      if (readvals == 1 && 3*nrows != nnz)
        gk_errexit(SIGERR, "Error: The number of numbers (%zd %d) in the input file is not a multiple of 3.\n", nnz, readvals);
      if (readvals == 0 && 2*nrows != nnz)
        gk_errexit(SIGERR, "Error: The number of numbers (%zd %d) in the input file is not a multiple of 2.\n", nnz, readvals);

      nnz = nrows;
      numbering = (numbering ? - 1 : 0);

      /* read the data into three arrays */
      iinds = gk_i32malloc(nnz, "iinds");
      jinds = gk_i32malloc(nnz, "jinds");
      vals  = (readvals ? gk_fmalloc(nnz, "vals") : NULL);

      fpin = gk_fopen(filename, "r", "gk_csr_Read: fpin");
      for (nrows=0, ncols=0, i=0; i<nnz; i++) {
        if (readvals) {
          if (fscanf(fpin, "%d %d %f", &iinds[i], &jinds[i], &vals[i]) != 3)
            gk_errexit(SIGERR, "Error: Failed to read (i, j, val) for nnz: %zd.\n", i);
        }
        else {
          if (fscanf(fpin, "%d %d", &iinds[i], &jinds[i]) != 2)
            gk_errexit(SIGERR, "Error: Failed to read (i, j) value for nnz: %zd.\n", i);
        }
        iinds[i] += numbering;
        jinds[i] += numbering;

        if (nrows < iinds[i])
          nrows = iinds[i];
        if (ncols < jinds[i])
          ncols = jinds[i];
      }
      nrows++;
      ncols++;
      gk_fclose(fpin);

      /* convert (i, j, v) into a CSR matrix */
      mat = gk_csr_Create();
      mat->nrows = nrows;
      mat->ncols = ncols;
      rowptr = mat->rowptr = gk_zsmalloc(nrows+1, 0, "rowptr");
      rowind = mat->rowind = gk_i32malloc(nnz, "rowind");
      if (readvals)
        rowval = mat->rowval = gk_fmalloc(nnz, "rowval");

      for (i=0; i<nnz; i++)
        rowptr[iinds[i]]++;
      MAKECSR(i, nrows, rowptr);

      for (i=0; i<nnz; i++) {
        rowind[rowptr[iinds[i]]] = jinds[i];
        if (readvals)
          rowval[rowptr[iinds[i]]] = vals[i];
        rowptr[iinds[i]]++;
      }
      SHIFTCSR(i, nrows, rowptr);

      gk_free((void **)&iinds, &jinds, &vals, LTERM);

      return mat;

      break;

    case GK_CSR_FMT_BIJV:
      mat = gk_csr_Create();

      fpin = gk_fopen(filename, "rb", "gk_csr_Read: fpin");

      if (fread(&(mat->nrows), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the nrows from file %s!\n", filename);
      if (fread(&(mat->ncols), sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the ncols from file %s!\n", filename);
      if (fread(&nnz, sizeof(size_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the nnz from file %s!\n", filename);
      if (fread(&readvals, sizeof(int32_t), 1, fpin) != 1)
        gk_errexit(SIGERR, "Failed to read the readvals from file %s!\n", filename);

      /* read the data into three arrays */
      iinds = gk_i32malloc(nnz, "iinds");
      jinds = gk_i32malloc(nnz, "jinds");
      vals  = (readvals ? gk_fmalloc(nnz, "vals") : NULL);

      for (i=0; i<nnz; i++) {
        if (fread(&(iinds[i]), sizeof(int32_t), 1, fpin) != 1)
          gk_errexit(SIGERR, "Failed to read iinds[i] from file %s!\n", filename);
        if (fread(&(jinds[i]), sizeof(int32_t), 1, fpin) != 1)
          gk_errexit(SIGERR, "Failed to read jinds[i] from file %s!\n", filename);
        if (readvals) {
          if (fread(&(vals[i]), sizeof(float), 1, fpin) != 1)
            gk_errexit(SIGERR, "Failed to read vals[i] from file %s!\n", filename);
        }
        //printf("%d %d\n", iinds[i], jinds[i]);
      }
      gk_fclose(fpin);

      /* convert (i, j, v) into a CSR matrix */
      rowptr = mat->rowptr = gk_zsmalloc(mat->nrows+1, 0, "rowptr");
      rowind = mat->rowind = gk_i32malloc(nnz, "rowind");
      if (readvals)
        rowval = mat->rowval = gk_fmalloc(nnz, "rowval");

      for (i=0; i<nnz; i++)
        rowptr[iinds[i]]++;
      MAKECSR(i, mat->nrows, rowptr);

      for (i=0; i<nnz; i++) {
        rowind[rowptr[iinds[i]]] = jinds[i];
        if (readvals)
          rowval[rowptr[iinds[i]]] = vals[i];
        rowptr[iinds[i]]++;
      }
      SHIFTCSR(i, mat->nrows, rowptr);

      gk_free((void **)&iinds, &jinds, &vals, LTERM);

      return mat;

      break;


    /* the following are handled by a common input code, that comes after the switch */

    case GK_CSR_FMT_CLUTO:
      fpin = gk_fopen(filename, "r", "gk_csr_Read: fpin");
      do {
        if (gk_getline(&line, &lnlen, fpin) <= 0)
          gk_errexit(SIGERR, "Premature end of input file: file:%s\n", filename);
      } while (line[0] == '%');

      if (sscanf(line, "%zu %zu %zu", &nrows, &ncols, &nnz) != 3)
        gk_errexit(SIGERR, "Header line must contain 3 integers.\n");

      readsizes = 0;
      readwgts  = 0;
      readvals  = 1;
      numbering = 1;

      break;

    case GK_CSR_FMT_METIS:
      fpin = gk_fopen(filename, "r", "gk_csr_Read: fpin");
      do {
        if (gk_getline(&line, &lnlen, fpin) <= 0)
          gk_errexit(SIGERR, "Premature end of input file: file:%s\n", filename);
      } while (line[0] == '%');

      fmt = ncon = 0;
      nfields = sscanf(line, "%zu %zu %zu %zu", &nrows, &nnz, &fmt, &ncon);
      if (nfields < 2)
        gk_errexit(SIGERR, "Header line must contain at least 2 integers (#vtxs and #edges).\n");

      ncols = nrows;
      nnz *= 2;

      if (fmt > 111)
        gk_errexit(SIGERR, "Cannot read this type of file format [fmt=%zu]!\n", fmt);

      sprintf(fmtstr, "%03zu", fmt%1000);
      readsizes = (fmtstr[0] == '1');
      readwgts  = (fmtstr[1] == '1');
      readvals  = (fmtstr[2] == '1');
      numbering = 1;
      ncon      = (ncon == 0 ? 1 : ncon);

      break;

    case GK_CSR_FMT_CSR:
      readsizes = 0;
      readwgts  = 0;

      gk_getfilestats(filename, &nrows, &nnz, NULL, NULL);

      if (readvals == 1 && nnz%2 == 1)
        gk_errexit(SIGERR, "Error: The number of numbers (%zd %d) in the input file is not even.\n", nnz, readvals);
      if (readvals == 1)
        nnz = nnz/2;
      fpin = gk_fopen(filename, "r", "gk_csr_Read: fpin");

      break;

    default:
      gk_errexit(SIGERR, "Unknown csr format.\n");
      return NULL;
  }

  mat = gk_csr_Create();

  mat->nrows = nrows;

  rowptr = mat->rowptr = gk_zmalloc(nrows+1, "gk_csr_Read: rowptr");
  rowind = mat->rowind = gk_imalloc(nnz, "gk_csr_Read: rowind");
  if (readvals != 2)
    rowval = mat->rowval = gk_fsmalloc(nnz, 1.0, "gk_csr_Read: rowval");

  if (readsizes)
    mat->rsizes = gk_fsmalloc(nrows, 0.0, "gk_csr_Read: rsizes");

  if (readwgts)
    mat->rwgts = gk_fsmalloc(nrows*ncon, 0.0, "gk_csr_Read: rwgts");

  /*----------------------------------------------------------------------
   * Read the sparse matrix file
   *---------------------------------------------------------------------*/
  numbering = (numbering ? -1 : 0);
  for (ncols=0, rowptr[0]=0, k=0, i=0; i<nrows; i++) {
    do {
      if (gk_getline(&line, &lnlen, fpin) == -1)
        gk_errexit(SIGERR, "Premature end of input file: file while reading row %d\n", i);
    } while (line[0] == '%');

    head = line;
    tail = NULL;

    /* Read vertex sizes */
    if (readsizes) {
#ifdef __MSC__
      mat->rsizes[i] = (float)strtod(head, &tail);
#else
      mat->rsizes[i] = strtof(head, &tail);
#endif
      if (tail == head)
        gk_errexit(SIGERR, "The line for vertex %zd does not have size information\n", i+1);
      if (mat->rsizes[i] < 0)
        errexit("The size for vertex %zd must be >= 0\n", i+1);
      head = tail;
    }

    /* Read vertex weights */
    if (readwgts) {
      for (l=0; l<ncon; l++) {
#ifdef __MSC__
        mat->rwgts[i*ncon+l] = (float)strtod(head, &tail);
#else
        mat->rwgts[i*ncon+l] = strtof(head, &tail);
#endif
        if (tail == head)
          errexit("The line for vertex %zd does not have enough weights "
                  "for the %d constraints.\n", i+1, ncon);
        if (mat->rwgts[i*ncon+l] < 0)
          errexit("The weight vertex %zd and constraint %zd must be >= 0\n", i+1, l);
        head = tail;
      }
    }

   
    /* Read the rest of the row */
    while (1) {
      ival = (int)strtol(head, &tail, 0);
      if (tail == head) 
        break;
      head = tail;
      
      if ((rowind[k] = ival + numbering) < 0)
        gk_errexit(SIGERR, "Error: Invalid column number %d at row %zd.\n", ival, i);

      ncols = gk_max(rowind[k], ncols);

      if (readvals == 1) {
#ifdef __MSC__
        fval = (float)strtod(head, &tail);
#else
	fval = strtof(head, &tail);
#endif
        if (tail == head)
          gk_errexit(SIGERR, "Value could not be found for column! Row:%zd, NNZ:%zd\n", i, k);
        head = tail;

        rowval[k] = fval;
      }
      k++;
    }
    rowptr[i+1] = k;
  }

  if (format == GK_CSR_FMT_METIS) {
    ASSERT(ncols+1 == mat->nrows);
    mat->ncols = mat->nrows;
  }
  else {
    mat->ncols = ncols+1;
  }

  if (k != nnz)
    gk_errexit(SIGERR, "gk_csr_Read: Something wrong with the number of nonzeros in "
                       "the input file. NNZ=%zd, ActualNNZ=%zd.\n", nnz, k);

  gk_fclose(fpin);

  gk_free((void **)&line, LTERM);

  return mat;
}


/**************************************************************************/
/*! Writes the row-based structure of a matrix into a file.
    \param mat is the matrix to be written,
    \param filename is the name of the output file.
    \param format is one of: GK_CSR_FMT_CLUTO, GK_CSR_FMT_CSR, 
           GK_CSR_FMT_BINROW, GK_CSR_FMT_BINCOL, GK_CSR_FMT_BIJV.
    \param writevals is either 1 or 0 indicating if the values will be 
           written or not. This is only applicable when GK_CSR_FMT_CSR
           is used.
    \param numbering is either 1 or 0 indicating if the internal 0-based 
           numbering will be shifted by one or not during output. This 
           is only applicable when GK_CSR_FMT_CSR is used.
*/
/**************************************************************************/
void gk_csr_Write(gk_csr_t *mat, char *filename, int format, int writevals, int numbering)
{
  ssize_t i, j;
  int32_t edge[2];
  FILE *fpout;

  format = gk_csr_DetermineFormat(filename, format);

  switch (format) {
    case GK_CSR_FMT_METIS:
      if (mat->nrows != mat->ncols || mat->rowptr[mat->nrows]%2 == 1)
        gk_errexit(SIGERR, "METIS output format requires a square symmetric matrix.\n");

      if (filename)
        fpout = gk_fopen(filename, "w", "gk_csr_Write: fpout");
      else
        fpout = stdout; 

      fprintf(fpout, "%d %zd\n", mat->nrows, mat->rowptr[mat->nrows]/2);
      for (i=0; i<mat->nrows; i++) {
        for (j=mat->rowptr[i]; j<mat->rowptr[i+1]; j++) 
          fprintf(fpout, " %d", mat->rowind[j]+1);
        fprintf(fpout, "\n");
      }
      if (filename)
        gk_fclose(fpout);
      break;

    case GK_CSR_FMT_BINROW:
      if (filename == NULL)
        gk_errexit(SIGERR, "The filename parameter cannot be NULL.\n");
      fpout = gk_fopen(filename, "wb", "gk_csr_Write: fpout");

      fwrite(&(mat->nrows), sizeof(int32_t), 1, fpout); 
      fwrite(&(mat->ncols), sizeof(int32_t), 1, fpout); 
      fwrite(mat->rowptr, sizeof(ssize_t), mat->nrows+1, fpout); 
      fwrite(mat->rowind, sizeof(int32_t), mat->rowptr[mat->nrows], fpout); 
      if (writevals)
        fwrite(mat->rowval, sizeof(float), mat->rowptr[mat->nrows], fpout); 

      gk_fclose(fpout);
      return;

      break;

    case GK_CSR_FMT_BINCOL:
      if (filename == NULL)
        gk_errexit(SIGERR, "The filename parameter cannot be NULL.\n");
      fpout = gk_fopen(filename, "wb", "gk_csr_Write: fpout");

      fwrite(&(mat->nrows), sizeof(int32_t), 1, fpout); 
      fwrite(&(mat->ncols), sizeof(int32_t), 1, fpout); 
      fwrite(mat->colptr, sizeof(ssize_t), mat->ncols+1, fpout); 
      fwrite(mat->colind, sizeof(int32_t), mat->colptr[mat->ncols], fpout); 
      if (writevals) 
        fwrite(mat->colval, sizeof(float), mat->colptr[mat->ncols], fpout); 

      gk_fclose(fpout);
      return;

      break;

    case GK_CSR_FMT_IJV:
      if (filename == NULL)
        gk_errexit(SIGERR, "The filename parameter cannot be NULL.\n");
      fpout = gk_fopen(filename, "w", "gk_csr_Write: fpout");

      numbering = (numbering ? 1 : 0);
      for (i=0; i<mat->nrows; i++) {
        for (j=mat->rowptr[i]; j<mat->rowptr[i+1]; j++) {
          if (writevals)
            fprintf(fpout, "%zd %d %.8f\n", i+numbering, mat->rowind[j]+numbering, mat->rowval[j]);
          else
            fprintf(fpout, "%zd %d\n", i+numbering, mat->rowind[j]+numbering);
        }
      }

      gk_fclose(fpout);
      return;

      break;

    case GK_CSR_FMT_BIJV:
      if (filename == NULL)
        gk_errexit(SIGERR, "The filename parameter cannot be NULL.\n");
      fpout = gk_fopen(filename, "wb", "gk_csr_Write: fpout");

      fwrite(&(mat->nrows), sizeof(int32_t), 1, fpout); 
      fwrite(&(mat->ncols), sizeof(int32_t), 1, fpout); 
      fwrite(&(mat->rowptr[mat->nrows]), sizeof(size_t), 1, fpout); 
      fwrite(&writevals, sizeof(int32_t), 1, fpout); 

      for (i=0; i<mat->nrows; i++) {
        edge[0] = i;
        for (j=mat->rowptr[i]; j<mat->rowptr[i+1]; j++) {
          edge[1] = mat->rowind[j];
          fwrite(edge, sizeof(int32_t), 2, fpout);
          if (writevals) 
            fwrite(&(mat->rowval[j]), sizeof(float), 1, fpout);
        }
      }

      gk_fclose(fpout);
      return;

      break;

    default:
      if (filename)
        fpout = gk_fopen(filename, "w", "gk_csr_Write: fpout");
      else
        fpout = stdout; 

      if (format == GK_CSR_FMT_CLUTO) {
        fprintf(fpout, "%d %d %zd\n", mat->nrows, mat->ncols, mat->rowptr[mat->nrows]);
        writevals = 1;
        numbering = 1;
      }

      for (i=0; i<mat->nrows; i++) {
        for (j=mat->rowptr[i]; j<mat->rowptr[i+1]; j++) {
          fprintf(fpout, " %d", mat->rowind[j]+(numbering ? 1 : 0));
          if (writevals) 
            fprintf(fpout, " %f", mat->rowval[j]);
        }
        fprintf(fpout, "\n");
      }
      if (filename)
        gk_fclose(fpout);
  }
}


/*************************************************************************/
/*! Prunes certain rows/columns of the matrix. The prunning takes place 
    by analyzing the row structure of the matrix. The prunning takes place
    by removing rows/columns but it does not affect the numbering of the
    remaining rows/columns.
   
    \param mat the matrix to be prunned,
    \param what indicates if the rows (GK_CSR_ROW) or the columns (GK_CSR_COL)
           of the matrix will be prunned,
    \param minf is the minimum number of rows (columns) that a column (row) must
           be present in order to be kept,
    \param maxf is the maximum number of rows (columns) that a column (row) must
          be present at in order to be kept.
    \returns the prunned matrix consisting only of its row-based structure. 
          The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_Prune(gk_csr_t *mat, int what, int minf, int maxf)
{
  ssize_t i, j, nnz;
  int nrows, ncols;
  ssize_t *rowptr, *nrowptr;
  int *rowind, *nrowind, *collen;
  float *rowval, *nrowval;
  gk_csr_t *nmat;

  nmat = gk_csr_Create();
  
  nrows = nmat->nrows = mat->nrows;
  ncols = nmat->ncols = mat->ncols;

  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_Prune: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(rowptr[nrows], "gk_csr_Prune: nrowind");
  nrowval = nmat->rowval = gk_fmalloc(rowptr[nrows], "gk_csr_Prune: nrowval");


  switch (what) {
    case GK_CSR_COL:
      collen = gk_ismalloc(ncols, 0, "gk_csr_Prune: collen");

      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) {
          ASSERT(rowind[j] < ncols);
          collen[rowind[j]]++;
        }
      }
      for (i=0; i<ncols; i++)
        collen[i] = (collen[i] >= minf && collen[i] <= maxf ? 1 : 0);

      nrowptr[0] = 0;
      for (nnz=0, i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) {
          if (collen[rowind[j]]) {
            nrowind[nnz] = rowind[j];
            nrowval[nnz] = rowval[j];
            nnz++;
          }
        }
        nrowptr[i+1] = nnz;
      }
      gk_free((void **)&collen, LTERM);
      break;

    case GK_CSR_ROW:
      nrowptr[0] = 0;
      for (nnz=0, i=0; i<nrows; i++) {
        if (rowptr[i+1]-rowptr[i] >= minf && rowptr[i+1]-rowptr[i] <= maxf) {
          for (j=rowptr[i]; j<rowptr[i+1]; j++, nnz++) {
            nrowind[nnz] = rowind[j];
            nrowval[nnz] = rowval[j];
          }
        }
        nrowptr[i+1] = nnz;
      }
      break;

    default:
      gk_csr_Free(&nmat);
      gk_errexit(SIGERR, "Unknown prunning type of %d\n", what);
      return NULL;
  }

  return nmat;
}


/*************************************************************************/
/*! Eliminates certain entries from the rows/columns of the matrix. The 
    filtering takes place by keeping only the highest weight entries whose
    sum accounts for a certain fraction of the overall weight of the 
    row/column.
   
    \param mat the matrix to be prunned,
    \param what indicates if the rows (GK_CSR_ROW) or the columns (GK_CSR_COL)
           of the matrix will be prunned,
    \param norm indicates the norm that will be used to aggregate the weights
           and possible values are 1 or 2,
    \param fraction is the fraction of the overall norm that will be retained
           by the kept entries.
    \returns the filtered matrix consisting only of its row-based structure. 
           The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_LowFilter(gk_csr_t *mat, int what, int norm, float fraction)
{
  ssize_t i, j, nnz;
  int nrows, ncols, ncand, maxlen=0;
  ssize_t *rowptr, *colptr, *nrowptr;
  int *rowind, *colind, *nrowind;
  float *rowval, *colval, *nrowval, rsum, tsum;
  gk_csr_t *nmat;
  gk_fkv_t *cand;

  nmat = gk_csr_Create();
  
  nrows = nmat->nrows = mat->nrows;
  ncols = nmat->ncols = mat->ncols;

  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;
  colptr = mat->colptr;
  colind = mat->colind;
  colval = mat->colval;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_LowFilter: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(rowptr[nrows], "gk_csr_LowFilter: nrowind");
  nrowval = nmat->rowval = gk_fmalloc(rowptr[nrows], "gk_csr_LowFilter: nrowval");


  switch (what) {
    case GK_CSR_COL:
      if (mat->colptr == NULL) 
        gk_errexit(SIGERR, "Cannot filter columns when column-based structure has not been created.\n");

      gk_zcopy(nrows+1, rowptr, nrowptr);

      for (i=0; i<ncols; i++) 
        maxlen = gk_max(maxlen, colptr[i+1]-colptr[i]);

      #pragma omp parallel private(i, j, ncand, rsum, tsum, cand)
      {
        cand = gk_fkvmalloc(maxlen, "gk_csr_LowFilter: cand");

        #pragma omp for schedule(static)
        for (i=0; i<ncols; i++) {
          for (tsum=0.0, ncand=0, j=colptr[i]; j<colptr[i+1]; j++, ncand++) {
            cand[ncand].val = colind[j];
            cand[ncand].key = colval[j];
            tsum += (norm == 1 ? colval[j] : colval[j]*colval[j]);
          }
          gk_fkvsortd(ncand, cand);

          for (rsum=0.0, j=0; j<ncand && rsum<=fraction*tsum; j++) {
            rsum += (norm == 1 ? cand[j].key : cand[j].key*cand[j].key);
            nrowind[nrowptr[cand[j].val]] = i;
            nrowval[nrowptr[cand[j].val]] = cand[j].key;
            nrowptr[cand[j].val]++;
          }
        }

        gk_free((void **)&cand, LTERM);
      }

      /* compact the nrowind/nrowval */
      for (nnz=0, i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<nrowptr[i]; j++, nnz++) {
          nrowind[nnz] = nrowind[j];
          nrowval[nnz] = nrowval[j];
        }
        nrowptr[i] = nnz;
      }
      SHIFTCSR(i, nrows, nrowptr);

      break;

    case GK_CSR_ROW:
      if (mat->rowptr == NULL) 
        gk_errexit(SIGERR, "Cannot filter rows when row-based structure has not been created.\n");

      for (i=0; i<nrows; i++) 
        maxlen = gk_max(maxlen, rowptr[i+1]-rowptr[i]);

      #pragma omp parallel private(i, j, ncand, rsum, tsum, cand)
      {
        cand = gk_fkvmalloc(maxlen, "gk_csr_LowFilter: cand");

        #pragma omp for schedule(static)
        for (i=0; i<nrows; i++) {
          for (tsum=0.0, ncand=0, j=rowptr[i]; j<rowptr[i+1]; j++, ncand++) {
            cand[ncand].val = rowind[j];
            cand[ncand].key = rowval[j];
            tsum += (norm == 1 ? rowval[j] : rowval[j]*rowval[j]);
          }
          gk_fkvsortd(ncand, cand);

          for (rsum=0.0, j=0; j<ncand && rsum<=fraction*tsum; j++) {
            rsum += (norm == 1 ? cand[j].key : cand[j].key*cand[j].key);
            nrowind[rowptr[i]+j] = cand[j].val;
            nrowval[rowptr[i]+j] = cand[j].key;
          }
          nrowptr[i+1] = rowptr[i]+j;
        }

        gk_free((void **)&cand, LTERM);
      }

      /* compact nrowind/nrowval */
      nrowptr[0] = nnz = 0;
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<nrowptr[i+1]; j++, nnz++) {
          nrowind[nnz] = nrowind[j];
          nrowval[nnz] = nrowval[j];
        }
        nrowptr[i+1] = nnz;
      }

      break;

    default:
      gk_csr_Free(&nmat);
      gk_errexit(SIGERR, "Unknown prunning type of %d\n", what);
      return NULL;
  }

  return nmat;
}


/*************************************************************************/
/*! Eliminates certain entries from the rows/columns of the matrix. The 
    filtering takes place by keeping only the highest weight top-K entries 
    along each row/column and those entries whose weight is greater than
    a specified value.
   
    \param mat the matrix to be prunned,
    \param what indicates if the rows (GK_CSR_ROW) or the columns (GK_CSR_COL)
           of the matrix will be prunned,
    \param topk is the number of the highest weight entries to keep.
    \param keepval is the weight of a term above which will be kept. This
           is used to select additional terms past the first topk.
    \returns the filtered matrix consisting only of its row-based structure. 
           The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_TopKPlusFilter(gk_csr_t *mat, int what, int topk, float keepval)
{
  ssize_t i, j, k, nnz;
  int nrows, ncols, ncand;
  ssize_t *rowptr, *colptr, *nrowptr;
  int *rowind, *colind, *nrowind;
  float *rowval, *colval, *nrowval;
  gk_csr_t *nmat;
  gk_fkv_t *cand;

  nmat = gk_csr_Create();
  
  nrows = nmat->nrows = mat->nrows;
  ncols = nmat->ncols = mat->ncols;

  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;
  colptr = mat->colptr;
  colind = mat->colind;
  colval = mat->colval;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_LowFilter: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(rowptr[nrows], "gk_csr_LowFilter: nrowind");
  nrowval = nmat->rowval = gk_fmalloc(rowptr[nrows], "gk_csr_LowFilter: nrowval");


  switch (what) {
    case GK_CSR_COL:
      if (mat->colptr == NULL) 
        gk_errexit(SIGERR, "Cannot filter columns when column-based structure has not been created.\n");

      cand = gk_fkvmalloc(nrows, "gk_csr_LowFilter: cand");

      gk_zcopy(nrows+1, rowptr, nrowptr);
      for (i=0; i<ncols; i++) {
        for (ncand=0, j=colptr[i]; j<colptr[i+1]; j++, ncand++) {
          cand[ncand].val = colind[j];
          cand[ncand].key = colval[j];
        }
        gk_fkvsortd(ncand, cand);

        k = gk_min(topk, ncand);
        for (j=0; j<k; j++) {
          nrowind[nrowptr[cand[j].val]] = i;
          nrowval[nrowptr[cand[j].val]] = cand[j].key;
          nrowptr[cand[j].val]++;
        }
        for (; j<ncand; j++) {
          if (cand[j].key < keepval) 
            break;

          nrowind[nrowptr[cand[j].val]] = i;
          nrowval[nrowptr[cand[j].val]] = cand[j].key;
          nrowptr[cand[j].val]++;
        }
      }

      /* compact the nrowind/nrowval */
      for (nnz=0, i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<nrowptr[i]; j++, nnz++) {
          nrowind[nnz] = nrowind[j];
          nrowval[nnz] = nrowval[j];
        }
        nrowptr[i] = nnz;
      }
      SHIFTCSR(i, nrows, nrowptr);

      gk_free((void **)&cand, LTERM);
      break;

    case GK_CSR_ROW:
      if (mat->rowptr == NULL) 
        gk_errexit(SIGERR, "Cannot filter rows when row-based structure has not been created.\n");

      cand = gk_fkvmalloc(ncols, "gk_csr_LowFilter: cand");

      nrowptr[0] = 0;
      for (nnz=0, i=0; i<nrows; i++) {
        for (ncand=0, j=rowptr[i]; j<rowptr[i+1]; j++, ncand++) {
          cand[ncand].val = rowind[j];
          cand[ncand].key = rowval[j];
        }
        gk_fkvsortd(ncand, cand);

        k = gk_min(topk, ncand);
        for (j=0; j<k; j++, nnz++) {
          nrowind[nnz] = cand[j].val;
          nrowval[nnz] = cand[j].key;
        }
        for (; j<ncand; j++, nnz++) {
          if (cand[j].key < keepval) 
            break;

          nrowind[nnz] = cand[j].val;
          nrowval[nnz] = cand[j].key;
        }
        nrowptr[i+1] = nnz;
      }

      gk_free((void **)&cand, LTERM);
      break;

    default:
      gk_csr_Free(&nmat);
      gk_errexit(SIGERR, "Unknown prunning type of %d\n", what);
      return NULL;
  }

  return nmat;
}


/*************************************************************************/
/*! Eliminates certain entries from the rows/columns of the matrix. The 
    filtering takes place by keeping only the terms whose contribution to
    the total length of the document is greater than a user-splied multiple
    over the average.

    This routine assumes that the vectors are normalized to be unit length.
   
    \param mat the matrix to be prunned,
    \param what indicates if the rows (GK_CSR_ROW) or the columns (GK_CSR_COL)
           of the matrix will be prunned,
    \param zscore is the multiplicative factor over the average contribution 
           to the length of the document.
    \returns the filtered matrix consisting only of its row-based structure. 
           The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_ZScoreFilter(gk_csr_t *mat, int what, float zscore)
{
  ssize_t i, j, nnz;
  int nrows;
  ssize_t *rowptr, *nrowptr;
  int *rowind, *nrowind;
  float *rowval, *nrowval, avgwgt;
  gk_csr_t *nmat;

  nmat = gk_csr_Create();
  
  nmat->nrows = mat->nrows;
  nmat->ncols = mat->ncols;

  nrows  = mat->nrows; 
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_ZScoreFilter: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(rowptr[nrows], "gk_csr_ZScoreFilter: nrowind");
  nrowval = nmat->rowval = gk_fmalloc(rowptr[nrows], "gk_csr_ZScoreFilter: nrowval");


  switch (what) {
    case GK_CSR_COL:
      gk_errexit(SIGERR, "This has not been implemented yet.\n");
      break;

    case GK_CSR_ROW:
      if (mat->rowptr == NULL) 
        gk_errexit(SIGERR, "Cannot filter rows when row-based structure has not been created.\n");

      nrowptr[0] = 0;
      for (nnz=0, i=0; i<nrows; i++) {
        avgwgt = zscore/(rowptr[i+1]-rowptr[i]);
        for (j=rowptr[i]; j<rowptr[i+1]; j++) {
          if (rowval[j] > avgwgt) {
            nrowind[nnz] = rowind[j];
            nrowval[nnz] = rowval[j];
            nnz++;
          }
        }
        nrowptr[i+1] = nnz;
      }
      break;

    default:
      gk_csr_Free(&nmat);
      gk_errexit(SIGERR, "Unknown prunning type of %d\n", what);
      return NULL;
  }

  return nmat;
}


/*************************************************************************/
/*! Compacts the column-space of the matrix by removing empty columns.
    As a result of the compaction, the column numbers are renumbered. 
    The compaction operation is done in place and only affects the row-based
    representation of the matrix.
    The new columns are ordered in decreasing frequency.
   
    \param mat the matrix whose empty columns will be removed.
*/
/**************************************************************************/
void gk_csr_CompactColumns(gk_csr_t *mat)
{
  ssize_t i;
  int nrows, ncols, nncols;
  ssize_t *rowptr;
  int *rowind, *colmap;
  gk_ikv_t *clens;

  nrows  = mat->nrows;
  ncols  = mat->ncols;
  rowptr = mat->rowptr;
  rowind = mat->rowind;

  colmap = gk_imalloc(ncols, "gk_csr_CompactColumns: colmap");

  clens = gk_ikvmalloc(ncols, "gk_csr_CompactColumns: clens");
  for (i=0; i<ncols; i++) {
    clens[i].key = 0;
    clens[i].val = i;
  }

  for (i=0; i<rowptr[nrows]; i++) 
    clens[rowind[i]].key++;
  gk_ikvsortd(ncols, clens);

  for (nncols=0, i=0; i<ncols; i++) {
    if (clens[i].key > 0) 
      colmap[clens[i].val] = nncols++;
    else
      break;
  }

  for (i=0; i<rowptr[nrows]; i++) 
    rowind[i] = colmap[rowind[i]];

  mat->ncols = nncols;

  gk_free((void **)&colmap, &clens, LTERM);
}


/*************************************************************************/
/*! Sorts the indices in increasing order
    \param mat the matrix itself,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating which set of
           indices to sort.
*/
/**************************************************************************/
void gk_csr_SortIndices(gk_csr_t *mat, int what)
{
  int n, nn=0;
  ssize_t *ptr;
  int *ind;
  float *val;

  switch (what) {
    case GK_CSR_ROW:
      if (!mat->rowptr)
        gk_errexit(SIGERR, "Row-based view of the matrix does not exists.\n");

      n   = mat->nrows;
      ptr = mat->rowptr;
      ind = mat->rowind;
      val = mat->rowval;
      break;

    case GK_CSR_COL:
      if (!mat->colptr)
        gk_errexit(SIGERR, "Column-based view of the matrix does not exists.\n");

      n   = mat->ncols;
      ptr = mat->colptr;
      ind = mat->colind;
      val = mat->colval;
      break;

    default:
      gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
      return;
  }

  #pragma omp parallel if (n > 100)
  {
    ssize_t i, j, k;
    gk_ikv_t *cand;
    float *tval;

    #pragma omp single
    for (i=0; i<n; i++) 
      nn = gk_max(nn, ptr[i+1]-ptr[i]);
  
    cand = gk_ikvmalloc(nn, "gk_csr_SortIndices: cand");
    tval = gk_fmalloc(nn, "gk_csr_SortIndices: tval");
  
    #pragma omp for schedule(static)
    for (i=0; i<n; i++) {
      for (k=0, j=ptr[i]; j<ptr[i+1]; j++) {
        if (j > ptr[i] && ind[j] < ind[j-1])
          k = 1; /* an inversion */
        cand[j-ptr[i]].val = j-ptr[i];
        cand[j-ptr[i]].key = ind[j];
        tval[j-ptr[i]]     = val[j];
      }
      if (k) {
        gk_ikvsorti(ptr[i+1]-ptr[i], cand);
        for (j=ptr[i]; j<ptr[i+1]; j++) {
          ind[j] = cand[j-ptr[i]].key;
          val[j] = tval[cand[j-ptr[i]].val];
        }
      }
    }

    gk_free((void **)&cand, &tval, LTERM);
  }

}


/*************************************************************************/
/*! Creates a row/column index from the column/row data.
    \param mat the matrix itself,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating which index
           will be created.
*/
/**************************************************************************/
void gk_csr_CreateIndex(gk_csr_t *mat, int what)
{
  /* 'f' stands for forward, 'r' stands for reverse */
  ssize_t i, j, k, nf, nr;
  ssize_t *fptr, *rptr;
  int *find, *rind;
  float *fval, *rval;

  switch (what) {
    case GK_CSR_COL:
      nf   = mat->nrows;
      fptr = mat->rowptr;
      find = mat->rowind;
      fval = mat->rowval;

      if (mat->colptr) gk_free((void **)&mat->colptr, LTERM);
      if (mat->colind) gk_free((void **)&mat->colind, LTERM);
      if (mat->colval) gk_free((void **)&mat->colval, LTERM);

      nr   = mat->ncols;
      rptr = mat->colptr = gk_zsmalloc(nr+1, 0, "gk_csr_CreateIndex: rptr");
      rind = mat->colind = gk_imalloc(fptr[nf], "gk_csr_CreateIndex: rind");
      rval = mat->colval = (fval ? gk_fmalloc(fptr[nf], "gk_csr_CreateIndex: rval") : NULL);
      break;
    case GK_CSR_ROW:
      nf   = mat->ncols;
      fptr = mat->colptr;
      find = mat->colind;
      fval = mat->colval;

      if (mat->rowptr) gk_free((void **)&mat->rowptr, LTERM);
      if (mat->rowind) gk_free((void **)&mat->rowind, LTERM);
      if (mat->rowval) gk_free((void **)&mat->rowval, LTERM);

      nr   = mat->nrows;
      rptr = mat->rowptr = gk_zsmalloc(nr+1, 0, "gk_csr_CreateIndex: rptr");
      rind = mat->rowind = gk_imalloc(fptr[nf], "gk_csr_CreateIndex: rind");
      rval = mat->rowval = (fval ? gk_fmalloc(fptr[nf], "gk_csr_CreateIndex: rval") : NULL);
      break;
    default:
      gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
      return;
  }


  for (i=0; i<nf; i++) {
    for (j=fptr[i]; j<fptr[i+1]; j++)
      rptr[find[j]]++;
  }
  MAKECSR(i, nr, rptr);
  
  if (rptr[nr] > 6*nr) {
    for (i=0; i<nf; i++) {
      for (j=fptr[i]; j<fptr[i+1]; j++) 
        rind[rptr[find[j]]++] = i;
    }
    SHIFTCSR(i, nr, rptr);

    if (fval) {
      for (i=0; i<nf; i++) {
        for (j=fptr[i]; j<fptr[i+1]; j++) 
          rval[rptr[find[j]]++] = fval[j];
      }
      SHIFTCSR(i, nr, rptr);
    }
  }
  else {
    if (fval) {
      for (i=0; i<nf; i++) {
        for (j=fptr[i]; j<fptr[i+1]; j++) {
          k = find[j];
          rind[rptr[k]]   = i;
          rval[rptr[k]++] = fval[j];
        }
      }
    }
    else {
      for (i=0; i<nf; i++) {
        for (j=fptr[i]; j<fptr[i+1]; j++) 
          rind[rptr[find[j]]++] = i;
      }
    }
    SHIFTCSR(i, nr, rptr);
  }
}


/*************************************************************************/
/*! Normalizes the rows/columns of the matrix to be unit 
    length.
    \param mat the matrix itself,
    \param what indicates what will be normalized and is obtained by
           specifying GK_CSR_ROW, GK_CSR_COL, GK_CSR_ROW|GK_CSR_COL. 
    \param norm indicates what norm is to normalize to, 1: 1-norm, 2: 2-norm
*/
/**************************************************************************/
void gk_csr_Normalize(gk_csr_t *mat, int what, int norm)
{
  ssize_t i, j;
  int n;
  ssize_t *ptr;
  float *val, sum;


  if (what&GK_CSR_ROW && mat->rowval) {
    n   = mat->nrows;
    ptr = mat->rowptr;
    val = mat->rowval;

    #pragma omp parallel for if (ptr[n] > OMPMINOPS) private(j,sum) schedule(static)
    for (i=0; i<n; i++) {
      sum = 0.0;
      if (norm == 1) {
        for (j=ptr[i]; j<ptr[i+1]; j++) 
          sum += val[j]; /* assume val[j] > 0 */ 
        if (sum > 0)
          sum = 1.0/sum;
      }
      else if (norm == 2) {
        for (j=ptr[i]; j<ptr[i+1]; j++) 
          sum += val[j]*val[j];
        if (sum > 0)
          sum = 1.0/sqrt(sum); 
      }
      for (j=ptr[i]; j<ptr[i+1]; j++)
        val[j] *= sum;
    }
  }

  if (what&GK_CSR_COL && mat->colval) {
    n   = mat->ncols;
    ptr = mat->colptr;
    val = mat->colval;

    #pragma omp parallel for if (ptr[n] > OMPMINOPS) private(j,sum) schedule(static)
    for (i=0; i<n; i++) {
      sum = 0.0;
      if (norm == 1) {
        for (j=ptr[i]; j<ptr[i+1]; j++) 
          sum += val[j]; /* assume val[j] > 0 */ 
        if (sum > 0)
          sum = 1.0/sum;
      }
      else if (norm == 2) {
        for (j=ptr[i]; j<ptr[i+1]; j++) 
          sum += val[j]*val[j];
        if (sum > 0)
          sum = 1.0/sqrt(sum); 
      }
      for (j=ptr[i]; j<ptr[i+1]; j++)
        val[j] *= sum;
    }
  }

}


/*************************************************************************/
/*! Applies different row scaling methods.
    \param mat the matrix itself,
    \param type indicates the type of row scaling. Possible values are:
           GK_CSR_MAXTF, GK_CSR_SQRT, GK_CSR_LOG, GK_CSR_IDF, GK_CSR_MAXTF2.
*/
/**************************************************************************/
void gk_csr_Scale(gk_csr_t *mat, int type)
{
  ssize_t i, j;
  int nrows, ncols, nnzcols, bgfreq;
  ssize_t *rowptr;
  int *rowind, *collen;
  float *rowval, *cscale, maxtf;
  double logscale = 1.0/log(2.0);

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  switch (type) {
    case GK_CSR_MAXTF: /* TF' = .5 + .5*TF/MAX(TF) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j, maxtf) schedule(static)
      for (i=0; i<nrows; i++) {
        maxtf = fabs(rowval[rowptr[i]]);
        for (j=rowptr[i]; j<rowptr[i+1]; j++) 
          maxtf = (maxtf < fabs(rowval[j]) ? fabs(rowval[j]) : maxtf);
  
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          rowval[j] = .5 + .5*rowval[j]/maxtf;
      }
      break;

    case GK_CSR_MAXTF2: /* TF' = .1 + .9*TF/MAX(TF) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j, maxtf) schedule(static)
      for (i=0; i<nrows; i++) {
        maxtf = fabs(rowval[rowptr[i]]);
        for (j=rowptr[i]; j<rowptr[i+1]; j++) 
          maxtf = (maxtf < fabs(rowval[j]) ? fabs(rowval[j]) : maxtf);
  
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          rowval[j] = .1 + .9*rowval[j]/maxtf;
      }
      break;

    case GK_CSR_SQRT: /* TF' = .1+SQRT(TF) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = .1+sign(rowval[j], sqrt(fabs(rowval[j])));
        }
      }
      
      break;

    case GK_CSR_POW25: /* TF' = .1+POW(TF,.25) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = .1+sign(rowval[j], sqrt(sqrt(fabs(rowval[j]))));
        }
      }
      break;

    case GK_CSR_POW65: /* TF' = .1+POW(TF,.65) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = .1+sign(rowval[j], powf(fabs(rowval[j]), .65));
        }
      }
      break;

    case GK_CSR_POW75: /* TF' = .1+POW(TF,.75) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = .1+sign(rowval[j], powf(fabs(rowval[j]), .75));
        }
      }
      break;

    case GK_CSR_POW85: /* TF' = .1+POW(TF,.85) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = .1+sign(rowval[j], powf(fabs(rowval[j]), .85));
        }
      }
      break;

    case GK_CSR_LOG: /* TF' = 1+log_2(TF) */
      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) schedule(static,32)
      for (i=0; i<rowptr[nrows]; i++) {
        if (rowval[i] != 0.0)
          rowval[i] = 1+(rowval[i]>0.0 ? log(rowval[i]) : -log(-rowval[i]))*logscale;
      }
#ifdef XXX
      #pragma omp parallel for private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++) { 
          if (rowval[j] != 0.0)
            rowval[j] = 1+(rowval[j]>0.0 ? log(rowval[j]) : -log(-rowval[j]))*logscale;
            //rowval[j] = 1+sign(rowval[j], log(fabs(rowval[j]))*logscale);
        }
      }
#endif
      break;

    case GK_CSR_IDF: /* TF' = TF*IDF */
      ncols  = mat->ncols;
      cscale = gk_fmalloc(ncols, "gk_csr_Scale: cscale");
      collen = gk_ismalloc(ncols, 0, "gk_csr_Scale: collen");

      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          collen[rowind[j]]++;
      }

      #pragma omp parallel for if (ncols > OMPMINOPS) schedule(static)
      for (i=0; i<ncols; i++)
        cscale[i] = (collen[i] > 0 ? log(1.0*nrows/collen[i]) : 0.0);

      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          rowval[j] *= cscale[rowind[j]];
      }
      
      gk_free((void **)&cscale, &collen, LTERM);
      break;

    case GK_CSR_IDF2: /* TF' = TF*IDF */
      ncols  = mat->ncols;
      cscale = gk_fmalloc(ncols, "gk_csr_Scale: cscale");
      collen = gk_ismalloc(ncols, 0, "gk_csr_Scale: collen");

      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          collen[rowind[j]]++;
      }

      nnzcols = 0;
      #pragma omp parallel for if (ncols > OMPMINOPS) schedule(static) reduction(+:nnzcols)
      for (i=0; i<ncols; i++)
        nnzcols += (collen[i] > 0 ? 1 : 0);

      bgfreq = gk_max(10, (ssize_t)(.5*rowptr[nrows]/nnzcols));
      printf("nnz: %zd, nnzcols: %d, bgfreq: %d\n", rowptr[nrows], nnzcols, bgfreq);

      #pragma omp parallel for if (ncols > OMPMINOPS) schedule(static)
      for (i=0; i<ncols; i++)
        cscale[i] = (collen[i] > 0 ? log(1.0*(nrows+2*bgfreq)/(bgfreq+collen[i])) : 0.0);

      #pragma omp parallel for if (rowptr[nrows] > OMPMINOPS) private(j) schedule(static)
      for (i=0; i<nrows; i++) {
        for (j=rowptr[i]; j<rowptr[i+1]; j++)
          rowval[j] *= cscale[rowind[j]];
      }

      gk_free((void **)&cscale, &collen, LTERM);
      break;

    default:
      gk_errexit(SIGERR, "Unknown scaling type of %d\n", type);
  }
}


/*************************************************************************/
/*! Computes the sums of the rows/columns
    \param mat the matrix itself,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating which 
           sums to compute.
*/
/**************************************************************************/
void gk_csr_ComputeSums(gk_csr_t *mat, int what)
{
  ssize_t i;
  int n;
  ssize_t *ptr;
  float *val, *sums;

  switch (what) {
    case GK_CSR_ROW:
      n   = mat->nrows;
      ptr = mat->rowptr;
      val = mat->rowval;

      if (mat->rsums) 
        gk_free((void **)&mat->rsums, LTERM);

      sums = mat->rsums = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: sums");
      break;
    case GK_CSR_COL:
      n   = mat->ncols;
      ptr = mat->colptr;
      val = mat->colval;

      if (mat->csums) 
        gk_free((void **)&mat->csums, LTERM);

      sums = mat->csums = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: sums");
      break;
    default:
      gk_errexit(SIGERR, "Invalid sum type of %d.\n", what);
      return;
  }

  if (val) {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      sums[i] = gk_fsum(ptr[i+1]-ptr[i], val+ptr[i], 1);
  }
  else {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      sums[i] = ptr[i+1]-ptr[i];
  }
}


/*************************************************************************/
/*! Computes the norms of the rows/columns

    \param mat the matrix itself,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating which 
           squared norms to compute.

    \note If the rowval/colval arrays are NULL, the matrix is assumed
          to be binary and the norms are computed accordingly.
*/
/**************************************************************************/
void gk_csr_ComputeNorms(gk_csr_t *mat, int what)
{
  ssize_t i;
  int n;
  ssize_t *ptr;
  float *val, *norms;

  switch (what) {
    case GK_CSR_ROW:
      n   = mat->nrows;
      ptr = mat->rowptr;
      val = mat->rowval;

      if (mat->rnorms) gk_free((void **)&mat->rnorms, LTERM);

      norms = mat->rnorms = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: norms");
      break;
    case GK_CSR_COL:
      n   = mat->ncols;
      ptr = mat->colptr;
      val = mat->colval;

      if (mat->cnorms) gk_free((void **)&mat->cnorms, LTERM);

      norms = mat->cnorms = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: norms");
      break;
    default:
      gk_errexit(SIGERR, "Invalid norm type of %d.\n", what);
      return;
  }

  if (val) {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      norms[i] = sqrt(gk_fdot(ptr[i+1]-ptr[i], val+ptr[i], 1, val+ptr[i], 1));
  }
  else {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      norms[i] = sqrt(ptr[i+1]-ptr[i]);
  }
}


/*************************************************************************/
/*! Computes the squared of the norms of the rows/columns

    \param mat the matrix itself,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating which 
           squared norms to compute.

    \note If the rowval/colval arrays are NULL, the matrix is assumed
          to be binary and the norms are computed accordingly.
*/
/**************************************************************************/
void gk_csr_ComputeSquaredNorms(gk_csr_t *mat, int what)
{
  ssize_t i;
  int n;
  ssize_t *ptr;
  float *val, *norms;

  switch (what) {
    case GK_CSR_ROW:
      n   = mat->nrows;
      ptr = mat->rowptr;
      val = mat->rowval;

      if (mat->rnorms) gk_free((void **)&mat->rnorms, LTERM);

      norms = mat->rnorms = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: norms");
      break;
    case GK_CSR_COL:
      n   = mat->ncols;
      ptr = mat->colptr;
      val = mat->colval;

      if (mat->cnorms) gk_free((void **)&mat->cnorms, LTERM);

      norms = mat->cnorms = gk_fsmalloc(n, 0, "gk_csr_ComputeSums: norms");
      break;
    default:
      gk_errexit(SIGERR, "Invalid norm type of %d.\n", what);
      return;
  }

  if (val) {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      norms[i] = gk_fdot(ptr[i+1]-ptr[i], val+ptr[i], 1, val+ptr[i], 1);
  }
  else {
    #pragma omp parallel for if (ptr[n] > OMPMINOPS) schedule(static)
    for (i=0; i<n; i++) 
      norms[i] = ptr[i+1]-ptr[i];
  }
}


/*************************************************************************/
/*! Returns a new matrix whose rows/columns are shuffled.
   
    \param mat the matrix to be shuffled,
    \param what indicates if the rows (GK_CSR_ROW), columns (GK_CSR_COL),
           or both (GK_CSR_ROWCOL) will be shuffled,
    \param symmetric indicates if the same shuffling will be applied to 
           both rows and columns. This is valid with nrows==ncols and 
           GK_CSR_ROWCOL was specified.
    \returns the shuffled matrix. 
          The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_Shuffle(gk_csr_t *mat, int what, int symmetric)
{
  ssize_t i, j;
  int nrows, ncols;
  ssize_t *rowptr, *nrowptr;
  int *rowind, *nrowind;
  int *rperm, *cperm;
  float *rowval, *nrowval;
  gk_csr_t *nmat;

  if (what == GK_CSR_ROWCOL && symmetric && mat->nrows != mat->ncols)
    gk_errexit(SIGERR, "The matrix is not square for a symmetric rowcol shuffling.\n");

  nrows  = mat->nrows;
  ncols  = mat->ncols;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  rperm = gk_imalloc(nrows, "gk_csr_Shuffle: rperm");
  cperm = gk_imalloc(ncols, "gk_csr_Shuffle: cperm");

  switch (what) {
    case GK_CSR_ROW:
      gk_RandomPermute(nrows, rperm, 1);
      for (i=0; i<20; i++)
        gk_RandomPermute(nrows, rperm, 0);

      for (i=0; i<ncols; i++)
        cperm[i] = i;
      break;

    case GK_CSR_COL:
      gk_RandomPermute(ncols, cperm, 1);
      for (i=0; i<20; i++)
        gk_RandomPermute(ncols, cperm, 0);

      for (i=0; i<nrows; i++)
        rperm[i] = i;
      break;

    case GK_CSR_ROWCOL:
      gk_RandomPermute(nrows, rperm, 1);
      for (i=0; i<20; i++)
        gk_RandomPermute(nrows, rperm, 0);

      if (symmetric)
        gk_icopy(nrows, rperm, cperm);
      else {
        gk_RandomPermute(ncols, cperm, 1);
        for (i=0; i<20; i++)
          gk_RandomPermute(ncols, cperm, 0);
      }
      break;

    default:
      gk_free((void **)&rperm, &cperm, LTERM);
      gk_errexit(SIGERR, "Unknown shuffling type of %d\n", what);
      return NULL;
  }

  nmat = gk_csr_Create();
  nmat->nrows = nrows;
  nmat->ncols = ncols;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_Shuffle: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(rowptr[nrows], "gk_csr_Shuffle: nrowind");
  nrowval = nmat->rowval = (rowval ? gk_fmalloc(rowptr[nrows], "gk_csr_Shuffle: nrowval") : NULL) ;

  for (i=0; i<nrows; i++)
    nrowptr[rperm[i]] = rowptr[i+1]-rowptr[i];
  MAKECSR(i, nrows, nrowptr);

  for (i=0; i<nrows; i++) {
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      nrowind[nrowptr[rperm[i]]] = cperm[rowind[j]];
      if (nrowval)
        nrowval[nrowptr[rperm[i]]] = rowval[j];
      nrowptr[rperm[i]]++;
    }
  }
  SHIFTCSR(i, nrows, nrowptr);

  gk_free((void **)&rperm, &cperm, LTERM);

  return nmat;

}


/*************************************************************************/
/*! Returns the transpose of the matrix.
   
    \param mat the matrix to be transposed,
    \returns the transposed matrix. 
          The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_Transpose(gk_csr_t *mat)
{
  int nrows, ncols;
  ssize_t *colptr;
  int32_t *colind;
  float *colval;
  gk_csr_t *nmat;

  colptr = mat->colptr;
  colind = mat->colind;
  colval = mat->colval;

  mat->colptr = NULL;
  mat->colind = NULL;
  mat->colval = NULL;

  gk_csr_CreateIndex(mat, GK_CSR_COL);

  nmat = gk_csr_Create();
  nmat->nrows  = mat->ncols;
  nmat->ncols  = mat->nrows;
  nmat->rowptr = mat->colptr;
  nmat->rowind = mat->colind;
  nmat->rowval = mat->colval;

  mat->colptr = colptr;
  mat->colind = colind;
  mat->colval = colval;

  return nmat;

}


/*************************************************************************/
/*! Computes the similarity between two rows/columns

    \param mat the matrix itself. The routine assumes that the indices
           are sorted in increasing order.
    \param i1 is the first row/column,
    \param i2 is the second row/column,
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating the type of
           objects between the similarity will be computed,
    \param simtype is the type of similarity and is one of GK_CSR_COS,
           GK_CSR_JAC, GK_CSR_MIN, GK_CSR_AMIN
    \returns the similarity between the two rows/columns.
*/
/**************************************************************************/
float gk_csr_ComputeSimilarity(gk_csr_t *mat, int i1, int i2, int what, 
          int simtype)
{
  int nind1, nind2;
  int *ind1, *ind2;
  float *val1, *val2, stat1, stat2, sim;

  switch (what) {
    case GK_CSR_ROW:
      if (!mat->rowptr)
        gk_errexit(SIGERR, "Row-based view of the matrix does not exists.\n");
      nind1 = mat->rowptr[i1+1]-mat->rowptr[i1];
      nind2 = mat->rowptr[i2+1]-mat->rowptr[i2];
      ind1  = mat->rowind + mat->rowptr[i1];
      ind2  = mat->rowind + mat->rowptr[i2];
      val1  = mat->rowval + mat->rowptr[i1];
      val2  = mat->rowval + mat->rowptr[i2];
      break;

    case GK_CSR_COL:
      if (!mat->colptr)
        gk_errexit(SIGERR, "Column-based view of the matrix does not exists.\n");
      nind1 = mat->colptr[i1+1]-mat->colptr[i1];
      nind2 = mat->colptr[i2+1]-mat->colptr[i2];
      ind1  = mat->colind + mat->colptr[i1];
      ind2  = mat->colind + mat->colptr[i2];
      val1  = mat->colval + mat->colptr[i1];
      val2  = mat->colval + mat->colptr[i2];
      break;

    default:
      gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
      return 0.0;
  }


  switch (simtype) {
    case GK_CSR_COS:
    case GK_CSR_JAC:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2]*val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1]*val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1]*val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2]*val2[i2];
          i2++;
        }
        else {
          sim   += val1[i1]*val2[i2];
          stat1 += val1[i1]*val1[i1];
          stat2 += val2[i2]*val2[i2];
          i1++;
          i2++;
        }
      }
      if (simtype == GK_CSR_COS)
        sim = (stat1*stat2 > 0.0 ? sim/sqrt(stat1*stat2) : 0.0);
      else 
        sim = (stat1+stat2-sim > 0.0 ? sim/(stat1+stat2-sim) : 0.0);
      break;

    case GK_CSR_MIN:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2];
          i2++;
        }
        else {
          sim   += gk_min(val1[i1],val2[i2]);
          stat1 += val1[i1];
          stat2 += val2[i2];
          i1++;
          i2++;
        }
      }
      sim = (stat1+stat2-sim > 0.0 ? sim/(stat1+stat2-sim) : 0.0);

      break;

    case GK_CSR_AMIN:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2];
          i2++;
        }
        else {
          sim   += gk_min(val1[i1],val2[i2]);
          stat1 += val1[i1];
          stat2 += val2[i2];
          i1++;
          i2++;
        }
      }
      sim = (stat1 > 0.0 ? sim/stat1 : 0.0);

      break;

    default:
      gk_errexit(SIGERR, "Unknown similarity measure %d\n", simtype);
      return -1;
  }

  return sim;

}


/*************************************************************************/
/*! Computes the similarity between two rows/columns

    \param mat_a the first matrix. The routine assumes that the indices
           are sorted in increasing order.
    \param mat_b the second matrix. The routine assumes that the indices
           are sorted in increasing order.
    \param i1 is the row/column from the first matrix (mat_a),
    \param i2 is the row/column from the second matrix (mat_b),
    \param what is either GK_CSR_ROW or GK_CSR_COL indicating the type of
           objects between the similarity will be computed,
    \param simtype is the type of similarity and is one of GK_CSR_COS,
           GK_CSR_JAC, GK_CSR_MIN, GK_CSR_AMIN
    \returns the similarity between the two rows/columns.
*/
/**************************************************************************/
float gk_csr_ComputePairSimilarity(gk_csr_t *mat_a, gk_csr_t *mat_b, 
          int i1, int i2, int what, int simtype)
{
  int nind1, nind2;
  int *ind1, *ind2;
  float *val1, *val2, stat1, stat2, sim;

  switch (what) {
    case GK_CSR_ROW:
      if (!mat_a->rowptr || !mat_b->rowptr)
        gk_errexit(SIGERR, "Row-based view of the matrix does not exists.\n");
      nind1 = mat_a->rowptr[i1+1]-mat_a->rowptr[i1];
      nind2 = mat_b->rowptr[i2+1]-mat_b->rowptr[i2];
      ind1  = mat_a->rowind + mat_a->rowptr[i1];
      ind2  = mat_b->rowind + mat_b->rowptr[i2];
      val1  = mat_a->rowval + mat_a->rowptr[i1];
      val2  = mat_b->rowval + mat_b->rowptr[i2];
      break;

    case GK_CSR_COL:
      if (!mat_a->colptr || !mat_b->colptr)
        gk_errexit(SIGERR, "Column-based view of the matrix does not exists.\n");
      nind1 = mat_a->colptr[i1+1]-mat_a->colptr[i1];
      nind2 = mat_b->colptr[i2+1]-mat_b->colptr[i2];
      ind1  = mat_a->colind + mat_a->colptr[i1];
      ind2  = mat_b->colind + mat_b->colptr[i2];
      val1  = mat_a->colval + mat_a->colptr[i1];
      val2  = mat_b->colval + mat_b->colptr[i2];
      break;

    default:
      gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
      return 0.0;
  }


  switch (simtype) {
    case GK_CSR_COS:
    case GK_CSR_JAC:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2]*val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1]*val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1]*val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2]*val2[i2];
          i2++;
        }
        else {
          sim   += val1[i1]*val2[i2];
          stat1 += val1[i1]*val1[i1];
          stat2 += val2[i2]*val2[i2];
          i1++;
          i2++;
        }
      }
      if (simtype == GK_CSR_COS)
        sim = (stat1*stat2 > 0.0 ? sim/sqrt(stat1*stat2) : 0.0);
      else 
        sim = (stat1+stat2-sim > 0.0 ? sim/(stat1+stat2-sim) : 0.0);
      break;

    case GK_CSR_MIN:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2];
          i2++;
        }
        else {
          sim   += gk_min(val1[i1],val2[i2]);
          stat1 += val1[i1];
          stat2 += val2[i2];
          i1++;
          i2++;
        }
      }
      sim = (stat1+stat2-sim > 0.0 ? sim/(stat1+stat2-sim) : 0.0);

      break;

    case GK_CSR_AMIN:
      sim = stat1 = stat2 = 0.0;
      i1 = i2 = 0;
      while (i1<nind1 && i2<nind2) {
        if (i1 == nind1) {
          stat2 += val2[i2];
          i2++;
        }
        else if (i2 == nind2) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] < ind2[i2]) {
          stat1 += val1[i1];
          i1++;
        }
        else if (ind1[i1] > ind2[i2]) {
          stat2 += val2[i2];
          i2++;
        }
        else {
          sim   += gk_min(val1[i1],val2[i2]);
          stat1 += val1[i1];
          stat2 += val2[i2];
          i1++;
          i2++;
        }
      }
      sim = (stat1 > 0.0 ? sim/stat1 : 0.0);

      break;

    default:
      gk_errexit(SIGERR, "Unknown similarity measure %d\n", simtype);
      return -1;
  }

  return sim;

}

/*************************************************************************/
/*! Finds the n most similar rows (neighbors) to the query.

    \param mat the matrix itself
    \param nqterms is the number of columns in the query
    \param qind is the list of query columns
    \param qval is the list of correspodning query weights
    \param simtype is the type of similarity and is one of GK_CSR_DOTP,
           GK_CSR_COS, GK_CSR_JAC, GK_CSR_MIN, GK_CSR_AMIN. In case of 
           GK_CSR_COS, the rows and the query are assumed to be of unit 
           length.
    \param nsim is the maximum number of requested most similar rows.
           If -1 is provided, then everything is returned unsorted.
    \param minsim is the minimum similarity of the requested most 
           similar rows
    \param hits is the result set. This array should be at least
           of length nsim.
    \param i_marker is an array of size equal to the number of rows
           whose values are initialized to -1. If NULL is provided
           then this array is allocated and freed internally.
    \param i_cand is an array of size equal to the number of rows.
           If NULL is provided then this array is allocated and freed 
           internally.
    \returns The number of identified most similar rows, which can be
             smaller than the requested number of nnbrs in those cases
             in which there are no sufficiently many neighbors.
*/
/**************************************************************************/
int gk_csr_GetSimilarRows(gk_csr_t *mat, int nqterms, int *qind, 
        float *qval, int simtype, int nsim, float minsim, gk_fkv_t *hits, 
        int *i_marker, gk_fkv_t *i_cand)
{
  ssize_t i, ii, j, k;
  int nrows, ncols, ncand;
  ssize_t *colptr;
  int *colind, *marker;
  float *colval, *rnorms, mynorm, *rsums, mysum;
  gk_fkv_t *cand;

  if (nqterms == 0)
    return 0;

  nrows  = mat->nrows;
  ncols  = mat->ncols;
  GKASSERT((colptr = mat->colptr) != NULL);
  GKASSERT((colind = mat->colind) != NULL);
  GKASSERT((colval = mat->colval) != NULL);

  marker = (i_marker ? i_marker : gk_ismalloc(nrows, -1, "gk_csr_SimilarRows: marker"));
  cand   = (i_cand   ? i_cand   : gk_fkvmalloc(nrows, "gk_csr_SimilarRows: cand"));

  switch (simtype) {
    case GK_CSR_DOTP:
    case GK_CSR_COS:
      for (ncand=0, ii=0; ii<nqterms; ii++) {
        i = qind[ii];
        if (i < ncols) {
          for (j=colptr[i]; j<colptr[i+1]; j++) {
            k = colind[j];
            if (marker[k] == -1) {
              cand[ncand].val = k;
              cand[ncand].key = 0;
              marker[k]       = ncand++;
            }
            cand[marker[k]].key += colval[j]*qval[ii];
          }
        }
      }
      break;

    case GK_CSR_JAC:
      for (ncand=0, ii=0; ii<nqterms; ii++) {
        i = qind[ii];
        if (i < ncols) {
          for (j=colptr[i]; j<colptr[i+1]; j++) {
            k = colind[j];
            if (marker[k] == -1) {
              cand[ncand].val = k;
              cand[ncand].key = 0;
              marker[k]       = ncand++;
            }
            cand[marker[k]].key += colval[j]*qval[ii];
          }
        }
      }

      GKASSERT((rnorms = mat->rnorms) != NULL);
      mynorm = gk_fdot(nqterms, qval, 1, qval, 1);

      for (i=0; i<ncand; i++)
        cand[i].key = cand[i].key/(rnorms[cand[i].val]+mynorm-cand[i].key);
      break;

    case GK_CSR_MIN:
      for (ncand=0, ii=0; ii<nqterms; ii++) {
        i = qind[ii];
        if (i < ncols) {
          for (j=colptr[i]; j<colptr[i+1]; j++) {
            k = colind[j];
            if (marker[k] == -1) {
              cand[ncand].val = k;
              cand[ncand].key = 0;
              marker[k]       = ncand++;
            }
            cand[marker[k]].key += gk_min(colval[j], qval[ii]);
          }
        }
      }

      GKASSERT((rsums = mat->rsums) != NULL);
      mysum = gk_fsum(nqterms, qval, 1);

      for (i=0; i<ncand; i++)
        cand[i].key = cand[i].key/(rsums[cand[i].val]+mysum-cand[i].key);
      break;

    /* Assymetric MIN  similarity */
    case GK_CSR_AMIN:
      for (ncand=0, ii=0; ii<nqterms; ii++) {
        i = qind[ii];
        if (i < ncols) {
          for (j=colptr[i]; j<colptr[i+1]; j++) {
            k = colind[j];
            if (marker[k] == -1) {
              cand[ncand].val = k;
              cand[ncand].key = 0;
              marker[k]       = ncand++;
            }
            cand[marker[k]].key += gk_min(colval[j], qval[ii]);
          }
        }
      }

      mysum = gk_fsum(nqterms, qval, 1);

      for (i=0; i<ncand; i++)
        cand[i].key = cand[i].key/mysum;
      break;

    default:
      gk_errexit(SIGERR, "Unknown similarity measure %d\n", simtype);
      return -1;
  }

  /* go and prune the hits that are bellow minsim */
  for (j=0, i=0; i<ncand; i++) {
    marker[cand[i].val] = -1;
    if (cand[i].key >= minsim) 
      cand[j++] = cand[i];
  }
  ncand = j;

  if (nsim == -1 || nsim >= ncand) {
    nsim = ncand;
  }
  else {
    nsim = gk_min(nsim, ncand);
    gk_dfkvkselect(ncand, nsim, cand);
    gk_fkvsortd(nsim, cand);
  }

  gk_fkvcopy(nsim, cand, hits);

  if (i_marker == NULL)
    gk_free((void **)&marker, LTERM);
  if (i_cand == NULL)
    gk_free((void **)&cand, LTERM);

  return nsim;
}


/*************************************************************************/
/*! Returns a symmetric version of a square matrix. The symmetric version
    is constructed by applying an A op A^T operation, where op is one of
    GK_CSR_SYM_SUM, GK_CSR_SYM_MIN, GK_CSR_SYM_MAX, GK_CSR_SYM_AVG.
   
    \param mat the matrix to be symmetrized,
    \param op indicates the operation to be performed. The possible values are
           GK_CSR_SYM_SUM, GK_CSR_SYM_MIN, GK_CSR_SYM_MAX, and GK_CSR_SYM_AVG.

    \returns the symmetrized matrix consisting only of its row-based structure. 
          The input matrix is not modified. 
*/
/**************************************************************************/
gk_csr_t *gk_csr_MakeSymmetric(gk_csr_t *mat, int op)
{
  ssize_t i, j, k, nnz;
  int nrows, nadj, hasvals;
  ssize_t *rowptr, *colptr, *nrowptr;
  int *rowind, *colind, *nrowind, *marker, *ids;
  float *rowval=NULL, *colval=NULL, *nrowval=NULL, *wgts=NULL;
  gk_csr_t *nmat;

  if (mat->nrows != mat->ncols) {
    fprintf(stderr, "gk_csr_MakeSymmetric: The matrix needs to be square.\n");
    return NULL;
  }

  hasvals = (mat->rowval != NULL);

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  if (hasvals)
    rowval = mat->rowval;

  /* create the column view for efficient processing */
  colptr = gk_zsmalloc(nrows+1, 0, "colptr");
  colind = gk_i32malloc(rowptr[nrows], "colind");
  if (hasvals)
    colval = gk_fmalloc(rowptr[nrows], "colval");

  for (i=0; i<nrows; i++) {
    for (j=rowptr[i]; j<rowptr[i+1]; j++) 
      colptr[rowind[j]]++;
  }
  MAKECSR(i, nrows, colptr);

  for (i=0; i<nrows; i++) {
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      colind[colptr[rowind[j]]] = i;
      if (hasvals)
        colval[colptr[rowind[j]]] = rowval[j];
      colptr[rowind[j]]++;
    }
  }
  SHIFTCSR(i, nrows, colptr);


  nmat = gk_csr_Create();
  
  nmat->nrows = mat->nrows;
  nmat->ncols = mat->ncols;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_MakeSymmetric: nrowptr");
  nrowind = nmat->rowind = gk_imalloc(2*rowptr[nrows], "gk_csr_MakeSymmetric: nrowind");
  if (hasvals)
    nrowval = nmat->rowval = gk_fmalloc(2*rowptr[nrows], "gk_csr_MakeSymmetric: nrowval");

  marker = gk_ismalloc(nrows, -1, "marker");
  ids    = gk_imalloc(nrows, "ids");
  if (hasvals)
    wgts = gk_fmalloc(nrows, "wgts");

  nrowptr[0] = nnz = 0;
  for (i=0; i<nrows; i++) {
    nadj = 0;
    /* out-edges */
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      ids[nadj] = rowind[j]; 
      if (hasvals)
        wgts[nadj] = (op == GK_CSR_SYM_AVG ? 0.5*rowval[j] : rowval[j]);
      marker[rowind[j]] = nadj++;
    }

    /* in-edges */
    for (j=colptr[i]; j<colptr[i+1]; j++) {
      if (marker[colind[j]] == -1) {
        if (op != GK_CSR_SYM_MIN) {
          ids[nadj] = colind[j]; 
          if (hasvals) 
            wgts[nadj] = (op == GK_CSR_SYM_AVG ? 0.5*colval[j] : colval[j]);
          nadj++;
        }
      }
      else {
        if (hasvals) {
          switch (op) {
            case GK_CSR_SYM_MAX:
              wgts[marker[colind[j]]] = gk_max(colval[j], wgts[marker[colind[j]]]);
              break;
            case GK_CSR_SYM_MIN:
              wgts[marker[colind[j]]] = gk_min(colval[j], wgts[marker[colind[j]]]);
              break;
            case GK_CSR_SYM_SUM:
              wgts[marker[colind[j]]] += colval[j];
              break;
            case GK_CSR_SYM_AVG:
              wgts[marker[colind[j]]] = 0.5*(wgts[marker[colind[j]]] + colval[j]);
              break;
            default:
              errexit("Unsupported op for MakeSymmetric!\n");
          }
        }
        marker[colind[j]] = -1;
      }
    }

    /* go over out edges again to resolve any edges that were not found in the in
     * edges */
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      if (marker[rowind[j]] != -1) {
        if (op == GK_CSR_SYM_MIN)
          ids[marker[rowind[j]]] = -1;
        marker[rowind[j]] = -1;
      }
    }

    /* put the non '-1' entries in ids[] into i's row */
    for (j=0; j<nadj; j++) {
      if (ids[j] != -1) {
        nrowind[nnz] = ids[j];
        if (hasvals)
          nrowval[nnz] = wgts[j];
        nnz++;
      }
    }
    nrowptr[i+1] = nnz;
  }

  gk_free((void **)&colptr, &colind, &colval, &marker, &ids, &wgts, LTERM);

  return nmat;
}


/*************************************************************************/
/*! This function finds the connected components in a graph stored in
    CSR format.

    \param mat is the graph structure in CSR format
    \param cptr is the ptr structure of the CSR representation of the 
           components. The length of this vector must be mat->nrows+1.
    \param cind is the indices structure of the CSR representation of 
           the components. The length of this vector must be mat->nrows.
    \param cids is an array that stores the component # of each vertex
           of the graph. The length of this vector must be mat->nrows.

    \returns the number of components that it found.

    \note The cptr, cind, and cids parameters can be NULL, in which case 
          only the number of connected components is returned.
*/
/*************************************************************************/
int gk_csr_FindConnectedComponents(gk_csr_t *mat, int32_t *cptr, int32_t *cind, 
        int32_t *cids)
{
  ssize_t i, ii, j, jj, k, nvtxs, first, last, ntodo, ncmps;
  ssize_t *xadj;
  int32_t *adjncy, *pos, *todo;
  int32_t mustfree_ccsr=0, mustfree_where=0;

  if (mat->nrows != mat->ncols) {
    fprintf(stderr, "gk_csr_FindComponents: The matrix needs to be square.\n");
    return -1;
  }

  nvtxs  = mat->nrows;
  xadj   = mat->rowptr;
  adjncy = mat->rowind;

  /* Deal with NULL supplied cptr/cind vectors */
  if (cptr == NULL) {
    cptr = gk_i32malloc(nvtxs+1, "gk_csr_FindComponents: cptr");
    cind = gk_i32malloc(nvtxs, "gk_csr_FindComponents: cind");
    mustfree_ccsr = 1;
  }

  /* The list of vertices that have not been touched yet. 
     The valid entries are from [0..ntodo). */
  todo = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_csr_FindComponents: todo"));

  /* For a vertex that has not been visited, pos[i] is the position in the
     todo list that this vertex is stored. 
     If a vertex has been visited, pos[i] = -1. */
  pos = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_csr_FindComponents: pos"));


  /* Find the connected componends */
  ncmps = -1;
  ntodo = nvtxs;     /* All vertices have not been visited */
  first = last = 0;  /* Point to the first and last vertices that have been touched
                        but not explored. 
                        These vertices are stored in cind[first]...cind[last-1]. */

  while (first < last || ntodo > 0) {
    if (first == last) { /* Find another starting vertex */
      cptr[++ncmps] = first;  /* Mark the end of the current CC */

      /* put the first vertex in the todo list as the start of the new CC */
      ASSERT(pos[todo[0]] != -1);
      cind[last++] = todo[0];  

      pos[todo[0]] = -1;
      todo[0] = todo[--ntodo];
      pos[todo[0]] = 0;
    }

    i = cind[first++];  /* Get the first visited but unexplored vertex */

    for (j=xadj[i]; j<xadj[i+1]; j++) {
      k = adjncy[j];
      if (pos[k] != -1) {
        cind[last++] = k;

        /* Remove k from the todo list and put the last item in the todo 
           list at the position that k was so that the todo list will be
           consequtive. The pos[] array is updated accordingly to keep track
           the location of the vertices in the todo[] list. */
        todo[pos[k]] = todo[--ntodo];
        pos[todo[pos[k]]] = pos[k];
        pos[k] = -1;
      }
    }
  }
  cptr[++ncmps] = first;

  /* see if we need to return cids */
  if (cids != NULL) {
    for (i=0; i<ncmps; i++) {
      for (j=cptr[i]; j<cptr[i+1]; j++)
        cids[cind[j]] = i;
    }
  }

  if (mustfree_ccsr)
    gk_free((void **)&cptr, &cind, LTERM);

  gk_free((void **)&pos, &todo, LTERM);

  return (int) ncmps;
}


/*************************************************************************/
/*! Returns a matrix that has been reordered according to the provided
    row/column permutation. The matrix is required to be square and the same
    permutation is applied to both rows and columns.

    \param[IN] mat is the matrix to be re-ordered.
    \param[IN] perm is the new ordering of the rows & columns
    \param[IN] iperm is the original ordering of the re-ordered matrix's rows & columns
    \returns the newly created reordered matrix.

    \note Either perm or iperm can be NULL but not both.
*/
/**************************************************************************/
gk_csr_t *gk_csr_ReorderSymmetric(gk_csr_t *mat, int32_t *perm, int32_t *iperm)
{
  ssize_t j, jj;
  ssize_t *rowptr, *nrowptr;
  int i, k, u, v, nrows;
  int freeperm=0, freeiperm=0;
  int32_t *rowind, *nrowind;
  float *rowval, *nrowval;
  gk_csr_t *nmat;

  if (mat->nrows != mat->ncols) {
    fprintf(stderr, "gk_csr_ReorderSymmetric: The matrix needs to be square.\n");
    return NULL;
  }

  if (perm == NULL && iperm == NULL)
    return NULL;

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  nmat = gk_csr_Create();

  nmat->nrows = nrows;
  nmat->ncols = nrows;

  nrowptr = nmat->rowptr = gk_zmalloc(nrows+1, "gk_csr_ReorderSymmetric: rowptr");
  nrowind = nmat->rowind = gk_i32malloc(rowptr[nrows], "gk_csr_ReorderSymmetric: rowind");
  nrowval = nmat->rowval = gk_fmalloc(rowptr[nrows], "gk_csr_ReorderSymmetric: rowval");

  /* allocate memory for the different structures present in the matrix */
  if (mat->rlabels)
    nmat->rlabels = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: rlabels");
  if (mat->rmap)
    nmat->rmap = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: rmap");
  if (mat->rnorms)
    nmat->rnorms = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: rnorms");
  if (mat->rsums)
    nmat->rsums = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: rsums");
  if (mat->rsizes)
    nmat->rsizes = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: rsizes");
  if (mat->rvols)
    nmat->rvols = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: rvols");
  if (mat->rwgts)
    nmat->rwgts = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: rwgts");

  if (mat->clabels)
    nmat->clabels = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: clabels");
  if (mat->cmap)
    nmat->cmap = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: cmap");
  if (mat->cnorms)
    nmat->cnorms = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: cnorms");
  if (mat->csums)
    nmat->csums = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: csums");
  if (mat->csizes)
    nmat->csizes = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: csizes");
  if (mat->cvols)
    nmat->cvols = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: cvols");
  if (mat->cwgts)
    nmat->cwgts = gk_fmalloc(nrows, "gk_csr_ReorderSymmetric: cwgts");



  /* create perm/iperm if not provided */
  if (perm == NULL) {
    freeperm = 1;
    perm = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: perm"); 
    for (i=0; i<nrows; i++)
      perm[iperm[i]] = i;
  }
  if (iperm == NULL) {
    freeiperm = 1;
    iperm = gk_i32malloc(nrows, "gk_csr_ReorderSymmetric: iperm"); 
    for (i=0; i<nrows; i++)
      iperm[perm[i]] = i;
  }

  /* fill-in the information of the re-ordered matrix */
  nrowptr[0] = jj = 0;
  for (v=0; v<nrows; v++) {
    u = iperm[v];
    for (j=rowptr[u]; j<rowptr[u+1]; j++, jj++) {
      nrowind[jj] = perm[rowind[j]];
      nrowval[jj] = rowval[j];
    }

    if (mat->rlabels)
      nmat->rlabels[v] = mat->rlabels[u];
    if (mat->rmap)
      nmat->rmap[v] = mat->rmap[u];
    if (mat->rnorms)
      nmat->rnorms[v] = mat->rnorms[u];
    if (mat->rsums)
      nmat->rsums[v] = mat->rsums[u];
    if (mat->rsizes)
      nmat->rsizes[v] = mat->rsizes[u];
    if (mat->rvols)
      nmat->rvols[v] = mat->rvols[u];
    if (mat->rwgts)
      nmat->rwgts[v] = mat->rwgts[u];

    if (mat->clabels)
      nmat->clabels[v] = mat->clabels[u];
    if (mat->cmap)
      nmat->cmap[v] = mat->cmap[u];
    if (mat->cnorms)
      nmat->cnorms[v] = mat->cnorms[u];
    if (mat->csums)
      nmat->csums[v] = mat->csums[u];
    if (mat->csizes)
      nmat->csizes[v] = mat->csizes[u];
    if (mat->cvols)
      nmat->cvols[v] = mat->cvols[u];
    if (mat->cwgts)
      nmat->cwgts[v] = mat->cwgts[u];

    nrowptr[v+1] = jj;
  }


  /* free memory */
  if (freeperm)
    gk_free((void **)&perm, LTERM);
  if (freeiperm)
    gk_free((void **)&iperm, LTERM);

  return nmat;
}


/*************************************************************************/
/*! This function computes a permutation of the rows/columns of a symmetric
    matrix based on a breadth-first-traversal. It can be used for re-ordering 
    the matrix to reduce its bandwidth for better cache locality.

    \param[IN]  mat is the matrix whose ordering to be computed.
    \param[IN]  maxdegree is the maximum number of nonzeros of the rows that
                will participate in the BFS ordering. Rows with more nonzeros
                will be put at the front of the ordering in decreasing degree
                order. 
    \param[IN]  v is the starting row of the BFS. A value of -1 indicates that
                a randomly selected row will be used.
    \param[OUT] perm[i] stores the ID of row i in the re-ordered matrix.
    \param[OUT] iperm[i] stores the ID of the row that corresponds to 
                the ith vertex in the re-ordered matrix.

    \note The perm or iperm (but not both) can be NULL, at which point, 
          the corresponding arrays are not returned. Though the program
          works fine when both are NULL, doing that is not smart.
          The returned arrays should be freed with gk_free().
*/
/*************************************************************************/
void gk_csr_ComputeBFSOrderingSymmetric(gk_csr_t *mat, int maxdegree, int v, 
          int32_t **r_perm, int32_t **r_iperm)
{
  int i, k, nrows, first, last;
  ssize_t j, *rowptr;
  int32_t *rowind, *cot, *pos;

  if (mat->nrows != mat->ncols) {
    fprintf(stderr, "gk_csr_ComputeBFSOrderingSymmetric: The matrix needs to be square.\n");
    return;
  }
  if (maxdegree < mat->nrows && v != -1) {
    fprintf(stderr, "gk_csr_ComputeBFSOrderingSymmetric: Since maxdegree node renumbering is requested the starting row should be -1.\n");
    return;
  }
  if (mat->nrows <= 0)
    return;

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;

  /* This array will function like pos + touched of the CC method */
  pos = gk_i32incset(nrows, 0, gk_i32malloc(nrows, "gk_csr_ComputeBFSOrderingSymmetric: pos"));

  /* This array ([C]losed[O]pen[T]odo => cot) serves three purposes. 
     Positions from [0...first) is the current iperm[] vector of the explored rows; 
     Positions from [first...last) is the OPEN list (i.e., visited rows);
     Positions from [last...nrows) is the todo list. */
  cot = gk_i32incset(nrows, 0, gk_i32malloc(nrows, "gk_csr_ComputeBFSOrderingSymmetric: cot"));

  first = last = 0;

  /* deal with maxdegree handling */
  if (maxdegree < nrows) {
    last = nrows;
    for (i=nrows-1; i>=0; i--) {
      if (rowptr[i+1]-rowptr[i] < maxdegree) {
        cot[--last] = i;
        pos[i] = last;
      }
      else {
        cot[first++] = i;
        pos[i] = -1;
      }
    }
    GKASSERT(first == last);

    if (last > 0) { /* reorder them in degree decreasing order */
      gk_ikv_t *cand = gk_ikvmalloc(first, "gk_csr_ComputeBFSOrderingSymmetric: cand");

      for (i=0; i<first; i++) {
        k = cot[i];
        cand[i].key = (int)(rowptr[k+1]-rowptr[k]);
        cand[i].val = k;
      }

      gk_ikvsortd(first, cand);
      for (i=0; i<first; i++) 
        cot[i] = cand[i].val;

      gk_free((void **)&cand, LTERM);
    }

    v = cot[last + RandomInRange(nrows-last)];
  }


  /* swap v with the front of the todo list */
  cot[pos[v]] = cot[last];
  pos[cot[last]] = pos[v];

  cot[last] = v;
  pos[v] = last;


  /* start processing the nodes */
  while (first < nrows) {
    if (first == last) { /* find another starting row */
      k = cot[last];
      GKASSERT(pos[k] != -1);
      pos[k] = -1; /* mark node as being visited */
      last++;
    }

    i = cot[first++];  /* the ++ advances the explored rows */
    for (j=rowptr[i]; j<rowptr[i+1]; j++) {
      k = rowind[j];
      /* if a node has already been visited, its perm[] will be -1 */
      if (pos[k] != -1) {
        /* pos[k] is the location within iperm of where k resides (it is in the 'todo' part); 
           It is placed in that location cot[last] (end of OPEN list) that we 
           are about to overwrite and update pos[cot[last]] to reflect that. */
        cot[pos[k]]    = cot[last]; /* put the head of the todo list to 
                                       where k was in the todo list */
        pos[cot[last]] = pos[k];    /* update perm to reflect the move */

        cot[last++] = k;  /* put node at the end of the OPEN list */
        pos[k]      = -1; /* mark node as being visited */
      }
    }
  }

  /* time to decide what to return */
  if (r_perm != NULL) {
    /* use the 'pos' array to build the perm array */
    for (i=0; i<nrows; i++)
      pos[cot[i]] = i;

    *r_perm = pos;
    pos = NULL;
  }

  if (r_iperm != NULL) {
    *r_iperm = cot;
    cot = NULL;
  }

  /* cleanup memory */
  gk_free((void **)&pos, &cot, LTERM);

}


/*************************************************************************/
/*! This function computes a permutation of the rows of a symmetric matrix
    based on a best-first-traversal. It can be used for re-ordering the matrix
    to reduce its bandwidth for better cache locality.

    \param[IN]  mat is the matrix structure.
    \param[IN]  v is the starting row of the best-first traversal.
    \param[IN]  type indicates the criteria to use to measure the 'bestness'
                of a row.
    \param[OUT] perm[i] stores the ID of row i in the re-ordered matrix.
    \param[OUT] iperm[i] stores the ID of the row that corresponds to 
                the ith row in the re-ordered matrix.

    \note The perm or iperm (but not both) can be NULL, at which point, 
          the corresponding arrays are not returned. Though the program
          works fine when both are NULL, doing that is not smart.
          The returned arrays should be freed with gk_free().
*/
/*************************************************************************/
void gk_csr_ComputeBestFOrderingSymmetric(gk_csr_t *mat, int v, int type, 
          int32_t **r_perm, int32_t **r_iperm)
{
  ssize_t j, jj, *rowptr;
  int i, k, u, nrows, nopen, ntodo;
  int32_t *rowind, *perm, *degrees, *wdegrees, *sod, *level, *ot, *pos;
  gk_i32pq_t *queue;

  if (mat->nrows != mat->ncols) {
    fprintf(stderr, "gk_csr_ComputeBestFOrderingSymmetric: The matrix needs to be square.\n");
    return;
  }
  if (mat->nrows <= 0)
    return;

  nrows  = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;


  /* the degree of the vertices in the closed list */
  degrees = gk_i32smalloc(nrows, 0, "gk_csr_ComputeBestFOrderingSymmetric: degrees");

  /* the weighted degree of the vertices in the closed list for type==3 */
  wdegrees = gk_i32smalloc(nrows, 0, "gk_csr_ComputeBestFOrderingSymmetric: wdegrees");

  /* the sum of differences for type==4 */
  sod = gk_i32smalloc(nrows, 0, "gk_csr_ComputeBestFOrderingSymmetric: sod");

  /* the encountering level of a vertex type==5 */
  level = gk_i32smalloc(nrows, 0, "gk_csr_ComputeBestFOrderingSymmetric: level");

  /* The open+todo list of vertices. 
     The vertices from [0..nopen] are the open vertices.
     The vertices from [nopen..ntodo) are the todo vertices.
     */
  ot = gk_i32incset(nrows, 0, gk_i32malloc(nrows, "gk_csr_ComputeBestFOrderingSymmetric: ot"));

  /* For a vertex that has not been explored, pos[i] is the position in the ot list. */
  pos = gk_i32incset(nrows, 0, gk_i32malloc(nrows, "gk_csr_ComputeBestFOrderingSymmetric: pos"));

  /* if perm[i] >= 0, then perm[i] is the order of vertex i; otherwise perm[i] == -1. */
  perm = gk_i32smalloc(nrows, -1, "gk_csr_ComputeBestFOrderingSymmetric: perm");

  /* create the queue and put the starting vertex in it */
  queue = gk_i32pqCreate(nrows);
  gk_i32pqInsert(queue, v, 1);

  /* put v at the front of the open list */
  pos[0] = ot[0] = v;
  pos[v] = ot[v] = 0;
  nopen = 1;
  ntodo = nrows;

  /* start processing the nodes */
  for (i=0; i<nrows; i++) {
    if (nopen == 0) { /* deal with non-connected graphs */
      gk_i32pqInsert(queue, ot[0], 1);  
      nopen++;
    }

    if ((v = gk_i32pqGetTop(queue)) == -1)
      gk_errexit(SIGERR, "The priority queue got empty ahead of time [i=%d].\n", i);

    if (perm[v] != -1)
      gk_errexit(SIGERR, "The perm[%d] has already been set.\n", v);
    perm[v] = i;

    if (ot[pos[v]] != v)
      gk_errexit(SIGERR, "Something went wrong [ot[pos[%d]]!=%d.\n", v, v);
    if (pos[v] >= nopen)
      gk_errexit(SIGERR, "The position of v is not in open list. pos[%d]=%d is >=%d.\n", v, pos[v], nopen);

    /* remove v from the open list and re-arrange the todo part of the list */
    ot[pos[v]]       = ot[nopen-1];
    pos[ot[nopen-1]] = pos[v];
    if (ntodo > nopen) {
      ot[nopen-1]      = ot[ntodo-1];
      pos[ot[ntodo-1]] = nopen-1;
    }
    nopen--;
    ntodo--;

    for (j=rowptr[v]; j<rowptr[v+1]; j++) {
      u = rowind[j];
      if (perm[u] == -1) {
        /* update ot list, if u is not in the open list by putting it at the end
           of the open list. */
        if (degrees[u] == 0) {
          ot[pos[u]]     = ot[nopen];
          pos[ot[nopen]] = pos[u];
          ot[nopen]      = u;
          pos[u]         = nopen;
          nopen++;

          level[u] = level[v]+1;
          gk_i32pqInsert(queue, u, 0);  
        }


        /* update the in-closed degree */
        degrees[u]++;

        /* update the queues based on the type */
        switch (type) {
          case 1: /* DFS */
            gk_i32pqUpdate(queue, u, 1000*(i+1)+degrees[u]);
            break;

          case 2: /* Max in closed degree */
            gk_i32pqUpdate(queue, u, degrees[u]);
            break;

          case 3: /* Sum of orders in closed list */
            wdegrees[u] += i;
            gk_i32pqUpdate(queue, u, wdegrees[u]);
            break;

          case 4: /* Sum of order-differences */
            /* this is handled at the end of the loop */
            ;
            break;

          case 5: /* BFS with in degree priority */
            gk_i32pqUpdate(queue, u, -(1000*level[u] - degrees[u]));
            break;

          case 6: /* Hybrid of 1+2 */
            gk_i32pqUpdate(queue, u, (i+1)*degrees[u]);
            break;

          default:
            ;
        }
      }
    }

    if (type == 4) { /* update all the vertices in the open list */
      for (j=0; j<nopen; j++) {
        u = ot[j];
        if (perm[u] != -1)
          gk_errexit(SIGERR, "For i=%d, the open list contains a closed row: ot[%zd]=%d, perm[%d]=%d.\n", i, j, u, u, perm[u]);
        sod[u] += degrees[u];
        if (i<1000 || i%25==0)
          gk_i32pqUpdate(queue, u, sod[u]);
      }
    }

    /*
    for (j=0; j<ntodo; j++) {
      if (pos[ot[j]] != j)
        gk_errexit(SIGERR, "pos[ot[%zd]] != %zd.\n", j, j);
    }
    */

  }


  /* time to decide what to return */
  if (r_iperm != NULL) {
    /* use the 'degrees' array to build the iperm array */
    for (i=0; i<nrows; i++)
      degrees[perm[i]] = i;

    *r_iperm = degrees;
    degrees = NULL;
  }

  if (r_perm != NULL) {
    *r_perm = perm;
    perm = NULL;
  }




  /* cleanup memory */
  gk_i32pqDestroy(queue);
  gk_free((void **)&perm, &degrees, &wdegrees, &sod, &ot, &pos, &level, LTERM);

}

