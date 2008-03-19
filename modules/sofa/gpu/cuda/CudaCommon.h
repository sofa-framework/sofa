#ifndef CUDACOMMON_H
#define CUDACOMMON_H

// Default size of thread blocks
// Between 16 and 512
enum { BSIZE=64 };

enum { MBSIZE=96 };
// Max size of thread blocks
enum { MAXTHREADS=512 };

#endif
