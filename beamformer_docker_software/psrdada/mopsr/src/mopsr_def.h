#ifndef __MOPSR_DEF_H
#define __MOPSR_DEF_H

#include "dada_def.h"

#define MOPSR_DEFAULT_UDPDB_PORT   4001
#define MOPSR_DEFAULT_PWC_PORT     49122
#define MOPSR_DEFAULT_PWC_LOGPORT  49123
#define MOPSR_NUM_UDP_PACKETS      8192

#define MOPSR_NDIM   2

#define MOPSR_MAX_MODULES_PER_PFB  16
#define MOPSR_MAX_PFBS 24
#define MOPSR_MAX_BAYS 88
#define MOPSR_MAX_MODULES 384
#define MOPSR_MAX_CHANS   128

#define UDP_HEADER   16             // size of header/sequence number
//#define UDP_DATA     8192           // obs bytes per packet [missing 8 bytes atm]
//#define UDP_PAYLOAD  8208           // header + datasize
#define UDP_DATA     7680           // obs bytes per packet [missing 8 bytes atm]
#define UDP_PAYLOAD  7696           // header + datasize
#define UDP_NPACKS   65536          // 512 MB worth
#define UDP_IFACE    "192.168.4.14" // default interface

#endif // __MOPSR_DEF_H

