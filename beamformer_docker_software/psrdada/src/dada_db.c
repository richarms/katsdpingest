
#include "config.h"
#include "dada_def.h"
#include "ipcbuf.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef HAVE_HWLOC
#include <hwloc.h>
#endif

void usage ()
{
  fprintf (stdout,
          "dada_db - create or destroy the DADA shared memory ring buffer\n"
          "\n"
          "Usage: dada_db [options]\n"
          " -a hdrsz    size of each header buffer (in bytes) [default: %"PRIu64"]\n"
          " -b bufsz    size of each buffer (in bytes) [default: %"PRIu64"]\n"
#ifdef HAVE_HWLOC
          " -c node      assign memory from NUMA node  [default: all nodes]\n"
#endif
          " -d          destroy the shared memory area [default: create]\n"
          " -h          show help\n"
          " -k key      hexadecimal shared memory key  [default: %x]\n"
          " -l          lock the shared memory in RAM\n"
          " -n nbufs    number of buffers in ring      [default: %"PRIu64"]\n"
          " -p          page all blocks into RAM\n"
          " -r nread     number of readers             [default: 1]\n",
          DADA_DEFAULT_HEADER_SIZE,
          DADA_DEFAULT_BLOCK_SIZE,
          DADA_DEFAULT_BLOCK_KEY,
          DADA_DEFAULT_BLOCK_NUM);
}

int main (int argc, char** argv)
{
  uint64_t nbufs = DADA_DEFAULT_BLOCK_NUM;
  uint64_t bufsz = DADA_DEFAULT_BLOCK_SIZE;

  uint64_t nhdrs = IPCBUF_XFERS;
  uint64_t hdrsz = DADA_DEFAULT_HEADER_SIZE;

  key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

  ipcbuf_t data_block = IPCBUF_INIT;
  ipcbuf_t header = IPCBUF_INIT;

  int page = 0;
  int destroy = 0;
  int lock = 0;
  int arg;
  unsigned num_readers = 1;

#ifdef HAVE_HWLOC
  hwloc_topology_t topology;

  // Allocate and initialize topology object.
  hwloc_topology_init(&topology);

  // Perform the topology detection.
  hwloc_topology_load(topology);

  // numa node to bind to
  int numa_node = -1;

  while ((arg = getopt(argc, argv, "a:b:c:dhk:ln:pr:")) != -1) {
#else
  while ((arg = getopt(argc, argv, "a:b:dhk:ln:pr:")) != -1) {
#endif

    switch (arg)  {
    case 'h':
      usage ();
      return 0;

    case 'a':
      if (sscanf (optarg, "%"PRIu64"", &hdrsz) != 1) {
        fprintf (stderr, "dada_db: could not parse hdrsz from %s\n", optarg);
        return -1;
      }
      break;
        
    case 'b':
      if (sscanf (optarg, "%"PRIu64"", &bufsz) != 1) {
        fprintf (stderr, "dada_db: could not parse bufsz from %s\n", optarg);
        return -1;
      }
      break;

#ifdef HAVE_HWLOC
    case 'c':
      if (sscanf (optarg, "%d", &numa_node) != 1)
      {
        fprintf (stderr, "dada_db: could not parse numa_node from %s\n", optarg);
        return -1;
      }

      int numa_nodes = hwloc_get_nbobjs_by_type (topology, HWLOC_OBJ_NODE);
      if ((numa_node < 0) || (numa_node >= numa_nodes))
      {
        fprintf (stderr, "dada_db: only %d NUMA nodes available on this machine\n",
                 numa_nodes);
        return -1;

      }
      break;
#endif

    case 'd':
      destroy = 1;
      break;

    case 'k':
      if (sscanf (optarg, "%x", &dada_key) != 1) 
      {
        fprintf (stderr, "dada_db: could not parse key from %s\n", optarg);
        return -1;
      }
      break;

    case 'n':
      if (sscanf (optarg, "%"PRIu64"", &nbufs) != 1) {
        fprintf (stderr, "dada_db: could not parse nbufs from %s\n", optarg);
        return -1;
      }
      break;

    case 'r':
      if (sscanf (optarg, "%d", &num_readers) != 1) {
        fprintf (stderr, "dada_db: could not parse number of readers from %s\n", optarg);
        return -1;
      }
      break;

    case 'l':
      lock = 1;
      break;

    case 'p':
      page = 1;
      break;
    }
  }

  if (hdrsz < DADA_DEFAULT_HEADER_SIZE)
  {
    fprintf (stderr, "ERROR: header size must be greater than the default header size [%"PRIu64"]\n", DADA_DEFAULT_HEADER_SIZE);
    return -1;
  }

  if ((num_readers < 1) || (num_readers > 5))
  {
    fprintf (stderr, "Number of readers was not sensible: %d\n", num_readers);
    return -1;
  }

  if (destroy) {

    ipcbuf_connect (&data_block, dada_key);
    ipcbuf_destroy (&data_block);

    ipcbuf_connect (&header, dada_key + 1);
    ipcbuf_destroy (&header);

    fprintf (stderr, "Destroyed DADA data and header blocks\n");

    return 0;
  }

#ifdef HAVE_HWLOC

  // fetch the specified core
  hwloc_obj_t obj = hwloc_get_obj_by_type (topology, HWLOC_OBJ_NODE, numa_node);
  if (obj)
  {
    hwloc_membind_policy_t policy = HWLOC_MEMBIND_BIND;
    hwloc_membind_flags_t flags = HWLOC_MEMBIND_STRICT;

    int result = hwloc_set_membind_nodeset (topology, obj->nodeset, policy, flags);
    if (result < 0)
    {
      fprintf (stderr, "dada_db: failed to set memory binding policy: %s\n",
               strerror(errno));
      return -1;
    }
  }

#endif

  if (ipcbuf_create (&data_block, dada_key, nbufs, bufsz, num_readers) < 0) {
    fprintf (stderr, "Could not create DADA data block\n");
    return -1;
  }

  fprintf (stderr, "Created DADA data block with"
          " nbufs=%"PRIu64" bufsz=%"PRIu64" nread=%d\n", nbufs, bufsz, num_readers);

  if (ipcbuf_create (&header, dada_key + 1, nhdrs, hdrsz, num_readers) < 0) {
    fprintf (stderr, "Could not create DADA header block\n");
    return -1;
  }

  fprintf (stderr, "Created DADA header block with nhdrs = %"PRIu64", hdrsz "
                   "= %"PRIu64" bytes, nread=%d\n", nhdrs, hdrsz, num_readers);

  if (lock && ipcbuf_lock (&data_block) < 0) {
    fprintf (stderr, "Could not lock DADA data block into RAM\n");
    return -1;
  }

  if (lock && ipcbuf_lock (&header) < 0) {
    fprintf (stderr, "Could not lock DADA header block into RAM\n");
    return -1;
  }

  if (page && ipcbuf_page (&header) < 0) {
    fprintf (stderr, "Could not page DADA header block into RAM\n");
    return -1;
  }

  if (page && ipcbuf_page ((ipcbuf_t*) &data_block) < 0) {
    fprintf (stderr, "Could not page DADA data block into RAM\n");
    return -1;
  }

#ifdef HAVE_HWLOC
  // Destroy topology object.
  hwloc_topology_destroy(topology);
#endif

  return 0;
}

