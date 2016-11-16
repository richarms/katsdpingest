/*
 * caspsr_udpNib
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>

#ifdef HAVE_PGPLOT
#include <cpgplot.h>
#endif

#include "caspsr_udpNib.h"
#include "dada_generator.h"
#include "dada_affinity.h"
#include "sock.h"

/* disable the sending of data, just catch and forget */
// #define NO_SEND 1

/* debug mode */
#define _DEBUG 1

#define CASPSR_UDPNNIC_PACKS_PER_XFER 1000

#define CASPSR_UDPNIB_MESSAGE_SIZE 65536

#define be64atoh(x)     ((uint64_t)(((x)[0]<<56)|((x)[1]<<48)|((x)[2]<<40)| \
        ((x)[3]<<32)|((x)[4]<<24)|((x)[5]<<16)|((x)[6]<<8)|(x)[7]))
#define htobe64a(a,x)   (a)[0]=(uint8_t)((x)>>56), (a)[1]=(uint8_t)((x)>>48), \
        (a)[2]=(uint8_t)((x)>>40), (a)[3]=(uint8_t)((x)>>32), \
        (a)[4]=(uint8_t)((x)>>24), (a)[5]=(uint8_t)((x)>>16), \
        (a)[6]=(uint8_t)((x)>>8), (a)[7]=(uint8_t)(x)
#define le16atoh(x)     ((uint16_t)(((x)[1]<<8)|(x)[0]))


/* global variables */
int quit_threads = 0;
int quit_plot_thread = 0;
int start_pending = 0;
int stop_pending = 0;
int recording = 0;

const int Device = 0;
const int Inches = 1;
const int Millimetres = 2;
const int Pixels = 3;
const int World = 4;
const int Viewport = 5;

void get_scale (int from, int to, float* width, float* height);
void set_dimensions (unsigned width_pixels, unsigned height_pixels);

void usage()
{
  fprintf (stdout,
	   "caspsr_udpNib [options] i_dist n_dist host1 [hosti]\n"
	   " -i iface       interface for UDP packets [default all interfaces]\n"
	   " -o port        port for 'telnet' control commands\n"
	   " -p port        port for incoming UDP packets [default %d]\n"
	   " -q port        port for outgoing packets [default %d]\n"
     " -t secs        acquire for secs only [default continuous]\n"
     " -n packets     write num packets to each datablock [default %d]\n"
     " -b bytes       number of packets to append to each transfer\n"
     " -c Mibytes     clamp the max sending rate [MiB/s]\n"
     " -d             run as a daemon\n"
     " -r             receive data only, do not attempt to send it\n"
     " -s             show plots of memory buffers\n"
     " -h             print help text\n"
     " -v             verbose messages\n"
     "\n"
     "i_dist          index of distributor\n"
     "n_dist          number of distributors\n"
     "host1 - hosti   destination hosts to write to\n"
     "\n"
     "  The amount of memory used to buffer packets is:\n"
     "    nhost * UDP_DATA * packets_per_xfer\n"
     "  where\n"
     "     nhost is the number of destination hosts\n"
     "     UDP_DATA is 8kB [fixed for CASPSR]\n"
     "     packets_per_xfer is the '-n packets' option\n"
     "\n",
     CASPSR_DEFAULT_UDPNNIC_PORT, CASPSR_DEFAULT_UDPNNIC_PORT, 
     CASPSR_UDPNNIC_PACKS_PER_XFER);
}


/* 
 * Initialise memory and structs for senders
 */
int udpNib_initialize_senders (udpNib_t * ctx) 
{

  unsigned i=0;

  udpNib_sender_t * r = 0;

  /* size of each transfer/buffer */
  uint64_t buffer_size = UDP_PAYLOAD * ctx->packets_per_xfer;

  /* for each sender */
  for (i=0; i<ctx->n_senders; i++)
  {
    r = ctx->senders[i];
    r->log = ctx->log;
    r->size = buffer_size;
    r->w_total = buffer_size;
    r->w_count = 0;
    r->r_count = 0;
    r->fd = 0;

    /* allocate aligned buffer memory */
    if (posix_memalign ( (void **) &(r->buffer), 512, sizeof(char) * buffer_size) != 0) {
      multilog (ctx->log, LOG_ERR, "failed to allocated aligned memory: %s\n", 
                                   strerror(errno));
      return -1;
    }
    if (! r->buffer)
    {
      multilog (ctx->log, LOG_ERR, "failed to allocate %"PRIu64" bytes buffer memory "
                "for sender %d\n", buffer_size, i);
      return -1;
    } 

    if (ctx->verbose)
    {
      double mb_allocated = buffer_size / (1024 * 1024);
      multilog (ctx->log, LOG_INFO, "allocated %4.1f MB for sending thread %d\n", mb_allocated, i);
    }

    if (mlock((void *)  r->buffer, (size_t) buffer_size) < 0) 
      multilog (ctx->log, LOG_WARNING, "failed to lock buffer memory: %s\n", strerror(errno));


  }

  /* allocate required memory strucutres */
  ctx->packets = init_stats_t();
  ctx->bytes   = init_stats_t();

  return 0;

}

void udpNib_reset_senders (udpNib_t * ctx) 
{

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "udpNib_reset_senders()\n");

  unsigned i=0;
  udpNib_sender_t * r = 0;
  char zeroed_char = 'c';
  memset(&zeroed_char, 0, sizeof(zeroed_char));

  /* for each sender */
  for (i=0; i<ctx->n_senders; i++)
  {
    r = ctx->senders[i];
    if (ctx->verbose)
      multilog (ctx->log, LOG_INFO, "udpNib_reset_buffers: resetting memory "
                                    "buffer %d of %d [%"PRIu64" bytes]\n", i, 
                                    ctx->n_senders, r->size);
    r->w_count = 0;
    r->r_count = 0;
    memset(r->buffer, zeroed_char, r->size);
  }

  ctx->packet_reset = 0;

}

/* reset buffer counts and statistics */
int udpNib_reset_buffers(udpNib_t * ctx)
{
  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "udpNib_reset_buffers()\n");

  udpNib_reset_senders(ctx);
  reset_stats_t(ctx->packets);
  reset_stats_t(ctx->bytes);

  return 0;
}


/*
 * Clean up memory allocated to each sender
 */
int udpNib_dealloc_senders(udpNib_t * ctx)
{

  unsigned i = 0;
  for (i=0; i<ctx->n_senders; i++)
  {
    if (munlock((void *) ctx->senders[i]->buffer, (size_t) ctx->senders[i]->size) < 0 )
      multilog(ctx->log, LOG_WARNING, "failed to munlock buffer for sender %d: %s\n", i, strerror(errno));
    free (ctx->senders[i]->buffer);
  }

  return 0;
}        

time_t udpNib_start_function (udpNib_t * ctx)
{

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "udpNib_start_function()\n");

  /* create a zeroed packet for padding the datablocks if we drop packets */
  ctx->zeroed_packet = (char *) malloc(sizeof(char) * UDP_PAYLOAD);
  char zeroed_char = 'c';
  memset(&zeroed_char, 0, sizeof(zeroed_char));
  memset(ctx->zeroed_packet, zeroed_char, UDP_PAYLOAD);

  /* open sockets */
  ctx->fd = dada_udp_sock_in(ctx->log, ctx->interface, ctx->udp_port, ctx->verbose);
  if (ctx->fd < 0) {
    multilog (ctx->log, LOG_ERR, "Error, Failed to create udp socket\n");
    return 0;
  }

  /* set the socket size to */
  int sock_buf_size = 256*1024*1024;
  dada_udp_sock_set_buffer_size (ctx->log, ctx->fd, ctx->verbose, sock_buf_size);

  int on = 1;
  setsockopt(ctx->fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

  /* set the socket to non-blocking */
  sock_nonblock(ctx->fd);

  /* setup the next_seq to the initial value */
  ctx->packets_this_xfer = 0;
  ctx->next_seq = 0;
  ctx->n_sleeps = 0;
  ctx->ooo_packets = 0;
  ctx->ooo_ch_ids = 0;

  return 0;
}


/* 
 *   incremements the sendercounter and initializes the
 *   the memory strcutures
 */
int udpNib_new_sender (udpNib_t * ctx)
{

  if (ctx->verbose >= 2)
    multilog(ctx->log, LOG_INFO, "udpNib_new_sender: curr=%d n=%d\n", ctx->sender_i, ctx->n_senders);

  ctx->sender_i++;
  ctx->sender_i = ctx->sender_i % ctx->n_senders;

  if (ctx->verbose >= 2)
    multilog(ctx->log, LOG_INFO, "udpNib_new_sender: next = %d\n", ctx->sender_i);

  if (ctx->senders[ctx->sender_i]->r_count < ctx->senders[ctx->sender_i]->w_count) 
  {
    multilog (ctx->log, LOG_ERR, "sender %d had not read all the data from the previous xfer [r=%"PRIu64" w=%"PRIu64"]\n", ctx->sender_i, ctx->senders[ctx->sender_i]->r_count, ctx->senders[ctx->sender_i]->w_count);
    return -1;
  }
  
  ctx->senders[ctx->sender_i]->r_count = 0;
  ctx->senders[ctx->sender_i]->w_count = 0;

  return 0;
}


/*
 * Read data from the UDP socket and write to the ring buffers
 */
void * receiving_thread (void * arg)
{

  udpNib_t * ctx = (udpNib_t *) arg;

  /* multilogging facility */
  multilog_t * log = ctx->log;

  /* pointer for header decode function */
  unsigned char * arr;

  /* raw sequence number */
  uint64_t raw_seq_no = 0;

  /* "fixed" raw sequence number */
  uint64_t fixed_raw_seq_no = 0;

  /* offset raw sequence number */
  uint64_t offset_raw_seq_no = 0;

  /* decoded sequence number */
  uint64_t seq_no = 0;

  /* previously received sequence number */
  uint64_t prev_seq_no = 0;

  /* decoded channel id */
  uint64_t ch_id = 0;

  /* all packets seem to have an offset of -3 [sometimes -2] OMG */
  int64_t global_offset = -1;

  /* amount the seq number should increment by */
  int64_t seq_offset = 0;

  /* amount the seq number should increment by */
  uint64_t seq_inc = 1024 * ctx->n_distrib;

  /* used to decode/encode sequence number */
  uint64_t tmp = 0;
  unsigned char ch;

  /* flag for timeout */
  int timeout_ocurred = 0;

  /* flag for ignoring packets */
  unsigned ignore_packet = 0;

  /* small counter for problematic packets */
  int problem_packets = 0;

  /* flag for having a packet */
  unsigned have_packet = 0;

  /* data received from a recv_from call */
  size_t got = 0;

  /* offset for packet loss calculation */
  uint64_t offset = 0;

  /* determine the sequence number boundaries for curr and next buffers */
  int errsv;

  /* flag for start of data */
  unsigned waiting_for_start = 1;

  /* pointer to the current sender */
  udpNib_sender_t * r = 0;

  /* total bytes received + dropped */
  uint64_t bytes_total = 0;

  /* remainder of the fixed raw seq number modulo seq_inc */
  uint64_t remainder = 0;

  unsigned i = 0;
  unsigned j = 0;
  int thread_result = 0;

  struct timeval timeout;

  if (ctx->verbose)
    multilog(log, LOG_INFO, "receiving thread starting\n");

  /* set the CPU that this thread shall run on */
  if (dada_bind_thread_to_core(ctx->recv_core) < 0)
    multilog(ctx->log, LOG_WARNING, "failed to bind receiving_thread to core %d\n", ctx->send_core);

  /* the expected ch_id */
  ctx->ch_id = ctx->i_distrib + 1;

  /* calculate the offset expected in the sequence number */
  seq_offset = 1024 * (ctx->i_distrib);

  if (ctx->verbose)
    multilog (ctx->log, LOG_INFO, "receiving_thread: seq_inc=%"PRIu64", seq_offset=%"PRId64", gobal_offset=%"PRIi64"\n", seq_inc, seq_offset, global_offset);

  /* choose the first datablock and set the ipc ptr */
  ctx->sender_i = 0;
  r = ctx->senders[ctx->sender_i];

  // set recording state once we enter this main loop
  recording = 1;

  /* Continue to receive packets */
  while (!quit_threads && !stop_pending) 
  {

    have_packet = 0; 

    /* incredibly tight loop to try and get a packet */
    while (!have_packet && !quit_threads && !stop_pending)
    {

      /* receive 1 packet into the current senders buffer + offset */
      got = recvfrom (ctx->fd, r->buffer + r->w_count, UDP_PAYLOAD, 0, NULL, NULL);

      /* if we received a packet as expected */
      if (got == UDP_PAYLOAD) {

        have_packet = 1;
        ignore_packet = 0;

      /* a problem ocurred, most likely no packet at the non-blocking socket */
      } else if (got == -1) {

        errsv = errno;

        if (errsv == EAGAIN) {
          ctx->n_sleeps++;
        } else {

          multilog (log, LOG_ERR, "recvfrom failed: %s\n", strerror(errsv));
          thread_result = -1;
          pthread_exit((void *) &thread_result);
        }

      /* we received a packet of the WRONG size, ignore it */
      } else {

        multilog (log, LOG_ERR, "Received %d bytes, expected %d\n",
                                got, UDP_PAYLOAD);
        ignore_packet = 1;

      }
    }

    /* If we did get a packet within the timeout */
    if (have_packet) {

      /* set this pointer to the start of the buffer */
      arr = r->buffer + r->w_count;

      /* Decode the packets apsr specific header */
      raw_seq_no = UINT64_C (0);
      for (i = 0; i < 8; i++ )
      {
        tmp = UINT64_C (0);
        tmp = arr[8 - i - 1];
        raw_seq_no |= (tmp << ((i & 7) << 3));
      }

      /* handle the global offset in sequence numbers */
      fixed_raw_seq_no = raw_seq_no + global_offset;

      /* adjust for the offset of this distributor */
      offset_raw_seq_no = fixed_raw_seq_no - seq_offset;

      /* check the remainder of the sequence number, for errors in global offset */
      remainder = offset_raw_seq_no % seq_inc;

      if (remainder == 0) {
        // do nothing
      } 
      //else if (remainder < 10) 
      else if (remainder < (seq_inc / 2)) 
      {
        multilog(ctx->log, LOG_WARNING, "adjusting global offset from %"PRIi64" to %"PRIi64"\n", global_offset, (global_offset - remainder));
        global_offset -= remainder;
        fixed_raw_seq_no = raw_seq_no + global_offset;
        offset_raw_seq_no = fixed_raw_seq_no - seq_offset;
        remainder = offset_raw_seq_no % seq_inc;
      }
      else if (remainder >= seq_inc / 2)
      {
        multilog(ctx->log, LOG_WARNING, "adjusting global offset from %"PRIi64" to %"PRIi64"\n", global_offset, (global_offset + (seq_inc - remainder)));
        global_offset += (seq_inc - remainder);
        fixed_raw_seq_no = raw_seq_no + global_offset;
        offset_raw_seq_no = fixed_raw_seq_no - seq_offset;
        remainder = offset_raw_seq_no % seq_inc;
      } else {
        // the offset was too great to "fix"
      }

      seq_no = (offset_raw_seq_no) / seq_inc;

      if ((prev_seq_no) && ((seq_no - prev_seq_no) != 1))
      {
        multilog(ctx->log, LOG_INFO, "RESET : raw=%"PRIu64", fixed=%"PRIu64", offset=%"PRIu64", seq_inc=%"PRIu64", remainder=%d, seq=%"PRIu64", prev_seq=%"PRIu64"\n", raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no,  seq_inc, (int) (offset_raw_seq_no % seq_inc), seq_no, prev_seq_no);
      }

      if ((!ignore_packet) && (remainder != 0)) {

        multilog(ctx->log, LOG_INFO, "PROB  : raw=%"PRIu64", fixed=%"PRIu64", offset=%"PRIu64", seq_inc=%"PRIu64", remainder=%d, seq=%"PRIu64", prev_seq=%"PRIu64"\n", raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no, seq_inc, (int) (offset_raw_seq_no % seq_inc), seq_no, prev_seq_no);
        ignore_packet = 1;
        ctx->packet_reset = 1;
        problem_packets++;
      }

      prev_seq_no = seq_no;

      if ((seq_no < 2) && (!ignore_packet)) {

        multilog(ctx->log, LOG_INFO, "START : raw=%"PRIu64", fixed=%"PRIu64", offset=%"PRIu64", seq=%"PRIu64"\n", raw_seq_no, fixed_raw_seq_no, offset_raw_seq_no, seq_no);

      }

      /* If we are waiting for the sequence number to be reset */
      if ((waiting_for_start) && (!ignore_packet)) {

        if (seq_no < 10000) {

          if (ctx->verbose) 
            multilog(ctx->log, LOG_INFO, "ctx->packet_reset = 1\n");
          ctx->packet_reset = 1;
          waiting_for_start = 0;
          ctx->next_seq = 0;

        } else {
          ignore_packet = 1;
        }
      }
    }

    /* If we are still waiting for the start of data */
    if (!ignore_packet) {

      if (seq_no > ctx->next_seq)
        ctx->ooo_packets += seq_no - ctx->next_seq;

      /* we are going to process the packet we have */
      have_packet = 0;

      /* expected packet */
      if (seq_no == ctx->next_seq) {

        /* write the correct packet number to the buffer */
        arr = r->buffer + r->w_count;
        for (j = 0; j < 8; j++ )
        {
          ch = (fixed_raw_seq_no >> ((j & 7) << 3)) & 0xFF;
          arr[8 - j - 1] = ch;
        }

        ctx->packets_this_xfer++;
        ctx->packets->received++;
        ctx->bytes->received += UDP_PAYLOAD;
        ctx->next_seq++;

        bytes_total += UDP_PAYLOAD;

        /* update the senders count */
        r->w_count += UDP_PAYLOAD;

      /* packet is too early in the sequence */
      } else if (seq_no > ctx->next_seq) {

        /* zero the missing ones, and the one we just received as it is out of place */
        offset = (seq_no - ctx->next_seq) + 1;
        //multilog (log, LOG_WARNING, "Zeroing %"PRIu64" packets into datablock\n", offset);
        for (i=0; i<offset; i++) 
        {

          /* zero the data in the packet */
          // TODO should reinstate this
          //memcpy (r->buffer + r->w_count, ctx->zeroed_packet, UDP_PAYLOAD);

          // encode the sequence number into the memory buffer
          arr = r->buffer + r->w_count;
          fixed_raw_seq_no = ((ctx->next_seq + i) * seq_inc) + seq_offset;

          for (j = 0; j < 8; j++ )
          {
            ch = (fixed_raw_seq_no >> ((j & 7) << 3)) & 0xFF;
            arr[8 - j - 1] = ch;
            ch = (ctx->ch_id >> ((j & 7) << 3)) & 0xFF;
            arr[16 - j - 1] = ch;
          }
          //multilog (log, LOG_WARNING, "sending fake %"PRIu64", %"PRIu64"\n", fixed_raw_seq_no, ctx->ch_id);

          ctx->packets_this_xfer++;
          r->w_count += UDP_PAYLOAD;

          //multilog (log, LOG_WARNING, "[%d] ppxfer=%"PRIu64" w_count=%"PRIu64"\n", ctx->sender_i, ctx->packets_this_xfer, r->w_count);

          /* check for a change of data block */
          if (ctx->packets_this_xfer >= ctx->packets_per_xfer)  
          {
            //fprintf(stderr, "1: ctx->packets_this_xfer[%"PRIu64"] >= ctx->packets_per_xfer"
            //                "[%"PRIu64"]\n", ctx->packets_this_xfer, ctx->packets_per_xfer);

            if ( udpNib_new_sender (ctx) < 0 )
            {
              multilog (ctx->log, LOG_ERR, "Failed to move to a new sender\n");
              thread_result = -1;
              pthread_exit((void *) &thread_result);
            }
            r = ctx->senders[ctx->sender_i];
            ctx->packets_this_xfer = 0;
          }
        }

        ctx->packets->dropped += offset;
        ctx->bytes->dropped += offset * UDP_PAYLOAD;
        ctx->next_seq += offset;

        bytes_total += offset * UDP_PAYLOAD;

      }
      else 
      {
        /* packet is to late, already been zerod */
        if (stop_pending) 
          multilog (ctx->log, LOG_INFO, "seq_no [%"PRIu64"] < ctx->next_seq [%"PRIu64"]\n", seq_no, ctx->next_seq);
        else
          multilog (ctx->log, LOG_WARNING, "seq_no [%"PRIu64"] < ctx->next_seq [%"PRIu64"]\n", seq_no, ctx->next_seq);
         

      }

      /* check for a change of data block */
      if (ctx->packets_this_xfer >= ctx->packets_per_xfer) 
      {
        //fprintf(stderr, "XFER COMPLETE sender[%d] r_count=%"PRIu64", w_count=%"PRIu64", w_total=%"PRIu64", last_seq=%"PRIu64"\n", ctx->sender_i, r->r_count, r->w_count, r->w_total, seq_no);

        if ( udpNib_new_sender (ctx) < 0 )
        {
          multilog (ctx->log, LOG_ERR, "Failed to move to a new sender\n");
          thread_result = -1;
          pthread_exit((void *) &thread_result);
        }
        r = ctx->senders[ctx->sender_i];
        ctx->packets_this_xfer = 0;
      }
    } 

    // we have ignored more than 10 packets
    if (problem_packets > 10) {
      multilog (ctx->log, LOG_WARNING, "Ignored more than 10 packets, exiting\n");
      quit_threads = 1;  
    }

    if (ctx->bytes_to_acquire && (ctx->bytes_to_acquire > bytes_total))
      quit_threads = 1;

  }

  if (quit_threads && ctx->verbose) 
    multilog (ctx->log, LOG_INFO, "main_function: quit_threads detected\n");
 
  if (ctx->verbose) 
    multilog(log, LOG_INFO, "receiving thread exiting\n");

  /* return 0 */
  pthread_exit((void *) &thread_result);
}

/*
 * Close the udp socket and file
 */

int udpNib_stop_function (udpNib_t* ctx)
{

  multilog_t *log = ctx->log;

  /* get our context, contains all required params */
  if (ctx->packets->dropped && ctx->next_seq>0) {
    double percent = (double) ctx->packets->dropped / (double) ctx->next_seq;
    percent *= 100;

    multilog(log, LOG_INFO, "packets dropped %"PRIu64" / %"PRIu64 " = %8.6f %\n"
             ,ctx->packets->dropped, ctx->next_seq, percent);

  }

  close(ctx->fd);
  recording = 0;
  return 0;

}


int main (int argc, char **argv)
{

  /* DADA Logger */ 
  multilog_t* log = 0;

  /* number of this distributor */
  unsigned i_distrib = 0;

  /* total number of distributors being used */
  unsigned n_distrib = 0;

  /* Interface on which to listen for udp packets */
  char * interface = "any";

  /* port for control commands */
  int control_port = 0;

  /* port for incoming UDP packets */
  int inc_port = CASPSR_DEFAULT_UDPNNIC_PORT;

  /* port for outgoing UDP packets */
  int out_port = CASPSR_DEFAULT_UDPNNIC_PORT;

  /* multilog output port */
  int l_port = CASPSR_DEFAULT_PWC_LOGPORT;

  /* Flag set in daemon mode */
  char daemon = 0;

  /* Flag set in verbose mode */
  char verbose = 0;

  /* number of seconds/bytes to acquire for */
  unsigned nsecs = 0;

  /* actual struct with info */
  udpNib_t udpNnic;

  /* custom header from a file, implies no controlling pwcc */
  char * header_file = NULL;

  /* Pointer to array of "read" data */
  char *src;

  /* Ignore dropped packets */
  unsigned ignore_dropped = 0;

  /* number of packets for each tranfer */
  unsigned packets_per_xfer = CASPSR_UDPNNIC_PACKS_PER_XFER;

  /* number of packets to append to each transfer */ 
  int packets_to_append = 0;

  /* show plots of the memory buffers */
  int show_plots = 0;

  /* receive only flag */
  unsigned receive_only = 0;

  /* clamped output rate (for sending threads) [Million Bytes/second] */
  int clamped_output_rate = 0;

  int arg = 0;

  /* statistics thread */
  pthread_t stats_thread_id;

  /* plotting thread */
  pthread_t plotting_thread_id;

  /* control thread */
  pthread_t control_thread_id;

  /* sending threads */
  pthread_t * sending_thread_ids;

  /* receiving thread */
  pthread_t receiving_thread_id;

  while ((arg=getopt(argc,argv,"b:c:di:l:n:o:p:q:rst:vh")) != -1) {
    switch (arg) {

    case 'b':
      packets_to_append = atoi(optarg);
      break;

    case 'c':
      clamped_output_rate = atoi(optarg);
      break; 

    case 'd':
      daemon = 1;
      break; 

    case 'i':
      if (optarg)
        interface = optarg;
      break;
    
    case 'l':
      if (optarg) {
        l_port = atoi(optarg);
        break;
      } else {
        usage();
        return EXIT_FAILURE;
      }

    case 'n':
      packets_per_xfer = atoi(optarg);
      break;

    case 'o':
      control_port = atoi(optarg);
      break;

    case 'p':
      inc_port = atoi (optarg);
      break;

    case 'q':
      out_port = atoi (optarg);
      break;

    case 'r':
      receive_only = 1;
      break;

    case 's':
      show_plots = 1;
      break;

    case 't':
      nsecs = atoi (optarg);
      break;

    case 'v':
      verbose = 1;
      break;

    case 'h':
      usage();
      return 0;
      
    default:
      usage ();
      return 0;
      
    }
  }
  
  if (packets_to_append) {
    fprintf(stderr, "packet appending not implemented yet!\n");
    exit(EXIT_FAILURE);
  }

  /* check the command line arguments */
  int num_dests = (argc-optind) - 2;
  int i = 0;
  int rval = 0;
  void* result = 0;

  if (num_dests < 1 || num_dests > 16) {
    fprintf(stderr, "ERROR: must have at between 1 and 16 dest hosts\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  /* parse the number this distributor */
  if (sscanf(argv[optind], "%d", &i_distrib) != 1) {
    fprintf(stderr, "ERROR: failed to parse the number of this distributor\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  /* parse the number of distributors */
  if (sscanf(argv[optind+1], "%d", &n_distrib) != 1) {
    fprintf(stderr, "ERROR: failed to parse the number of distributors\n\n");
    usage();
    exit(EXIT_FAILURE);
  }

  if ((n_distrib < 2) || (n_distrib > 4)) {
    fprintf(stderr, "ERROR: number of distributors must be between 2 and 4 [parsed %d]\n", n_distrib);
    usage();
    exit(EXIT_FAILURE);
  }

  if ((i_distrib < 0) || (i_distrib >= n_distrib)) {
    fprintf(stderr, "ERROR: this distributor's number must be between 0 and %d [parsed %d]\n", (n_distrib-1), i_distrib);
    usage();
    exit(EXIT_FAILURE);
  }

  log = multilog_open ("caspsr_udpNib", 0);

  if (daemon)
    be_a_daemon ();
  else
    multilog_add (log, stderr);

  /* check we will have enough memory */
  struct sysinfo myinfo;
  sysinfo(&myinfo); 

  uint64_t required_memory = 0;
  uint64_t total_memory = 0;
  total_memory = myinfo.mem_unit * myinfo.totalram; 
  required_memory = num_dests;
  required_memory *= UDP_PAYLOAD;
  required_memory *= (packets_per_xfer + packets_to_append);
  if (required_memory > 0.75 * total_memory) {
    double percent = ((double) required_memory) / ((double) total_memory);
    fprintf(stderr, "ERROR: parameters would try to use %3.0f%% of RAM [max 75%%]\n", percent*100);
    exit(EXIT_FAILURE);
  }

  /* allocate sender data strutures */
  udpNnic.senders = (udpNib_sender_t **) malloc(sizeof(udpNib_sender_t*) * num_dests);
  assert(udpNnic.senders != NULL);

  sending_thread_ids = (pthread_t *) malloc(sizeof(pthread_t) * num_dests);
  assert(sending_thread_ids != NULL);

  /* parse destination hostnames from the command line */
  unsigned j = 0;
  for (i=optind+2; i<optind+2+num_dests; i++) {
    udpNnic.senders[j] = (udpNib_sender_t *) malloc(sizeof(udpNib_sender_t));
    udpNnic.senders[j]->host = strdup(argv[i]);
    udpNnic.senders[j]->port = out_port;
    udpNnic.senders[j]->ib_message_size = CASPSR_IB_MESSAGE_SIZE;
    udpNnic.senders[j]->ib_rx_depth = CASPSR_IB_RX_DEPTH;
    udpNnic.senders[j]->ib_use_cq = CASPSR_IB_USE_CQ;
    udpNnic.senders[j]->ib_mtu = caspsr_ib_mtu_to_enum(2048);
    udpNnic.senders[j]->ib_sl = 0;
    udpNnic.senders[j]->verbose = verbose;
    j++;
  }

  /* Setup context information */
  udpNnic.log = log;
  udpNnic.verbose = verbose;
  udpNnic.udp_port = inc_port;
  udpNnic.control_port = control_port;
  udpNnic.interface = strdup(interface);
  udpNnic.packets_per_xfer = packets_per_xfer;
  udpNnic.packets_to_append = packets_to_append;
  udpNnic.packets_this_xfer = 0;
  udpNnic.n_senders = num_dests;
  udpNnic.n_distrib = n_distrib;
  udpNnic.i_distrib = i_distrib;
  udpNnic.bytes_to_acquire = -1;
  udpNnic.ooo_ch_ids = 0;
  udpNnic.ooo_packets = 0;
  udpNnic.send_core = 0;
  udpNnic.recv_core = 0;
  udpNnic.send_sleeps = 0;
  udpNnic.n_sleeps = 0;
  udpNnic.receive_only = receive_only;
  udpNnic.clamped_output_rate = clamped_output_rate;
  udpNnic.mb_rcv_ps = 0;
  udpNnic.mb_drp_ps = 0;
  udpNnic.mb_snd_ps = 0;

  signal(SIGINT, signal_handler);

  /* Initialise senders */
  if (udpNib_initialize_senders(&udpNnic) < 0) {
    fprintf(stderr, "Failed to initialize senders\n");
    return -1;
  }

  if (control_port) 
  {
    if (verbose)
      multilog(log, LOG_INFO, "starting control_thread()\n");
    rval = pthread_create (&control_thread_id, 0, (void *) control_thread, (void *) &udpNnic);
    if (rval != 0) {
      multilog(log, LOG_INFO, "Error creating control_thread: %s\n", strerror(rval));
      return -1;
    }
  }

  if (verbose)
    multilog(log, LOG_INFO, "starting stats_thread()\n");
  rval = pthread_create (&stats_thread_id, 0, (void *) stats_thread, (void *) &udpNnic);
  if (rval != 0) {
    multilog(log, LOG_INFO, "Error creating stats_thread: %s\n", strerror(rval));
    return -1;
  }

  // Main control loop
  while (!quit_threads) 
  {

    if (udpNib_reset_buffers(&udpNnic) < 0) {
      fprintf(stderr, "Failed to reset sender buffers\n");
      return -1;
    }

    // if we have a control port/thread, wait for a START command
    while (control_port && !start_pending && !quit_threads) 
      sleep(1);

    if (quit_threads)
      break;

    if (verbose) multilog(log, LOG_INFO, "udpNib_start_function()\n");
    time_t utc = udpNib_start_function(&udpNnic);
    if (utc == -1 ) {
      multilog(log, LOG_ERR, "Could not run start function\n");
      return EXIT_FAILURE;
    }

    /* set the CPU/CORE for the threads */
    /* process on last 4 cores */
    udpNnic.recv_core = (i_distrib - (i_distrib % 2)) + 4; 
    udpNnic.send_core = (i_distrib - (i_distrib % 2)) + 5;

    /* make the receiving thread high priority */
    int min_pri = sched_get_priority_min(SCHED_RR);
    int max_pri = sched_get_priority_max(SCHED_RR);
    if (verbose)
      multilog(log, LOG_INFO, "sched_priority: min=%d, max=%d\n", min_pri, max_pri);

    struct sched_param send_parm;
    struct sched_param recv_parm;
    send_parm.sched_priority=(min_pri+max_pri)/2;
    recv_parm.sched_priority=max_pri;

    pthread_attr_t send_attr;
    pthread_attr_t recv_attr;

    if (pthread_attr_init(&send_attr) != 0) {
      fprintf(stderr, "pthread_attr_init failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_init(&recv_attr) != 0) {
      fprintf(stderr, "pthread_attr_init failed: %s\n", strerror(errno));
      return 1;
    }
  
    if (pthread_attr_setinheritsched(&recv_attr, PTHREAD_EXPLICIT_SCHED) != 0) {
      fprintf(stderr, "pthread_attr_setinheritsched failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setinheritsched(&send_attr, PTHREAD_EXPLICIT_SCHED) != 0) {
      fprintf(stderr, "pthread_attr_setinheritsched failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedpolicy(&send_attr, SCHED_RR) != 0) {
      fprintf(stderr, "pthread_attr_setschedpolicy failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedpolicy(&recv_attr, SCHED_RR) != 0) {
      fprintf(stderr, "pthread_attr_setschedpolicy failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedparam(&send_attr,&send_parm) != 0) {
      fprintf(stderr, "pthread_attr_setschedparam failed: %s\n", strerror(errno));
      return 1;
    }

    if (pthread_attr_setschedparam(&recv_attr,&recv_parm) != 0) {
      fprintf(stderr, "pthread_attr_setschedparam failed: %s\n", strerror(errno));
      return 1;
    }

    if (verbose)
      multilog(log, LOG_INFO, "starting sending threads\n");

    for (i = 0; i < num_dests; i++) 
    {
      rval = pthread_create (&(sending_thread_ids[i]), 0, (void *) sending_thread, (void *) &(udpNnic.senders[i]));
      if (rval != 0) {
        multilog(log, LOG_INFO, "Error creating sending_thread[%d]: %s\n", i, strerror(rval));
        return -1;
      }
    }

#ifdef HAVE_PGPLOT
    if (show_plots) {
      multilog(log, LOG_INFO, "starting plotting_thread()\n");
      rval = pthread_create (&plotting_thread_id, 0, (void *) plotting_thread, (void *) &udpNnic);
      if (rval != 0) {
        multilog(log, LOG_INFO, "Error creating plotting_thread: %s\n", strerror(rval));
        return -1;
      }
    }
#endif

    /* set the total number of bytes to acquire */
    udpNnic.bytes_to_acquire = 1600 * 1000 * 1000 * (int64_t) nsecs;
    udpNnic.bytes_to_acquire /= udpNnic.n_distrib;

    if (verbose)
    { 
      if (udpNnic.bytes_to_acquire) 
        multilog(log, LOG_INFO, "bytes_to_acquire = %"PRIu64" Million Bytes, nsecs=%d\n", udpNnic.bytes_to_acquire/1000000, nsecs);
      else
        multilog(log, LOG_INFO, "Acquiring data indefinitely\n");
    }

    if (verbose)
      multilog(log, LOG_INFO, "starting receiving_thread()\n");
    rval = pthread_create (&receiving_thread_id, &recv_attr, (void *) receiving_thread, (void *) &udpNnic);
    if (rval != 0) {
      multilog(log, LOG_INFO, "Error creating receiving_thread: %s\n", strerror(rval));
      return -1;
    }


    if (verbose) 
      multilog(log, LOG_INFO, "joining receiving_thread\n");
    pthread_join (receiving_thread_id, &result);

    //quit_threads = 1;

    if (verbose) 
      multilog(log, LOG_INFO, "joining sending threads\n");
    for (i=0; i<num_dests; i++) {
      pthread_join (sending_thread_ids[i], &result);
    }

#ifdef HAVE_PGPLOT
    if (show_plots) {
      quit_plot_thread = 1;
      multilog(log, LOG_INFO, "joining plotting_thread\n");
      pthread_join (plotting_thread_id, &result);
    }
#endif

    if (verbose) 
      multilog(log, LOG_INFO, "udpNib_stop_function\n");
    if ( udpNib_stop_function(&udpNnic) != 0)
      fprintf(stderr, "Error stopping acquisition");

    if (!control_port)
      quit_threads = 1;

  }

  if (control_port)
  {
    if (verbose)
      multilog(log, LOG_INFO, "joining control_thread\n");
    pthread_join (control_thread_id, &result);
  }

  if (verbose)
    multilog(log, LOG_INFO, "joining stats_thread\n");
  pthread_join (stats_thread_id, &result);

  /* clean up memory */
  if ( udpNib_dealloc_senders(&udpNnic) < 0) 
    fprintf(stderr, "failed to clean up senders\n");

  return EXIT_SUCCESS;

}


/* 
 *   Sending thread
 */
void * sending_thread(void * arg) 
{

  udpNib_sender_t * r = (udpNib_sender_t *) arg;

  // IB device list
  struct ibv_device ** dev_list;

  // IB device to use
  struct ibv_device * ib_dev;

  // caspsr ib struct
  caspsr_ib_t * ib; 

  // TODO check number of outstanding read requests ?
  int routs = 0;

  // remote ib dest
  ib_dest * remote_dest;

  // local ib dest
  ib_dest local_dest;

  int ib_port = 1;

  /* counters */
  unsigned i = 0;
  unsigned j = 0;

  //ctx->r_bytes = 0;
  //ctx->r_packs = 0;

  if (r->verbose)
    multilog (r->log, LOG_INFO, "sending_thread: opening sockets\n");

  // get the list of IB devices
  dev_list = ibv_get_device_list(NULL);
  if (!dev_list) {
    multilog (r->log, LOG_ERR, "Failed to get IB devices list\n");
    return;
  }

  // use the first infiniband device
  ib_dev = *dev_list;
  if (!ib_dev) {
    multilog (r->log, LOG_ERR, "No IB devices found\n");
    return;
  }

   
  /* initialize caspsr_ib struct */
  ib = caspsr_ib_init(ib_dev, r->ib_message_size, 1, r->ib_rx_depth, r->port, r->ib_use_cq);

  routs = caspsr_ib_post_recv(ib, 0, ib->rx_depth, CASPSR_IB_RECV_WRID);
  if (routs < ib->rx_depth) {
    multilog (r->log, LOG_ERR, "Couldn't post receive (%d)\n", routs);
    return;
  }

  /* if we are using a completion queue */
  if (r->ib_use_cq) {
    /* requests completion notification on the completion queue */
    if (ibv_req_notify_cq(ib->cq, 0)) {
      multilog (r->log, LOG_ERR, "Couldn't request CQ notification\n");
      return;
    }
  }

  // get the local lid
  local_dest.lid = caspsr_ib_get_lid(ib->context, ib_port);
  local_dest.qpn = ib->qp->qp_num;
  local_dest.psn = lrand48() & 0xffffff;
  if (!local_dest.lid) {
    multilog (r->log, LOG_ERR, "Couldn't get local LID\n");
    return ;
  }

  if (r->verbose) {
    printf("  local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
         local_dest.lid, local_dest.qpn, local_dest.psn);
  }

  remote_dest = caspsr_ib_server_exch_dest(ib, ib_port, r->ib_mtu, r->port, r->ib_sl, &local_dest);
  if (!remote_dest) {
    multilog (r->log, LOG_ERR, "Could not get remote IB device info\n");
    return;
  }

  printf("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x\n",
         remote_dest->lid, remote_dest->qpn, remote_dest->psn);

  ib->pending = CASPSR_IB_RECV_WRID;

  // main sending loop 
  size_t wrote = 0;
  unsigned keep_sending = 1;

  int reported = 0;

  // send data in 640,000 byte messages. for a full xfer, this should
  // take 5000 iterations, we will do it in 0.25 second bursts

  // 80 MB = 0.20 s with 4 distribs
  uint64_t transmission_chunk = 80 * 1000 * 1000; 
  uint64_t message_size = 640 * 1000;

  assert(transmission_chunk % message_size == 0);
  uint64_t iters = transmission_chunk / message_size;
  uint64_t scnt = 0;

  int num_cq_events = 0;

  /* just keep sending until asked to stop */
  while (keep_sending) 
  {

    // if we have 0.25 seconds of data to send
    if ( r->r_count + transmission_chunk <= r->w_count )
    {
      // send the data
      
      scnt = 0;
      while (scnt < iters) {
        if (r->ib_use_cq) {
          struct ibv_cq *ev_cq;
          void          *ev_ctx;

          if (ibv_get_cq_event(ib->channel, &ev_cq, &ev_ctx)) {
            multilog (r->log, LOG_ERR, "Failed to get cq_event\n");
            return;
          }

          ++num_cq_events;

          if (ev_cq != ib->cq) {
            multilog (r->log, LOG_ERR, "CQ event for unknown CQ %p\n", ev_cq);
            return;
          }

          if (ibv_req_notify_cq(ib->cq, 0)) {
            multilog (r->log, LOG_ERR, "Couldn't request CQ notification\n");
            return;
          }
        }

        {
          struct ibv_wc wc[2];
          int ne, i;
          uint64_t pollcount = 0;

          do { 
            ne = ibv_poll_cq(ib->cq, 2, wc);
            if (ne < 0) {
              multilog (r->log, LOG_ERR, "poll CQ failed %d\n", ne);
              return;
            }
          } while (!r->ib_use_cq && ne < 1);

          for (i = 0; i < ne; ++i) {
            if (wc[i].status != IBV_WC_SUCCESS) {
              multilog (r->log, LOG_ERR, "Failed status %s (%d) for wr_id %d\n",
                ibv_wc_status_str(wc[i].status),
                wc[i].status, (int) wc[i].wr_id);
              return;
            }

            switch ((int) wc[i].wr_id) {

              case CASPSR_IB_SEND_WRID:
                ++scnt;
                break;

              default:
                multilog (r->log, LOG_ERR, "Completion for unknown wr_id %d\n",
                  (int) wc[i].wr_id);
                return;
            }

            if (scnt < iters) {
              if (caspsr_ib_post_send(ib, 0)) {
                multilog (r->log, LOG_ERR, "Couldn't post send\n");
                return;
              }
              ib->pending = CASPSR_IB_SEND_WRID;
            }
          }
        }
      }

      r->r_count += transmission_chunk;

      // AJ new
      if ((r->r_count == r->w_count) && (r->r_count == r->w_total)) {
        r->w_count = 0;
        r->r_count = 0;
      }
    }

    /* check that everything that has been written has been sent */
    if (quit_threads || stop_pending) 
    {

      keep_sending = 0;

      // stop, but flush all data in buffers
      if (stop_pending == 1) {

        if (r->r_count != r->w_count)
          keep_sending = 1;      
       
        if (keep_sending && !reported) {
          multilog (r->log, LOG_INFO, "sending_thread: still sending\n");
          reported = 1;
        }

      // stop immediately
      } else {
        multilog (r->log, LOG_INFO, "sending_thread: stopping immediately\n");
      }
    }
  }

  if (r->verbose)
    multilog (r->log, LOG_INFO, "sending_thread: closing sockets\n");

  ibv_ack_cq_events(ib->cq, num_cq_events);

  if (caspsr_ib_close_ctx(ib))
    return;

  ibv_free_device_list(dev_list);
  free(remote_dest);

  if (r->verbose)
    multilog (r->log, LOG_INFO, "sending_thread: exiting\n");

  int thread_result = 0;
  pthread_exit((void *) &thread_result);

}

/*
 *  Thread to control the acquisition of data, allows only 1 connection at a time
 */
void control_thread(void * arg) {

  udpNib_t * ctx = (udpNib_t *) arg;

  multilog(ctx->log, LOG_INFO, "control_thread: starting\n");

  // port on which to listen for control commands
  int port = ctx->control_port;

  // buffer for incoming command strings
  int bufsize = 1024;
  char* buffer = (char *) malloc (sizeof(char) * bufsize);
  assert (buffer != 0);

  FILE *sockin = 0;
  FILE *sockout = 0;
  int listen_fd = 0;
  int fd = 0;
  char *rgot = 0;
  int readsocks = 0;
  fd_set socks;
  struct timeval timeout;

  // create a socket on which to listen
  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: creating socket on port %d\n", port);

  listen_fd = sock_create (&port);
  if (listen_fd < 0)  {
    multilog(ctx->log, LOG_ERR, "Failed to create socket for control commands: %s\n", strerror(errno));
    return;
  }

  while (!quit_threads) {

    // reset the FD set for selecting  
    FD_ZERO(&socks);
    FD_SET(listen_fd, &socks);
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    readsocks = select(listen_fd+1, &socks, (fd_set *) 0, (fd_set *) 0, &timeout);

    // error on select
    if (readsocks < 0) 
    {
      perror("select");
      exit(EXIT_FAILURE);
    }

    // no connections, just ignore
    else if (readsocks == 0) 
    {
    } 

    // accept the connection  
    else 
    {
   
      if (ctx->verbose) 
        multilog(ctx->log, LOG_INFO, "control_thread: accepting conection\n");

      fd =  sock_accept (listen_fd);
      if (fd < 0)  {
        multilog(ctx->log, LOG_WARNING, "control_thread: Error accepting "
                                        "connection %s\n", strerror(errno));
        break;
      }

      sockin = fdopen(fd,"r");
      if (!sockin)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating input "
                                        "stream %s\n", strerror(errno));


      sockout = fdopen(fd,"w");
      if (!sockout)
        multilog(ctx->log, LOG_WARNING, "control_thread: error creating output "
                                        "stream %s\n", strerror(errno));

      setbuf (sockin, 0);
      setbuf (sockout, 0);

      rgot = fgets (buffer, bufsize, sockin);

      if (rgot && !feof(sockin)) {

        buffer[strlen(buffer)-1] = '\0';

        if (ctx->verbose)
          multilog(ctx->log, LOG_INFO, "control_thread: received %s\n", buffer);

        // REQUEST STATISTICS
        if (strstr(buffer, "STATS") != NULL) 
        {
          fprintf (sockout, "mb_rcv_ps=%4.1f,mb_drp_ps=%4.1f,mb_snd_ps=%4.1f,"
                            "ooo_pkts=%"PRIu64",ooo_chids=%"PRIu64",mb_free="
                            "%4.1f,mb_total=%4.1f\r\n", 
                             ctx->mb_rcv_ps, ctx->mb_drp_ps, ctx->mb_snd_ps, 
                             ctx->ooo_packets, ctx->ooo_ch_ids, ctx->mb_free, ctx->mb_total);
          fprintf (sockout, "ok\r\n");
        }

        // START COMMAND
        else if (strstr(buffer,"START") != NULL) {

          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: START command received\n");

          start_pending = 1;
          while (recording != 1) 
          {
            sleep(1);
          }
          start_pending = 0;
          fprintf(sockout, "ok\r\n");

          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording started\n");
        }

        // FLUSH COMMAND - stop acquisition of data, but flush all packets 
        // already received
        else if (strstr(buffer,"FLUSH") != NULL)
        {
          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: FLUSH command received, stopping recording\n");

          stop_pending = 1;
          while (recording != 0)
          {
            sleep(1);
          }
          stop_pending = 0;
          fprintf(sockout, "ok\r\n");

          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording stopped\n");
        }

        else if (strstr(buffer,"STOP") != NULL)
        {
          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: STOP command received, stopping immediately\n");

          stop_pending = 2;
          while (recording != 0)
          {
            sleep(1);
          }
          stop_pending = 0;
          fprintf(sockout, "ok\r\n");

          //if (ctx->verbose)
            multilog(ctx->log, LOG_INFO, "control_thread: recording stopped\n");
        }


        // QUIT COMMAND, immediately exit 
        else if (strstr(buffer,"QUIT") != NULL) 
        {
          multilog(ctx->log, LOG_INFO, "control_thread: QUIT command received, exiting\n");
          quit_threads = 1;
          fprintf(sockout, "ok\r\n");
        }

        // UNRECOGNISED COMMAND
        else 
        {
          multilog(ctx->log, LOG_WARNING, "control_thread: unrecognised command: %s\n", buffer);
          fprintf(sockout, "fail\r\n");
        }
      }
    }

    close(fd);
  }
  close(listen_fd);

  if (ctx->verbose)
    multilog(ctx->log, LOG_INFO, "control_thread: exiting\n");

}

/* 
 *  Thread to print simple capture statistics
 */
void stats_thread(void * arg) {

  udpNib_t * ctx = (udpNib_t *) arg;

  uint64_t b_rcv_total = 0;
  uint64_t b_rcv_1sec = 0;
  uint64_t b_rcv_curr = 0;

  uint64_t b_drp_total = 0;
  uint64_t b_drp_1sec = 0;
  uint64_t b_drp_curr = 0;

  uint64_t b_snd_total = 0;
  uint64_t b_snd_1sec = 0;
  uint64_t b_snd_curr = 0;

  uint64_t s_snd_total = 0;
  uint64_t s_snd_1sec = 0;
  uint64_t s_snd_curr = 0;

  uint64_t s_rcv_total = 0;
  uint64_t s_rcv_1sec = 0;
  uint64_t s_rcv_curr = 0;

  uint64_t ooo_pkts = 0;
  uint64_t ooo_chids = 0;

  unsigned i=0;
  unsigned next_sender = 0;
  uint64_t bytes_buffered = 0;
  double mbytes_buffered = 0;
  uint64_t bytes_free = 0;

  udpNib_sender_t * r;

  while (!quit_threads)
  {
            
    /* get a snapshot of the data as quickly as possible */
    b_rcv_curr = ctx->bytes->received;
    b_drp_curr = ctx->bytes->dropped;
    b_snd_curr = ctx->r_bytes;
    s_snd_curr = ctx->send_sleeps;
    s_rcv_curr = ctx->n_sleeps;
    ooo_pkts   = ctx->ooo_packets;
    ooo_chids  = ctx->ooo_ch_ids;

    /* calc the values for the last second */
    b_rcv_1sec = b_rcv_curr - b_rcv_total;
    b_snd_1sec = b_snd_curr - b_snd_total;
    b_drp_1sec = b_drp_curr - b_drp_total;
    s_rcv_1sec = s_rcv_curr - s_rcv_total;
    s_snd_1sec = s_snd_curr - s_snd_total;

    /* update the totals */
    b_rcv_total = b_rcv_curr;
    b_snd_total = b_snd_curr;
    b_drp_total = b_drp_curr;
    s_rcv_total = s_rcv_curr;
    s_snd_total = s_snd_curr;

    ctx->mb_rcv_ps = (double) b_rcv_1sec / 1000000;
    ctx->mb_drp_ps = (double) b_drp_1sec / 1000000;
    ctx->mb_snd_ps = (double) b_snd_1sec / 1000000;

    /* determine how much memory is free in the senders */
    bytes_buffered = 0;
    bytes_free = 0;
    for (i=0; i<ctx->n_senders; i++) 
    {
      r = ctx->senders[i]; 
      bytes_buffered += (r->w_count - r->r_count);

      if (i == ctx->sender_i) {

        bytes_free += (r->w_total - (r->w_count - r->r_count));
        next_sender = (i+1) % ctx->n_senders;
        r = ctx->senders[next_sender]; 
        bytes_free  += (r->w_total - (r->w_count - r->r_count));
      }

    }

    ctx->mb_free = (double) bytes_free / 1000000;
    ctx->mb_total = ((double) (ctx->n_senders * r->w_total)) / 1000000;
    mbytes_buffered = (double) bytes_buffered / 1000000;

    if (ctx->verbose)
      fprintf (stderr,"R=%4.1f, D=%4.1f, S=%4.1f [MB/s] Buffered=%4.1f MB Free=%4.1f MB r_s=%"PRIu64", s_s=%"PRIu64", OoO pkts=%"PRIu64", chids=%"PRIu64"\n", ctx->mb_rcv_ps, ctx->mb_drp_ps, ctx->mb_snd_ps, mbytes_buffered, ctx->mb_free, s_rcv_1sec, s_snd_1sec, ooo_pkts, ooo_chids);

    sleep(1);
  }
}

#ifdef HAVE_PGPLOT

/* 
 * Create a pgplot window and display buffer information
 */
void plotting_thread(void * arg) {

  udpNib_t * ctx = (udpNib_t *) arg;

  multilog(ctx->log, LOG_INFO,"plotting_thread starting\n");

  char device[6];
  sprintf(device, "%d/xs", (ctx->i_distrib+1));

  unsigned recv_bins = (ctx->n_senders*3);
  unsigned extra_bins = 6;
  unsigned n_bins = (recv_bins + extra_bins);

  if (cpgopen(device) != 1) {
    fprintf(stderr, "plotting_thread: error opening plot device [%s]\n", device);
    return ;
  }

  set_dimensions(670, 300);

  float * x_vals = malloc(sizeof(float) * n_bins);
  float * y_vals = malloc(sizeof(float) * n_bins);
  float * y_rate = malloc(sizeof(float) * n_bins);
  unsigned i = 0;

  float ymax = (float) 440;

  cpgask(0);

  struct timeval timeout;

  for (i=0; i<n_bins; i++) 
  {
    x_vals[i] = (float) i;
    y_rate[i] = 0;
  }

  uint64_t b_last = 0;
  uint64_t b_now = 0;
  uint64_t d_last = 0;
  uint64_t d_now = 0;
  uint64_t r_last = 0;
  uint64_t r_now = 0;

  float r_val = 0;
  float w_val = 0;
  float y_val = 0;
  char string[64];
  int dumps_ps = 3;
  long sleep_time = 1000000 / dumps_ps;

  cpgsvp(0.1,0.9,0.1,0.9);
  cpgswin(-0.5, n_bins-0.5, 0, ymax);

  while (!quit_plot_thread)
  {

    /* the current buffer status for each sender */
    for (i=0; i<ctx->n_senders; i++) 
    {
      r_val = ((float) ctx->senders[i]->r_count) / ((float) ctx->senders[i]->size);
      w_val = ((float) ctx->senders[i]->w_count) / ((float) ctx->senders[i]->size);

      // normalise to 400
      r_val *= 400;
      w_val *= 400;

      y_val = w_val - r_val;

      if (y_val > 0) {
        y_rate[(i*3)+0] = (y_vals[i] - y_val) * dumps_ps;
        y_rate[(i*3)+1] = 0;
        y_rate[(i*3)+2] = 0;
      }

      y_vals[(i*3)+0] = y_val;
      y_vals[(i*3)+1] = w_val;
      y_vals[(i*3)+2] = r_val;
    }

    /* current overall recv/drop rate */
    b_now = ctx->bytes->received;
    d_now = ctx->bytes->dropped;
    r_now = ctx->r_bytes;

    if (b_now > 0) 
      y_rate[recv_bins+1] = (float) ((b_now - b_last) / 1000000) * dumps_ps;
    y_vals[recv_bins+1] = y_rate[recv_bins+1];
    b_last = b_now;

    if (d_now > 0) 
      y_rate[recv_bins+2] = (float) ((d_now - d_last) / 1000000) * dumps_ps;
    y_vals[recv_bins+2] = y_rate[recv_bins+2];
    d_last = d_now;

    y_vals[recv_bins+3] = y_vals[recv_bins+1] + y_vals[recv_bins+2];
    y_rate[recv_bins+3] = y_rate[recv_bins+1] + y_rate[recv_bins+2];

    if (r_now > 0)
      y_rate[recv_bins+4] = (float) ((r_now - r_last) / 1000000) * dumps_ps;
    y_vals[recv_bins+4] = y_rate[recv_bins+4];
    r_last = r_now;

    y_rate[recv_bins+5] = 0.0;
    y_vals[recv_bins+5] = 0.0;

    cpgbbuf();
    cpgpage();
    cpgsci(1);
    cpglab("Buffers", "Capacity [Million B]", "CASPSR Memory Buffer Status");
    cpgbox("BST", 0.0, 0, "BNST", 0.0, 0);

    for (i=0; i<ctx->n_senders; i++)
    {
      sprintf(string, "%d",i);
      cpgtext((i*3)-0.1, -20, string);
    }
    sprintf(string, "CAP");
    cpgtext(recv_bins+0.5, -20, string);

    sprintf(string, "DRP");
    cpgtext(recv_bins+1.5, -20, string);

    sprintf(string, "TOT");
    cpgtext(recv_bins+2.5, -20, string);

    sprintf(string, "SND");
    cpgtext(recv_bins+3.5, -20, string);

    cpgsch(0.7);
    for (i=0; i<(n_bins-1); i++)
    {
      sprintf(string, "%3.0f", y_rate[i]);
      cpgtext(i-0.3, -40, string);
    }

    cpgsch(1.0);
    cpgsci(1);
    cpgbin(n_bins, x_vals, y_vals, 1);
    cpgebuf();

    usleep(sleep_time);
  }

  free (x_vals);
  free (y_vals);

  multilog(ctx->log, LOG_INFO,"plotting_thread exiting\n");

}

#endif

/*
 *  Simple signal handler to exit more gracefully
 */
void signal_handler(int signalValue) {

  if (quit_threads) {
    fprintf(stderr, "received signal %d twice, hard exit\n", signalValue);
    exit(EXIT_FAILURE);
  }
  quit_threads = 1;

}

#ifdef HAVE_PGPLOT

/*
 * Set the PGPLOT dimensions
 */
void set_dimensions (unsigned width_pixels, unsigned height_pixels)
{
  float width_scale, height_scale;

  get_scale (Pixels, Inches, &width_scale, &height_scale);

  float width_inches = width_pixels * width_scale;
  float aspect_ratio = height_pixels * height_scale / width_inches;

  cpgpap( width_inches, aspect_ratio );

  float x1, x2, y1, y2;
  cpgqvsz (Pixels, &x1, &x2, &y1, &y2);
}

/*
 * Get scaling factors for PGPLOT
 */

void get_scale (int from, int to, float* width, float* height)
{
  float j = 0;
  float fx, fy;
  cpgqvsz (from, &j, &fx, &j, &fy);

  float tx, ty;
  cpgqvsz (to, &j, &tx, &j, &ty);

  if (width)
    *width = tx / fx;
  if (height)
    *height = ty / fy;
}

#endif



