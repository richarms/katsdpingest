/* $Source: /cvsroot/psrdada/psrdada/src/tmutil.h,v $
   $Revision: 1.4 $
   $Date: 2016/09/29 08:55:49 $
   $Author: ajameson $ */

#ifndef DADA_UTC_H
#define DADA_UTC_H

#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE /* glibc2 needs this for strptime  */
#endif
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

  /*! parse a string into struct tm; return equivalent time_t */
  time_t str2tm (struct tm* time, const char* str);

  /*! parse a string and return equivalent time_t */
  time_t str2time (const char* str);

  /*! parse a UTC string and return equivalent time_t */
  time_t str2utctime (const char* str);

  /*! parse a UTC time string into struct tm; return equivalent time_t */
  time_t str2utctm (struct tm* time, const char* str);

  /*! convert a UTC MJD time into the struct tm */
  time_t mjd2utctm (double mjd);

  void float_sleep (float seconds);

#ifdef __cplusplus
}
#endif

#endif

