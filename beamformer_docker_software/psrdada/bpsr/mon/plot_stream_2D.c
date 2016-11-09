/***************************************************************************/
/*                                                                         */
/* function plot_stream_2D                                                 */
/*                                                                         */
/*                                                                         */
/***************************************************************************/

#include "plot4mon.h"

int plot_stream_2D(float plotarray[], int nbin_x, int nsub_y, float yscale,
	       int firstdump_line,
               char inpdev[], char xlabel[], char ylabel[], char plottitle[])
{
  long kk=0;
  long nvalues,inivalue;
  float min_value,max_value,cutoff;
  float trf[6] = {-0.5, 1.0, 0.0, 
                  -0.5, 0.0, 1.0};
  float x[nbin_x];

  min_value = 0.0;
  max_value = 1.0;

   /*
   *   Rescale the value of the input array according to yscale
   */
  nvalues=(long)nbin_x*(long)nsub_y;
  while(kk<=nvalues-1) 
  {
     plotarray[kk]=plotarray[kk]/yscale;
     kk++;
  }
  /*
   *  Compute the minimum and maximum value of the input array
   */
  inivalue=0;
  compute_extremes(plotarray,nvalues,inivalue,&max_value,&min_value);
  cutoff=0.5;

  int blank=0;
  if (min_value == max_value == 0.00) {
    fprintf(stderr, "compute_extremes returned equal min/max values\n");
    min_value = 0.0;
    max_value = 0.1;
    blank = 1;
  }

  /*
   * Reporting number of values and extremes of the plot
   */
#ifndef PLOT4MON_QUIET
  printf("\n About to plot %ld values using pgplot \n", nvalues);
  printf(" 2D margins : X going from 0 to %d \n",nbin_x); 
  printf("            : Y going from 0 to %d \n",nsub_y); 
  printf(" Extremes: VALUEmin = %f     VALUEmax = %f \n",min_value,max_value);
#endif
  /*
   * Call cpgbeg to initiate PGPLOT and open the output device; 
   */
  if(cpgbeg(0, inpdev, 1, 1) != 1) return -2;
  /*
   * Call cpgenv to specify the range of the axes and to draw a box, and
   * cpglab to label it, with line size given by cpslw (default 1)
   * and text height given by cpsch (default 1.0) 
   */
  cpgenv(0.0, (float)nbin_x, 0.0, (float)nsub_y, 0, 0);
  cpgslw(4);
  cpgsch(1.2);
  cpglab(xlabel,ylabel,plottitle);
  /*
   * Call cpggray to plot the greyscale map whose rotation is given by trf
   */
  cpggray(plotarray, nbin_x, nsub_y, 1, nbin_x, 1, nsub_y, 
         max_value*cutoff, min_value, trf);
 
  if (firstdump_line) 
  {
    /* Call cpgline to join the points of the first row with a line */
    for (kk=0; kk<=nbin_x-1; kk++) x[kk]=(float)kk-1.0;
    cpgslw(2);
    cpgline(nbin_x-1, x, plotarray);
  }

  /*
   * Finally, call cpgend to terminate things properly.
   */
  cpgend();
#ifndef PLOT4MON_QUIET
  printf(" Plotting completed \n");
#endif
  return 0;
}
