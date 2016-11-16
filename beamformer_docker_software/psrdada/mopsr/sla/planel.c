#include "slalib.h"
#include "slamac.h"
void slaPlanel ( double date, int jform, double epoch, double orbinc,
                 double anode, double perih, double aorq, double e,
                 double aorl, double dm, double pv[6], int* jstat )
/*
**  - - - - - - - - - -
**   s l a P l a n e l
**  - - - - - - - - - -
**
**  Heliocentric position and velocity of a planet, asteroid or comet,
**  starting from orbital elements.
**
**  Given:
**     date    double     date, Modified Julian Date (JD - 2400000.5)
**     jform   int        choice of element set (1-3; Note 3)
**     epoch   double     epoch of elements (TT MJD)
**     orbinc  double     inclination (radians)
**     anode   double     longitude of the ascending node (radians)
**     perih   double     longitude or argument of perihelion (radians)
**     aorq    double     mean distance or perihelion distance (AU)
**     e       double     eccentricity
**     aorl    double     mean anomaly or longitude (radians, jform=1,2 only)
**     dm      double     daily motion (radians, jform=1 only)
**
**  Returned:
**     pv      double[6]  heliocentric x,y,z,xdot,ydot,zdot of date,
**                         J2000 equatorial triad (AU,AU/s)
**     jstat   int*       status:  0 = OK
**                                -1 = illegal jform
**                                -2 = illegal e
**                                -3 = illegal aorq
**                                -4 = illegal dm
**                                -5 = numerical error
**
**  Called:  slaEl2ue, slaUe2pv
**
**  Notes
**
**  1  The argument "date" is the instant for which the prediction is
**     required.  It is in the TT timescale (formerly Ephemeris Time,
**     ET) and is a Modified Julian Date (JD-2400000.5).
**
**  2  The elements are with respect to the J2000 ecliptic and equinox.
**
**  3  A choice of three different element-set options is available:
**
**     Option jform = 1, suitable for the major planets:
**
**       epoch  = epoch of elements (TT MJD)
**       orbinc = inclination i (radians)
**       anode  = longitude of the ascending node, big omega (radians)
**       perih  = longitude of perihelion, curly pi (radians)
**       aorq   = mean distance, a (AU)
**       e      = eccentricity, e (range 0 to <1)
**       aorl   = mean longitude L (radians)
**       dm     = daily motion (radians)
**
**     Option jform = 2, suitable for minor planets:
**
**       epoch  = epoch of elements (TT MJD)
**       orbinc = inclination i (radians)
**       anode  = longitude of the ascending node, big omega (radians)
**       perih  = argument of perihelion, little omega (radians)
**       aorq   = mean distance, a (AU)
**       e      = eccentricity, e (range 0 to <1)
**       aorl   = mean anomaly M (radians)
**
**     Option jform = 3, suitable for comets:
**
**       epoch  = epoch of elements and perihelion (TT MJD)
**       orbinc = inclination i (radians)
**       anode  = longitude of the ascending node, big omega (radians)
**       perih  = argument of perihelion, little omega (radians)
**       aorq   = perihelion distance, q (AU)
**       e      = eccentricity, e (range 0 to 10)
**
**     Unused arguments (dm for jform=2, aorl and dm for jform=3) are
**     not accessed.
**
**  4  Each of the three element sets defines an unperturbed
**     heliocentric orbit.  For a given epoch of observation, the
**     position of the body in its orbit can be predicted from these
**     elements, which are called "osculating elements", using standard
**     two-body analytical solutions.  However, due to planetary
**     perturbations, a given set of osculating elements remains usable
**     for only as long as the unperturbed orbit that it describes is an
**     adequate approximation to reality.  Attached to such a set of
**     elements is a date called the "osculating epoch", at which the
**     elements are, momentarily, a perfect representation of the
**     instantaneous position and velocity of the body.
**
**     Therefore, for any given problem there are up to three different
**     epochs in play, and it is vital to distinguish clearly between
**     them:
**
**     . The epoch of observation:  the moment in time for which the
**       position of the body is to be predicted.
**
**     . The epoch defining the position of the body:  the moment in
**       time at which, in the absence of purturbations, the specified
**       position (mean longitude, mean anomaly, or perihelion) is
**       reached.
**
**     . The osculating epoch:  the moment in time at which the given
**       elements are correct.
**
**     For the major-planet and minor-planet cases it is usual to make
**     the epoch that defines the position of the body the same as the
**     epoch of osculation.  Thus, only two different epochs are
**     involved:  the epoch of the elements and the epoch of
**     observation.
**
**     For comets, the epoch of perihelion fixes the position in the
**     orbit and in general a different epoch of osculation will be
**     chosen.  Thus, all three types of epoch are involved.
**
**     For the present routine:
**
**     . The epoch of observation is the argument date.
**
**     . The epoch defining the position of the body is the argument
**       epoch.
**
**     . The osculating epoch is not used and is assumed to be close
**       enough to the epoch of observation to deliver adequate
**       accuracy.  If not, a preliminary call to slaPertel may be
**       used to update the element-set (and its associated osculating
**       epoch) by applying planetary perturbations.
**
**  5  The reference frame for the result is with respect to the mean
**     equator and equinox of epoch J2000.
**
**  6  The algorithm was originally adapted from the EPHSLA program
**     of D.H.P.Jones (private communication, 1996).  The method is
**     based on Stumpff's Universal Variables.
**
**  Reference:  Everhart, E. & Pitkin, E.T., Am.J.Phys. 51, 712, 1983.
**
**  Last revision:   3 January 2003
**
**  Copyright P.T.Wallace.  All rights reserved.
*/
{
   double u[13];
   int j;



/* Validate elements and convert to "universal variables" parameters. */
   slaEl2ue ( date, jform,
              epoch, orbinc, anode, perih, aorq, e, aorl, dm, u, &j );

/* Determine the position and velocity. */
   if ( !j ) {
      slaUe2pv ( date, u, pv, &j );
      if ( j ) j = -5;
   }

/* Wrap up. */
   *jstat = j;

}
