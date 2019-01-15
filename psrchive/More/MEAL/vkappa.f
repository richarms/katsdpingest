      FUNCTION VKAPPA(R)                                                
C ----------------------------------------------------------------
C RETURNS VKAPPA = THE MAXIMUM LIKELIHOOD ESTIMATE OF 'KAPPA', THE
C CONCENTRATION PARAMETER OF VON MISES' DISTRIBUTION OF DIRECTIONS
C IN 2 DIMENSIONS, CORRESPONDING TO A SAMPLE MEAN VECTOR MODULUS R.
C VKAPPA = K(A), THE INVERSE FUNCTION OF  A(K) = RATIO OF MODIFIED
C BESSEL FUNCTIONS OF THE FIRST KIND, VIZ., A(K) = I1(K)/I0(K).
C ----------------------------------------------------------------
C
C  FOR 8S (SIGNIFICANT DECIMAL DIGITS) PRECISION AUXILIARY ROUTINE
C  FUNCTION BESRAT(V) MUST BE SET TO AT LEAST 9.3S
      DATA V1 /0.642/, V2 /0.95/
      A = R
      S = -1.0
C
C   ERROR SIGNAL: VALUE -1.0 RETURNED IF ARGUMENT -VE OR 1.0 OR MORE.
      IF (A.LT.0.0 .OR. A.GE.1.0) GO TO 30
      Y = 2.0/(1.0-A)
      IF (A.GT.0.85) GO TO 10
C
C   FOR R BELOW 0.85 USE ADJUSTED INVERSE TAYLOR SERIES.
      X = A*A
      S = (((A-5.6076)*A+5.0797)*A-4.6494)*Y*X - 1.0
      S = ((((S*X+15.0)*X+60.0)*X/360.0+1.0)*X-2.0)*A/(X-1.0)
      IF (V1-A) 20, 20, 30
C
C   FOR R ABOVE 0.85 USE CONTINUED FRACTION APPROXIMATION.
   10 IF (A.GE.0.95) X = 32.0/(120.0*A-131.5+Y)
      IF (A.LT.0.95) X = (-2326.0*A+4317.5526)*A - 2001.035224
      S = (Y+1.0+3.0/(Y-5.0-12.0/(Y-10.0-X)))*0.25
      IF (A.GE.V2) GO TO 30
C
C   FOR R IN (0.642,0.95) APPLY NEWTON-RAPHSON, TWICE IF R IN
C   (0.75,0.875), FOR 8S PRECISION, USING APPROXIMATE DERIVATIVE -
   20 Y = ((0.00048*Y-0.1589)*Y+0.744)*Y - 4.2932
      IF (A.LE.0.875) S = (BESRAT(S)-A)*Y + S
      IF (A.GE.0.75) S = (BESRAT(S)-A)*Y + S
   30 VKAPPA = S
      RETURN
      END