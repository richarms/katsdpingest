//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma/dsp/PuMaTwoBitCorrection.h,v $
   $Revision: 1.2 $
   $Date: 2006/07/09 13:27:08 $
   $Author: wvanstra $ */

#ifndef __PuMaTwoBitCorrection_h
#define __PuMaTwoBitCorrection_h

#include "dsp/SubByteTwoBitCorrection.h"

namespace dsp {
  
  //! Converts PuMa data from 2-bit digitized to floating point values
  class PuMaTwoBitCorrection: public SubByteTwoBitCorrection {

  public:

    //! Constructor initializes base class atributes
    PuMaTwoBitCorrection ();

    //! Return true if PuMaTwoBitCorrection can convert the Observation
    virtual bool matches (const Observation* observation);

  };
  
}

#endif
