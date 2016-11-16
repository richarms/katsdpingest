//-*-C++-*-
/***************************************************************************
 *
 *   Copyright (C) 2003 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

/* $Source: /cvsroot/dspsr/dspsr/Kernel/Formats/puma/dsp/PuMaFile.h,v $
   $Revision: 1.7 $
   $Date: 2008/05/28 21:12:43 $
   $Author: straten $ */


#ifndef __dsp_PuMaFile_h
#define __dsp_PuMaFile_h

#include "dsp/File.h"

namespace dsp {

  //! Loads TimeSeries data from a PuMa data file.
  /*! This class is heavily copied from PuMaData by Russell Edwards. */
  class PuMaFile : public File 
  {
  public:
   
    //! Construct and open file
    PuMaFile (const char* filename = 0);

    //! Destructor
    ~PuMaFile ();

    //! Returns true if filename appears to name a valid PuMa file
    bool is_valid (const char* filename) const;

  protected:

    //! Open the file
    void open_file (const char* filename);

    //! Set the Input::info attribute, given a loaded Header_type object
    void parse (const void* hdr);

  private:

    //! The PuMa header
    void* header;

  };

}

#endif // !defined(__PuMaFile_h)
  
