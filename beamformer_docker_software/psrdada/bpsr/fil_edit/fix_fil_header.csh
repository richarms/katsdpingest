#!/bin/csh
set inf = $1
set fil = $2
set filedit=$3
set nbeams=$4

set ra_deg = `cat $inf | grep '<start_coordinate>' -A 3 | grep ra | sed -e 's:.*>\([0-9\.]*\)<.*:\1:'`
set dec_deg = `cat $inf | grep '<start_coordinate>' -A 3 | grep dec | sed -e 's:.*>\(-*[0-9.]*\)<.*:\1:'`
set ibeam = `cat $inf  | grep '<receiver_beam>'| sed -e 's:.*>\(-*[0-9.]*\)<.*:\1:'`

set ra_tot = `echo ${ra_deg} | awk '{printf ("%f", $1/15)}'`  #total ra hours
set ra_h = `echo ${ra_tot} | awk '{print int($1)}' | awk '{printf "%02d",$1}'` #hours
set ra_min = `echo ${ra_tot} | awk '{print int(($1-int($1))*60)}' | awk '{printf "%02d",$1}'` #min
set ra_sec = `echo $ra_tot $ra_min | awk '{print int((($1-int($1))*60 - $2)*60)}' | awk '{printf "%02d",$1}'` #sec
set ra_rest = `echo $ra_tot $ra_min $ra_sec | awk '{print ((($1-int($1))*60 - $2)*60) -$3}' | awk '{printf "%0.2f",$1}' | sed -e 's:.::'`

#echo ra =  ${ra_h}:${ra_min}:${ra_sec}${ra_rest}
set ra = ${ra_h}${ra_min}${ra_sec}${ra_rest}

set dec_h = `echo ${dec_deg} | awk '{print int($1)}' | awk '{printf "%02d",$1}'` #hours
set dec_neu = `echo ${dec_deg} | sed -e 's:-::'`
set dec_min = `echo ${dec_neu} | awk '{print int(($1-int($1))*60)}' | awk '{printf "%02d",$1}'` #min
set dec_sec = `echo ${dec_neu} ${dec_min} | awk '{print int((($1-int($1))*60 - $2)*60)}' | awk '{printf "%02d",$1}'` #sec
set dec_rest = `echo ${dec_neu} ${dec_min} ${dec_sec} | awk '{print ((($1-int($1))*60 - $2)*60) -$3}' | awk '{printf "%0.1f",$1}' | sed -e 's:.::'`

#echo dec = ${dec_h}:${dec_min}:${dec_sec}${dec_rest}
set dec = ${dec_h}${dec_min}${dec_sec}${dec_rest}

set cmd="$filedit  --ra $ra --dec $dec --beam $ibeam --nbeams $nbeams $fil"
echo $cmd
$cmd
