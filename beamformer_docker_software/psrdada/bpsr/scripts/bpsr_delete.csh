#!/bin/csh -f

if ( "$1" == "" ) then
  echo Please specify the UTC of the observation to be deleted
  exit -1
endif

set UTC=$1
set RESULTS=/nfs/results/bpsr
set ARCHIVES=/nfs/archives/bpsr

echo "This script will delete all BPSR data for $UTC"
echo "Press <Enter> to continue or <Ctrl-C> to abort"

$<

if ( ! -d $RESULTS/$UTC ) then
  echo $RESULTS/$UTC directory not found ... aborting.
  exit -1
endif

if ( ! -d $ARCHIVES/$UTC ) then
  echo $ARCHIVES/$UTC directory not found ... aborting.
  exit -1
endif

if ( -f $ARCHIVES/$UTC/obs.single ) then
  echo $ARCHIVES/$UTC/obs.single exists ... aborting.
  exit -1
endif


touch $ARCHIVES/$UTC/obs.deleted
touch $RESULTS/$UTC/obs.deleted

echo Deleting BPSR beams 01 through 13 of $UTC

foreach beam ( 01 02 03 04 05 06 07 08 09 10 11 12 13 )

  if ( ! -d $ARCHIVES/$UTC/$beam ) then
    echo $ARCHIVES/$UTC/$beam not found ... aborting.
    exit -1
  endif

  set node=`ls -l $ARCHIVES/$UTC/$beam | awk -F"->" '{print $2}' | awk -F/ '{print $3}'`

  echo Deleting beam $beam on node $node

  rm -f $ARCHIVES/$UTC/$beam

  ssh -x -f bpsr@$node "rm -rf /lfs/data0/bpsr/archives/$UTC"
    
end

