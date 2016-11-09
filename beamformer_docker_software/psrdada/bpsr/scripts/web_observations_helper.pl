#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use Bpsr;
use strict;
use warnings;

my %cfg = Bpsr::getConfig();      # dada.cfg in a hash

my $cmd = "";
my $result = "";
my $response = "";
my @list = ();
my %active = ();
my $obs = "";
my @parts = ();

$cmd = "ls ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}." >& /dev/null";
($result, $response) = Dada::mySystem($cmd);

$cmd = "find ".$cfg{"SERVER_ARCHIVE_NFS_MNT"}." -mindepth 1 -maxdepth 1 -printf '\%f\n'";
($result, $response) = Dada::mySystem($cmd);

if ($result eq "ok") {

  @list = split(/\n/, $response);

  foreach $obs ( @list ) {
    $active{$obs} = 1;
  }
  
  @list = ();

  # now get the full local listing

  chdir $cfg{"CLIENT_ARCHIVE_DIR"};

  $cmd = "find . -mindepth 3 -maxdepth 3 ".
         "-type f -name 'obs.start' -printf '\%h/\%f/0\n' ".
         "-o -type f -name 'beam.transferred' -printf '\%h/\%f/0\n' ".
         "-o -type f -name 'sent.to.*' -printf '\%h/\%f/0\n' ".
         "-o -type f -name 'error.to.*' -printf '\%h/\%f/0\n' ".
         "-o -type f -name 'on.tape.*' -printf '\%h/\%f/0\n' ".
         "-o -type f -name 'integrated.ar' -printf '\%h/\%f/\%s\n' ".
         "-o -type f -name '*.fil' -printf '\%h/\%f/\%s\n' | sort";
  ($result, $response) = Dada::mySystem($cmd);
  
  if ($result eq "ok") {

    @list = split(/\n/, $response);

    foreach $obs ( @list ) {

      @parts = split(/\//, $obs);
      if ( defined $active{$parts[1]} ) {
        print $obs."\n";
      }
      @parts = ();
    }
  }
}

