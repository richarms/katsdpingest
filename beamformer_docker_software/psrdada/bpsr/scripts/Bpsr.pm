package Bpsr;

use lib $ENV{"DADA_ROOT"}."/bin";

use Dada;
use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Math::BigInt;
use Math::BigFloat;
use strict;
use vars qw($VERSION @ISA @EXPORT @EXPORT_OK);
use Sys::Hostname;
use Time::Local;
use POSIX qw(setsid);

require Exporter;
require AutoLoader;

@ISA = qw(Exporter AutoLoader);

@EXPORT_OK = qw(
  &waitForMultibobState
  &getMultibobState
  &configureMultibobServer
  &clientCommand
  &getObsDestinations
  &getConfig
  &getIBOBConfig
  &getROACHConfig
  &getBeamForPWCHost
);

$VERSION = '0.01';

my $DADA_ROOT = $ENV{'DADA_ROOT'};

use constant DEBUG_LEVEL  => 0;

sub waitForMultibobState($$$) {

  (my $stateString, my $handle, my $Twait) = @_;

  my $pwc;
  my @pwcs;
  my $myready = "no";
  my $counter = $Twait;
  my $i=0;

  if (DEBUG_LEVEL >= 1) {
    print $stateString." ".$Twait."\n";
  }
  while (($myready eq "no") && ($counter > 0)) {


    if ($counter == $Twait) {
      ;
    } elsif ($counter == ($Twait-1)) {
      if (DEBUG_LEVEL >= 1) { print STDERR "Waiting for ibobs to become  $stateString."; }
    } else {
      if (DEBUG_LEVEL >= 1) { print STDERR "."; }
    }

    $myready = "yes";

    (@pwcs) = getMultibobState($handle);

    for ($i=0; $i<=$#pwcs;$i++) {
      $pwc = @pwcs[$i];
      if ($pwc ne $stateString) {
        if (DEBUG_LEVEL >= 1) {
          print "Waiting for IBOB".$i." to transition to ".$stateString."\n";
        }
        $myready = "no";
      }
    }

    sleep 1;
    $counter--;
  }

  if (($counter+1) != $Twait) {
    if (DEBUG_LEVEL >= 1) { print STDERR "\n"; }
  }

  if ($myready eq "yes") {
    return ("ok", "");
  } else {
    return ("fail", "");
  }

}

sub getMultibobState($) {

  (my $handle) = @_;
  my $result = "fail";
  my $response = "";

  ($result, $response) = Dada::sendTelnetCommand($handle, "state");

  if ($result eq "ok") {
    #Parse the $response;
    my @array = split('\n',$response);
    my $line;
    my @temp_array;

    my @pwcs;
    foreach $line (@array) {

      if (index($line,"> ") == 0) {
        $line = substr($line,2);
      }

      if (index($line,"IBOB") == 0) {
        @temp_array = split(" ",$line);
        push (@pwcs, $temp_array[2]);
      }
    }

    return (@pwcs);
  } else {
    return 0;
  }

}


sub configureMultibobServer() {

  my %cfg   = Bpsr::getConfig();
  my %ibobs = Bpsr::getIBOBConfig();
  my %roaches  = Bpsr::getROACHConfig();

  # multibob_server runs on localhost
  my $host = Dada::getHostMachineName();
  my $port = $cfg{"IBOB_MANAGER_PORT"};

  my $result;
  my $response;

  my $handle = Dada::connectToMachine($host, $port, 10);

  if (!$handle) {
    return ("fail", "Could not connect to multibob_server ".$host.":".$port);
  }

  # ignore welcome message
  $response = <$handle>;

  logMessage(1, "multibob <- close");
  ($result, $response) = Dada::sendTelnetCommand($handle, "close");
  if ($result ne "ok") {
    logMessage(0, "multibob close command failed");
    return ($result, $response);
  } else {
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  # get the current hostports configuration

  ($result, $response) = Bpsr::waitForMultibobState("closed", $handle, 10);
  if ($result ne "ok") {
    logMessage(0, "multibob did not close successfully");
    return ($result, $response);
  }

  # get the current hostports configuration
  logMessage(1, "multibob <- hostports");
  ($result, $response) = Dada::sendTelnetCommand($handle, "hostports");
  if ($result ne "ok") {
    logMessage(0, "multibob hostports command failed");
    return ($result, $response);
  } else {
    $response =~ s/\n$//;
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  # setup the IBOB host/port mappings
  my $cmd = $cfg{"NUM_PWC"};
  my $i=0;
  my $j=0;
  my $ibob = 0;
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $ibob = 0;
    for ($j=0; $j<$ibobs{"NUM_IBOB"}; $j++) {
      if ($cfg{"PWC_".$i} eq $ibobs{"10GbE_CABLE_".$j}) {
        if ($ibob) {
          logMessage(1, "ERROR: configuration error in ibob config file");
        }
        $ibob = $j;
      }
    }
    $cmd .= " ".$i." ".$ibobs{"CONTROL_IP_".$ibob}." 23";
  }

  # if the hostports config on the ibob isn't what we require
  if ($cmd ne $response) {
    $cmd = "hostports ".$cmd;

    logMessage(1, "multibob <- ".$cmd);
    ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
    if ($result ne "ok") {
      logMessage(0, "multibob hostports command failed");
      return ($result, $response);
    }
  }

  logMessage(1, "multibob <- open");
  ($result, $response) = Dada::sendTelnetCommand($handle, "open");
  if ($result ne "ok") {
    logMessage(0, "multibob open command failed");
    return ($result, $response);
  } else {
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  ($result, $response) = Bpsr::waitForMultibobState("alive", $handle, 60);
  if ($result ne "ok") {
    logMessage(0, "multibob threads did not come alive after 60 seconds");
    return ($result, $response);
  } else {
    logMessage(0, "multibob threads now alive");
  }

  # setup the IBOB mac addresses
  $cmd = "macs ".$cfg{"NUM_PWC"};
  my $mac = "";
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    if (exists($ibobs{"10GbE_MAC_".$cfg{"PWC_".$i}})) {
      $mac = $ibobs{"10GbE_MAC_".$cfg{"PWC_".$i}};
      $mac =~ s/://g;
      $cmd .= " ".$i." ".$mac;
    } else {
      logMessage(1, "ERROR: configuration error in ibob config file, 10GbE_MAC_".$cfg{"PWC_".$i}." did not exist in ibobs hash");
    }
  
  }

  logMessage(1, "multibob <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  if ($result ne "ok") {
    logMessage(0, "multibob macs command failed");
    return ($result, $response);
  } else {
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  $cmd = "acclen 25";
  logMessage(1, "multibob <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  if ($result ne "ok") {
    logMessage(0, "multibob acclen command failed");
    return ($result, $response);
  } else {
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  $cmd = "levels";
  logMessage(1, "multibob <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  if ($result ne "ok") {
    logMessage(0, "multibob levels command failed");
    return ($result, $response);
  } else {
    logMessage(1, "multibob -> ".$result." ".$response);
  }

  $cmd = "exit";
  logMessage(1, "multibob <- ".$cmd);
  print $handle $cmd."\r\n";

  close($handle);

  return ("ok", "");
}

sub logMessage($$) {
  
  my ($level, $msg) = @_;

  if ($level <= 2) {
     print "[".Dada::getCurrentDadaTime(0)."] ".$msg."\n";
  }

}

sub clientCommand($$) {

  my ($command, $machine) = @_;

  my %cfg = Bpsr::getConfig();
  my $result = "fail";
  my $response = "Failure Message";

  my $handle = Dada::connectToMachine($machine, $cfg{"CLIENT_MASTER_PORT"}, 0);
  # ensure our file handle is valid
  if (!$handle) {
    return ("fail","Could not connect to machine ".$machine.":".$cfg{"CLIENT_MASTER_PORT"});
  }

  ($result, $response) = Dada::sendTelnetCommand($handle,$command);

  $handle->close();

  return ($result, $response);

}

# Return the destinations that an obs with the specified PID should be sent to
sub getObsDestinations($$) {
  
  my ($obs_pid, $dests) = @_;
  
  my $want_swin = 0;
  my $want_parkes = 0;
  
  if ($dests =~ m/swin/) {
    $want_swin = 1;
  }
  if ($dests =~ m/parkes/) {
    $want_parkes = 1;
  }

  return ($want_swin, $want_parkes);

}

sub getConfig() 
{
  my $config_file = $DADA_ROOT."/share/bpsr.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);

  my $pwc_config_file = $DADA_ROOT."/share/bpsr_pwcs.cfg";
  my %pwc_config = Dada::readCFGFileIntoHash($pwc_config_file, 0);

  my %combined = (%config, %pwc_config);
  return %combined;
}

sub getIBOBConfig()
{
  my $config_file = $DADA_ROOT."/share/ibob.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);
  return %config;
}

sub getROACHConfig()
{
  my $config_file = $DADA_ROOT."/share/roach.cfg";
  my %config = Dada::readCFGFileIntoHash($config_file, 0);
  return %config;
}


sub getBeamForPWCHost($) 
{
  my ($host) = @_;

  my %ibobs = Bpsr::getIBOBConfig();
  my $beam = 0;
  my $i = 0;
  my $test_host = "";

  for ($i=0; $i<$ibobs{"NUM_IBOB"}; $i++) 
  {
    $test_host = $ibobs{"10GbE_CABLE_".$i};
    if ($host =~ m/$test_host/)
    {
      $beam = $ibobs{"BEAM_".$i};
    }
  } 

  return $beam;

}

__END__
