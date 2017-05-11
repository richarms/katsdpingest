#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use IO::Socket;
use IO::Select;
use Net::hostent;
use threads;
use threads::shared;
use Time::Local;
use Switch;
use Dada;
use Caspsr;

Dada::preventDuplicateDaemon(basename($0));

#
# Global Variable Declarations
#
our $dl;
our $daemon_name;
our $tcs_cfg_file;
our $tcs_spec_file;
our %cfg;
our $quit_daemon : shared;
our $warn;
our $error;
our $pwcc_running : shared;
our $current_state : shared;
our $recording_start : shared;
our $pwcc_host;
our $pwcc_port;
our $client_master_port;
our %tcs_cmds;
our %site_cfg;
our $pwcc_thread;
our $utc_stop_thread;
our $utc_stop_remaining : shared;
our $utc_start_unix;
our $tcs_host;
our $tcs_port;
our $tcs_sock;


#
# Global Variable Initialization
#
%cfg           = Caspsr::getConfig();
$dl            = 1;
$daemon_name   = Dada::daemonBaseName($0);
$tcs_cfg_file  = $cfg{"CONFIG_DIR"}."/caspsr_tcs.cfg";
$tcs_spec_file = $cfg{"CONFIG_DIR"}."/caspsr_tcs.spec";
$warn = "";
$error = "";
$quit_daemon = 0;
$tcs_host = "";
$tcs_port = 0;
$tcs_sock = 0;
$pwcc_running = 0;
$current_state = "";
$recording_start = 0;
$pwcc_host = "";
$pwcc_port = 0;
$client_master_port = 0;
%tcs_cmds = ();
%site_cfg = ();
$pwcc_thread = 0;
$utc_start_unix = 0;
$utc_stop_thread = 0;
$utc_stop_remaining = -1;

#
# Constants
#
use constant PWCC_LOGFILE       => "dada_pwc_command.log";
use constant TERMINATOR         => "\r\n";

#
# Function Prototypes
#
sub main();

#
# Main
#

my $result = 0;
$result = main();

exit($result);


###############################################################################
#
# package functions
# 

sub main()
{
  $warn  = $cfg{"STATUS_DIR"}."/".$daemon_name.".warn";
  $error = $cfg{"STATUS_DIR"}."/".$daemon_name.".error";

  my $pid_file    = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";
  my $quit_file   = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $log_file    = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";

  my $server_host =     $cfg{"SERVER_HOST"};
  my $config_dir =      $cfg{"CONFIG_DIR"};

  # Connection to TCS
  $tcs_host =        $cfg{"TCS_INTERFACE_HOST"};
  $tcs_port =        $cfg{"TCS_INTERFACE_PORT"};
  my $tcs_state_port =  $cfg{"TCS_STATE_INFO_PORT"};

  # PWCC (dada_pwc_command) 
  $pwcc_host    = $cfg{"PWCC_HOST"};
  $pwcc_port    = $cfg{"PWCC_PORT"};

  # Set some global variables
  $client_master_port     = $cfg{"CLIENT_MASTER_PORT"};

  my $handle = "";
  my $peeraddr = "";
  my $hostinfo = "";  
  my $command = "";
  my $key = "";
  my $result = "";
  my $response = "";
  my $state_thread = 0;
  my $control_thread = 0;
  my $tcs_connected = 0;
  my $rh = 0;
  my $hostname = "";
  my $cmd = "";

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    return 1;
  }

  %tcs_cmds = ();
  %site_cfg = Dada::readCFGFileIntoHash($cfg{"CONFIG_DIR"}."/site.cfg", 0);

  # set initial state
  $current_state = "Idle";

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, $dl, "STARTING SCRIPT");

  Dada::logMsg(0, $dl, "Programming ibob");
  $cmd = "cat /home/dada/ib_ibob_config.txt | bibob_terminal ".$cfg{"IBOB_CONTROL_IP"}." ".$cfg{"IBOB_CONTROL_PORT"};
  Dada::logMsg(2, $dl, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "main: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsg(1, $dl, "main: ".$result." ".$response);
  }

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  foreach $key (keys (%site_cfg)) {
    Dada::logMsg(2, $dl, "site_cfg: ".$key." => ".$site_cfg{$key});
  }

  # generate the cfg file required to launch dada_pwc_command 
  ($result, $response) = generateConfigFile($tcs_cfg_file);

  # launch a persistent dada_pwc_command with the $tcs_cfg_file
  $pwcc_thread = threads->new(\&pwccThread);

  # start the stateThread
  Dada::logMsg(2, $dl, "main: stateThread()");
  $state_thread = threads->new(\&stateThread);

  my $read_set = new IO::Select();
  $read_set->add($tcs_sock);

  Dada::logMsg(2, $dl, "main: listening for TCS connection ".$tcs_host.":".$tcs_port);

  # main socket event loop
  while (! $quit_daemon) 
  {
    # get all the readable handles from the server
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles)
    {
      # we have received something on the listening socket
      if ($rh == $tcs_sock)
      {
        # only allow 1 connection from TCS
        if ($tcs_connected)
        {
          $handle = $rh->accept();
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          $handle->close();
          $handle = 0;
          Dada::logMsgWarn($warn, "rejecting additional connection from ".$hostinfo->name);
        }
        else
        {
          # accept the connection
          $handle = $rh->accept();
          $handle->autoflush(1);
          $read_set->add($handle);

          # get information about the connecting host
          $peeraddr = $handle->peeraddr;
          $hostinfo = gethostbyaddr($peeraddr);
          $hostname = $hostinfo->name;

          Dada::logMsg(1, $dl, "accepting connection from ".$hostname);
          $tcs_connected = 1;
          $handle = 0; 
        }

      # we have received something on an accepted socket
      }
      else
      {
        # get the raw "line"
        $command = <$rh>;

        # if we have lost the connection...
        if (! defined $command)
        {
          Dada::logMsg(1, $dl, "lost TCS connection from ".$hostname);
          $read_set->remove($rh);
          $rh->close();
          $tcs_connected = 0;

          # IMPORTANT 
          if (($current_state ne "Idle") && ($current_state ne "Stopped"))
          {
            # if the hostname is not the current machind 
            if (index($cfg{"SERVER_ALIASES"}, $hostname) < 0 ) 
            {
              Dada::logMsg(1, $dl, "performing STOP since TCS dissapeared");
              # use a fake file handle for the REPLY
              my $junk = "";
              open my $fh, ">", \$junk;
              processTCSCommand("stop", $fh);
              close $fh;
            }
          }
        }
        # we have received a command
        else
        {
          # clean the line up a little
          $command =~ s/\r//;
          $command =~ s/\n//;
          $command =~ s/#(.)*$//;
          $command =~ s/\s+$//;
          $command =~ s/\0//g;      # remove all null characters

          if ($command ne "")
          {
            # process the command from TCS including sending of reply
            ($result, $response) = processTCSCommand($command, $rh);
          }
          else
          {
            Dada::logMsgWarn($warn, "Received empty string from TCS");
            print $rh "ok".TERMINATOR;
          }
        }
      }
    }

    if (($current_state =~ m/^Recording/) && (time > $recording_start))
    {
      if ($utc_stop_remaining > 0)
      {
        $current_state = "Recording [".(time - $recording_start).", ".$utc_stop_remaining." remaining]";
      }
      else
      {
        $current_state = "Recording [".(time - $recording_start)." secs]";
      }
    }

    if ($current_state eq "Stopped") 
    {
      Dada::logMsg(1, $dl, "nexus now in IDLE state");
      $current_state = "Idle";
    }

    if ($utc_stop_thread && ($utc_stop_remaining < 0))
    {
      Dada::logMsg(1, $dl, "main: utc_stop_thread finished");
      $utc_stop_thread->join();
      $utc_stop_thread = 0;
    }
  }

  Dada::logMsg(0, $dl, "main: joining threads");

  # rejoin threads
  $control_thread->join();
  $pwcc_thread->join();
  $state_thread->join();

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  return 0;
}


#
# process a SINGLE command from the TCS socket
#
sub processTCSCommand($$) 
{
  my ($cmd, $tcs_sock) = @_;

  my $result = "";
  my $response = "";
  my $key = "";
  my $val = "";
  my $lckey = "";

  ($key, $val) = split(/ +/,$cmd,2);

  $lckey = lc $key;

  Dada::logMsg(1, $dl, "TCS -> ".$cmd);

  switch ($lckey)
  {
    case "start"
    {
      Dada::logMsg(2, $dl, "processTCSCommand: START");

      if (($current_state eq "Recording") || ($current_state =~ m/Starting/))
      {
        $result = "fail";
        Dada::logMsg(1, $dl, "TCS <- ".$result);
        print $tcs_sock $result.TERMINATOR;
        return ("fail", "received start command when in ".$current_state." state");
      }

      Dada::logMsg(2, $dl, "processTCSCommand: fixTCSCommands()");
      %tcs_cmds = fixTCSCommands(\%tcs_cmds);

      # check the TCS commands for validity
      Dada::logMsg(2, $dl, "processTCSCommand: parseTCSCommands()");
      ($result, $response) = parseTCSCommands();
      Dada::logMsg(2, $dl, "processTCSCommand: parseTCSCommands() ".$result ." ".$response);

      # send an immediate response to TCS so we dont get a timeout
      Dada::logMsg(1, $dl, "TCS <- ".$result);
      print $tcs_sock $result.TERMINATOR;

      if ($result ne "ok") 
      {
        Dada::logMsgWarn($error, "processTCSCommand: parseTCSCommands failed: ".$response);
        $current_state = "TCS Config Error: ".$response;
        
        # reset the tcs_cmds
        Dada::logMsg(1, $dl, "processTCSCommand: reseting TCS commands hash");
        %tcs_cmds = ();
      }
      else
      {
        # calculate the "estimated" UTC_START to TCS
        my $current_unix_time = time;
        my $guess_utc_start = "";
        $cmd = "dspsr_start_time -m 8 ".($current_unix_time + 5);
        Dada::logMsg(2, $dl, "processTCSCommand: ".$cmd);
        ($result, $response) = Dada::mySystem($cmd);
        Dada::logMsg(2, $dl, "processTCSCommand: ".$result." ".$response);
        if ($result eq "ok") 
        {
          $guess_utc_start = $response;
        }
        else
        {
          Dada::logMsg(2, $dl, "processTCSCommand: dspsr_start_time failed: ".$response);
          $guess_utc_start = Dada::printDadaUTCTime($current_unix_time + 10);
        }

        # sleep for a little bit [not 100% sure why]
        sleep(1);

        # send estimated UTC_START to TCS so it does not wait on other things
        $cmd = "start_utc ".$guess_utc_start;
        Dada::logMsg(1, $dl, "TCS <- ".$cmd." (guess at future UTC_START)");
        print $tcs_sock $cmd.TERMINATOR;

        # wait a maximum of 32 seconds for the state to return to IDLE
        my $max_wait = 32;
        while (($current_state ne "Stopped") && ($current_state ne "Idle") && ($max_wait > 0)) 
        {
          Dada::logMsg(1, $dl, "waiting for return to Idle [countdown=".$max_wait."]");
          $max_wait--;
          sleep(1);
        }
    
        $current_state = "Starting...";

        # clear the status directory
        Dada::logMsg(2, $dl, "processTCSCommand: clearStatusDir()");
        clearStatusDir();

        # Add site.config parameters to the tcs_cmds;
        Dada::logMsg(2, $dl, "processTCSCommand: addSiteConfig()");
        addSiteConfig();

        # check that the PWCC is actually running
        if (!$pwcc_running)
        {
          Dada::logMsgWarn($warn, "PWCC thread was not running, attemping to relaunch");
          $pwcc_thread->join();
          Dada::logMsg(0, $dl, "processTCSCommand: pwcc_thread was joined");
          $pwcc_thread = threads->new(\&pwccThread);
          Dada::logMsg(0, $dl, "processTCSCommand: pwcc_thread relaunched");
        }

        # create spec file for dada_pwc_command
        Dada::logMsg(2, $dl, "processTCSCommand: generateSpecificationFile(".$tcs_spec_file.")");
        ($result, $response) = generateSpecificationFile($tcs_spec_file);
        Dada::logMsg(2, $dl, "processTCSCommand: generateSpecificationFile() ".$result." ".$response);

        # issue the start command itself
        Dada::logMsg(2, $dl, "processTCSCommand: start(".$tcs_spec_file.")");
        ($result, $response) = start($tcs_spec_file);
        Dada::logMsg(2, $dl, "processTCSCommand: start() ".$result." ".$response.")");

        if ($result eq "fail")
        {
          $current_state = "Start Failed: ".$response;
        } 
        else
        {
          # determine the unix time of the utc_start
          my @t = split(/-|:/,$response); 
          $recording_start = timegm($t[5], $t[4], $t[3], $t[2], ($t[1]-1), $t[0]);

          Dada::logMsg(2, $dl, "processTCSCommand: START successful");
          $current_state = "Recording";
          Dada::logMsg(2, $dl, "processTCSCommand: STATE=Recording");
        }
      }
      %tcs_cmds = ();
    }

    case "stop" {

      Dada::logMsg(2, $dl, "Processing STOP command");

      # send an immediate reply so we dont induce a TCS timeout
      Dada::logMsg(1, $dl, "TCS <- ok");
      print $tcs_sock "ok".TERMINATOR;

      if (($current_state =~ m/Recording/) || ($current_state eq "Error")) {

        Dada::logMsg(2, $dl, "processTCSCommand: stopDemuxers()");
        ($result, $response) = stopDemuxers();
        Dada::logMsg(2, $dl, "processTCSCommand: stopDistibutors() ".$result." ".$response);
        my $utc_stop_time = $response;

        # tell the nexus to stop on the same UTC_STOP time in the future
        Dada::logMsg(2, $dl, "processTCSCommand: stopNexus(".$utc_stop_time.")");
        ($result, $response) = stopNexus($utc_stop_time);
        Dada::logMsg(2, $dl, "processTCSCommand: stopNexus() ".$result." ".$response);

        $current_state = "Stopping...";

        Dada::logMsg(2, $dl, "processTCSCommand: stopInBackground()");
        my $tmp_thr_id = threads->new(\&stopInBackground);
        $tmp_thr_id->detach();

      } elsif ($current_state eq "Idle") {

        Dada::logMsg(1, $dl, "Received additional stop command");  
        $result = "ok";

      } elsif ($current_state eq "Preparing") {
  
        Dada::logMsgWarn($warn, "Received STOP during preparing state");
        $result = "ok";

      } elsif ($current_state eq "Stopping...") {

        Dada::logMsgWarn($warn, "Received STOP whilst already Stopping");
        $result = "ok";

      } else {

        $result = "fail";
        $response = "received STOP command when in ".$current_state;

      }

    }

    case "quit" {

    } 

    # This should be a header parameter, add it to the tcs_cmds hash
    else {

      if ($key =~ m/SET_UTC_START/) {
    
        Dada::logMsgWarn($warn, "Ignoring ".$key." -> ".$val." from APSR piggyback");
        $result = "ok";
        $response = "";

      } elsif (($current_state =~ m/Starting/) || ($current_state eq "Recording")) {

        Dada::logMsgWarn($warn, "received ".$key." -> ".$val." when in ".$current_state);
        $result = "fail";
        $response = "unexpected header command when in ".$current_state;

      } else {

        $tcs_cmds{$key} = $val;
        $result = "ok";
        $response = "";

        # if we were in a TCS config error from a previous TCS start attempt, reset our
        # state so that we can try and recover
        if ($current_state =~ m/TCS Config Error/)
        {
          $current_state = "Idle";
        }

        if (($key eq "BANDWIDTH") && ($val eq "0.000000")) {
          $result = "fail";
          $response = "cowardly refusing to observe with bandwidth=0.0";
          $current_state = "Error";
        }

      }
    }
  }

  # If the command failed, log it
  if ($result eq "fail") {
    Dada::logMsgWarn($error, $response);
  }

  # Special "hack" case as we return "ok" to a start
  # command without waiting
  if ($tcs_sock && ($lckey ne "start") && ($lckey ne "stop"))
  {
    if ($result eq "fail") {
      $current_state = "Error";
      print $tcs_sock $result.TERMINATOR;
      Dada::logMsg(1, $dl, "TCS <- ".$result);

    } else {
      print $tcs_sock $result.TERMINATOR;
      Dada::logMsg(1, $dl, "TCS <- ".$result);
    }
  }
  return ($result, $response);
}
  
#
# Runs dada_pwc_command in non daemon mode. All ouput should be logged to
# the log file specified
#
sub pwccThread() {

  Dada::logMsg(1, $dl, "pwccThread: starting");

  my $result = "";
  my $response = "";
  my $cmd = "";

  if ( -f $tcs_cfg_file ) {

    $cmd = "dada_pwc_command ".$tcs_cfg_file." >> ".$cfg{"SERVER_LOG_DIR"}."/dada_pwc_command.log";
    Dada::logMsg(2, $dl, "pwccThread: ".$cmd);
    $pwcc_running = 1;
    ($result, $response) = Dada::mySystem($cmd);
    $pwcc_running = 0;
    Dada::logMsg(2, $dl, "pwccThread: ".$result." ".$response);

    if ($result ne "ok") {
      Dada::logMsgWarn($warn, "pwccThread: dada_pwc_command failed");
    }

  } else {
    Dada::logMsgWarn($warn, "pwccThread: tcs config file did not exist: ".$tcs_cfg_file);
  }

  Dada::logMsg(1, $dl, "pwccThread: exiting");
  return ($result);
}




#
# Opens a socket and reports the current state of the PWCC
#
sub stateThread() {

  Dada::logMsg(1, $dl, "stateThread: starting");

  my $host = $cfg{"SERVER_HOST"};
  my $port = $cfg{"TCS_STATE_INFO_PORT"};
  my $read_set = 0;
  my $handle = 0;
  my $line = "";
  my $rh = 0;
  my $hostname = "";
  my $hostinfo = 0;

  # open the listening socket
  Dada::logMsg(2, $dl, "stateThread: opening socket ".$host.":".$port);
  my $sock = new IO::Socket::INET (
    LocalHost => $host,
    LocalPort => $port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );

  if (!$sock) {
    Dada::logMsgWarn($error, "stateThread: could not create socket ".$host.":".$port);
    return 1;
  }

  Dada::logMsg(2, $dl, "stateThread: listening socket opened ".$host.":".$port);

  $read_set = new IO::Select();
  $read_set->add($sock);

  while (!$quit_daemon) {

    # Get all the readable handles from the read set
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);

    foreach $rh (@$readable_handles) {

      # If we are accepting a connection
      if ($rh == $sock) {
 
        $handle = $rh->accept();
        $handle->autoflush();
        $hostinfo = gethostbyaddr($handle->peeraddr);
        $hostname = $hostinfo->name;

        Dada::logMsg(3, $dl, "stateThread: Accepting connection from ".$hostname);
        $read_set->add($handle);
        $handle = 0;

      } else {

        $line = Dada::getLine($rh);

        if (! defined $line) {
          Dada::logMsg(3, $dl, "stateThread: closing read handle");
          $read_set->remove($rh);
          close($rh);

        } else {

          Dada::logMsg(3, $dl, "stateThread: received ".$line);
          if ($line eq "state") {
            print $rh $current_state."\r\n";
            Dada::logMsg(3, $dl, "stateThread: replied ".$current_state);
          }
        }
      }
    }
  }

  Dada::logMsg(2, $dl, "stateThread: exiting");
}



sub quitPWCCommand() {
  
  my $handle = 0;
  my $result = "";
  my $response = "";
  
  Dada::logMsg(2, $dl, "quitPWCCommand()");

  if (! $pwcc_running)
  {
    Dada::logMsg(2, $dl, "quitPWCCommand: dada_pwc_command not running");
    return ("ok", "");

  } else {

    Dada::logMsg(2, $dl, "quitPWCCommand: connecting to dada_pwc_command: ".$pwcc_host.":".$pwcc_port);
    $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);

    if ($handle) {

      # Ignore the "welcome" message
      $response = <$handle>;

      # Send quit command
      Dada::logMsg(2, $dl, "quitPWCCommand: sending quit to dada_pwc_command");
      print $handle "quit\r\n";
      $handle->close();

      # wait 2 seconds for the nexus to quite
      my $nwait = 2;
      while (($pwcc_running) && ($nwait > 0)) {
        sleep(1);
        $nwait--;
      }

      if ($pwcc_running) {
        Dada::logMsgWarn($warn, "Had to kill dada_pwc_command");
        ($result, $response) = Dada::killProcess("dada_pwc_command ".$tcs_cfg_file);
        Dada::logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      }

      return ("ok","");

    # try to kill the process manually
    } else {
      Dada::logMsgWarn($warn, "quitPWCCommand: could not connect to dada_pwc_command");
      ($result, $response) = Dada::killProcess("dada_pwc_command ".$tcs_cfg_file);
      Dada::logMsg(1, $dl, "quitPWCCommand: killProcess() ".$result." ".$response); 
      return ($result, $response);
    }

  }
}


#
# Send the START command to the pwcc
#
sub start($)
{
  my ($spec_file) = @_;

  my $nexus_sock = 0;
  my $rVal = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $utc_start = "UNKNOWN";

  # wait for the PWCC to be running
  while (!$pwcc_running) 
  {
    Dada::logMsg(0, $dl, "start: waiting for dada_pwc_command to start");
    sleep(1);
  }

  # connect to PWCC
  $nexus_sock = Dada::connectToMachine($pwcc_host, $pwcc_port, 5);
  if (!$nexus_sock)
  {
    Dada::logMsg(0, $dl, "start: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue START command"); 
  }
  else
  {
    # ignore the "welcome" message
    $result = <$nexus_sock>;

    # check we are in the IDLE state before continuing
    Dada::logMsg(2, $dl, "start: waiting for nexus to be in IDLE state");
    if (Dada::waitForState("idle", $nexus_sock, 5) != 0)
    {
      Dada::logMsg(0, $dl, "start: nexus was not in the idle state after 5 seconds"); 
      return ("fail", "nexus was not in IDLE state");
    }

    # send CONFIG command with 
    $cmd = "config ".$spec_file;
    Dada::logMsg(1, $dl, "nexus <- ".$cmd);
    ($result,$response) = Dada::sendTelnetCommand($nexus_sock, $cmd);
    Dada::logMsg(2, $dl, "nexus -> ".$result." ".$response);
    if ($result ne "ok")
    { 
      Dada::logMsg(0, $dl, "start: config command failed: ".$result." ".$response);
      return ("fail", "CONFIG command failed on nexus: ".$response)
    }

    # wait for the PREPARED state
    Dada::logMsg(2, $dl, "start: waiting for nexus to be in PREPARED state");
    if (Dada::waitForState("prepared",$nexus_sock,10) != 0)
    {
      Dada::logMsg(0, $dl, "start: nexus did not enter PREPARED state 10 seconds after config command");
      return ("fail", "nexus did not enter PREPARED state");
    }
    Dada::logMsg(2, $dl, "start: nexus now in PREPARED state");

    # send start command to the nexus
    $cmd = "start";
    Dada::logMsg(1, $dl, "nexus <- ".$cmd);
    ($result, $response) = Dada::sendTelnetCommand($nexus_sock, $cmd);
    Dada::logMsg(2, $dl, "nexus -> ".$result." ".$response);
    if ($result ne "ok")
    { 
      Dada::logMsg(0, $dl, "start: start command failed: ".$result." ".$response);
      return ("fail", "START command failed on nexus: ".$response);
    }

    # give the nexus a few seconds to prepare itself and open the infiniband connections
    Dada::logMsg(2, $dl, "allowing 2 seconds for IB connections to open");
    sleep(2);

    # tell the demuxers to start polling the incoming data for the reset packet
    Dada::logMsg(1, $dl, "demuxers <- START");
    Dada::logMsg(2, $dl, "start: threadedDemuxerCommand(START)");
    ($result, $response) = threadedDemuxerCommand("START");
    Dada::logMsg(2, $dl, "start: threadedDemuxerCommand() ".$result);

    # wait for the nexus to be in the recording state before starting UDP
    # data flow
    Dada::logMsg(2, $dl, "start: waiting for RECORDING state");
    if (Dada::waitForState("recording", $nexus_sock, 15) != 0)
    {
      Dada::logMsg(0, $dl, "start: nexus did not enter RECORDING state ".
                           "15 seconds after START command");
      $nexus_sock->close();
      return ("fail", "nexus did not enter RECORDING state");
    }

    # instruct the ibob to rearm and get the corresponding UTC_START
    $cmd = "bibob_start_observation -m ".$cfg{"ARCHIVE_MOD"}." ".
           $cfg{"IBOB_CONTROL_IP"}." ".$cfg{"IBOB_CONTROL_PORT"};
    Dada::logMsg(1, $dl, "start: ".$cmd);
    ($result,$response) = Dada::mySystem($cmd);
    Dada::logMsg(1, $dl, "start: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsg(0, $dl, "start: ".$cmd." failed: ".$response);
      return ("fail", "bibob_start failed");
    }
    $utc_start = $response;

    Dada::logMsg(2, $dl, "start: setUTC_START(".$utc_start.")");
    ($result, $response) = setUTC_START($nexus_sock, $utc_start);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "Failed to set UTC_START: ".$response);
      return ("fail", "Failed to set UTC_START");
    }

    $utc_start_unix = Dada::getUnixTimeUTC($utc_start);
    Dada::logMsg(2, $dl, "start: setting utc_start_unix=".$utc_start_unix." from ".$utc_start);

    # for the tcs simulator, if an observing length has been specified
    # in seconds, then start a thread to issue a UTC_STOP after this
    # time has elapsed
    if (exists($tcs_cmds{"LENGTH"})) {
      my $int = int($tcs_cmds{"LENGTH"});
      Dada::logMsg(1, $dl, "start: LENGTH=".$int);
      if ($int > 0) {
        $utc_stop_remaining = $int;
        Dada::logMsg(1, $dl, "start: will stop recording after ".$utc_stop_remaining." seconds");
        $utc_stop_thread = threads->new(\&utcStopCommand);
      }
    }

    Dada::logMsg(2, $dl, "CASPSR started with UTC_START=".$utc_start);
    return ("ok", $utc_start);
  }
}


#
# Sends a UTC_START command to the pwcc
#
sub setUTC_START($$) 
{
  my ($nexus_sock, $utc_start) = @_;
  Dada::logMsg(2, $dl, "setUTC_START(".$nexus_sock.", ".$utc_start.")");

  my $cmd = "";
  my $result = "";
  my $response = "";

  # wait for recording state
  # Dada::logMsg(2, $dl, "setUTC_START: waiting for RECORDING state");
  # if (Dada::waitForState("recording", $nexus_sock, 10) != 0)
  # {
  #   Dada::logMsg(0, $dl, "setUTC_START: nexus did not enter RECORDING state ".
  #                        "10 seconds after START command"); 
  #   $nexus_sock->close();
  #   return ("fail", "nexus did not enter RECORDING state");
  # }
  
  # send UTC Start command to the nexus
  $cmd = "set_utc_start ".$utc_start;
  
  Dada::logMsg(1, $dl, "nexus <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($nexus_sock, $cmd);
  Dada::logMsg(2, $dl, "nexus -> ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "setUTC_START: nexus returned ".$result." ".$response);
    return ("fail", $cmd." failed on nexus: ".$response);
  }
  
  # close the connection to the nexus
  $nexus_sock->close();
  $nexus_sock = 0;
  
  # send UTC Start to the demuxers
  $cmd = "SET_UTC_START ".$utc_start;
  Dada::logMsg(1, $dl, "demuxers <- ".$cmd);
  Dada::logMsg(2, $dl, "setUTC_START: threadedDemuxerCommand(".$cmd.")");
  ($result, $response) = threadedDemuxerCommand($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: threadedDemuxerCommand() ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "setUTC_START: ".$cmd." failed with ".$result." ".$response);
    return ("fail", $cmd." failed on nexus: ".$response);
  }

  my $source = "";
  my $ignore = "";
  my $i = 0;

  # Now that we know the UTC_START, create the required results and archive 
  # directories and put the observation summary file there...
  my $archive_dir = $cfg{"SERVER_ARCHIVE_DIR"}."/".$utc_start;
  my $results_dir = $cfg{"SERVER_RESULTS_DIR"}."/".$utc_start;

  # Create the observations' archive and results dirs
  $cmd = "mkdir -m 0755 ".$archive_dir;
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "setUTC_START: ".$cmd." failed: ".$response);
    return ("fail", "could not create archive_dir");
  }

  $cmd = "mkdir -m 0755 ".$results_dir;
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "setUTC_START: ".$cmd." failed: ".$response);
    return ("fail", "could not create results_dir");
  }

  # TODO make this work for multi fold sources 
  $source = $tcs_cmds{"SOURCE"};
  $source =~ s/^[JB]//;

  $cmd = "mkdir -m 0755 ".$results_dir."/".$source;
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($error, "setUTC_START: ".$cmd." failed: ".$response);
    return ("fail", "could not create results_dir");
  }
  
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $cmd = "mkdir -m 0755 ".$results_dir."/".$source."/".$cfg{"PWC_".$i};
    Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "setUTC_START: ".$cmd." failed: ".$response);
      return ("fail", "could not create ".$cfg{"PWC_".$i}."'s results_dir");
    }
  }

  my $fname = $archive_dir."/obs.info";
  Dada::logMsg(2, $dl, "setUTC_START: creating ".$fname);
  open FH, ">$fname" or return ("fail", "could not create writeable file: ".$fname);
  print FH "# Observation Summary created by: ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";
  print FH Dada::headerFormat("SOURCE",$tcs_cmds{"SOURCE"})."\n";
  print FH Dada::headerFormat("RA",$tcs_cmds{"RA"})."\n";
  print FH Dada::headerFormat("DEC",$tcs_cmds{"DEC"})."\n";
  print FH Dada::headerFormat("CFREQ",$tcs_cmds{"CFREQ"})."\n";
  print FH Dada::headerFormat("PID",$tcs_cmds{"PID"})."\n";
  print FH Dada::headerFormat("BANDWIDTH",$tcs_cmds{"BANDWIDTH"})."\n";
  print FH Dada::headerFormat("PROC_FILE",$tcs_cmds{"PROC_FILE"})."\n";
  print FH "\n";
  print FH Dada::headerFormat("MODE",$tcs_cmds{"MODE"})."\n";
  print FH Dada::headerFormat("NUM_PWC",$tcs_cmds{"NUM_PWC"})."\n";
  print FH Dada::headerFormat("NBIT",$tcs_cmds{"NBIT"})."\n";
  print FH Dada::headerFormat("NPOL",$tcs_cmds{"NPOL"})."\n";
  print FH Dada::headerFormat("NDIM",$tcs_cmds{"NDIM"})."\n";
  print FH Dada::headerFormat("RESOLUTION",$tcs_cmds{"RESOLUTION"})."\n";
  print FH Dada::headerFormat("CONFIG",$tcs_cmds{"CONFIG"})."\n";
  close FH;

  $cmd = "cp ".$fname." ".$results_dir."/";
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "could not copy obs.info to results dir: ".$response);
  }

  $cmd = "touch ".$archive_dir."/obs.processing";
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "could not create obs.process in archive dir: ".$response);
  }

  $cmd = "touch ".$results_dir."/obs.processing";
  Dada::logMsg(2, $dl, "setUTC_START: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "setUTC_START: ".$result." ".$response);
  if ($result ne "ok")
  {
    Dada::logMsgWarn($warn, "Could not create obs.process in results dir: ".$response);
  }

  Dada::logMsg(2, $dl, "setUTC_START: observation directories initialized");

  return ("ok", "");
}

################################################################################
#
# Ask the Demuxers to Stop on the time in the future
#
sub stopDemuxers()
{

  Dada::logMsg(2, $dl, "stopDemuxers()");

  my $result = "";
  my $response = "";
  my $host = "";
  my $port = $cfg{"DEMUX_CONTROL_PORT"};
  my @handles = ();
  my $i = 0;

  # open sockets to each demuxer
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) 
  {
    $host = $cfg{"DEMUX_".$i};
    $handles[$i] = Dada::connectToMachine($host, $port);
    if (!$handles[$i])
    {
      Dada::logMsgWarn($error, "stopDemuxers: could not connect to ".$host.":".$port);
      return ("fail", "");
    }
  }

  # set stopping time to 8 seconds in the future
  my $stop_time_unix = time + 8;
  if ($stop_time_unix < ($utc_start_unix + 32)) 
  {
    Dada::logMsg(1, $dl, "stopDemuxers: overriding stop time to minimum of 32 seconds");
    $stop_time_unix = $utc_start_unix + 32;
  }

  my $stop_time = Dada::printDadaUTCTime($stop_time_unix);
  my $cmd = "UTC_STOP ".$stop_time;

  Dada::logMsg(1, $dl, "demuxers <- ".$cmd); 

  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++)
  {
    $host =  $cfg{"DEMUX_".$i};
    Dada::logMsg(2, $dl, $host.":".$port." <- ".$cmd);
    ($result,$response) = Dada::sendTelnetCommand($handles[$i], $cmd);
    Dada::logMsg(2, $dl, $host.":".$port." -> ".$result." ".$response);
    if ($result ne "ok")
    {
      Dada::logMsgWarn($error, "stopDemuxers: ".$host.":".$port." <- ".$cmd." returned ".$response);
      return ("fail", "");
    }
  }

  Dada::logMsg(1, $dl, "demuxers -> ok");

  # close sockets
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++)
  {
    $handles[$i]->close();
  }

  return ("ok", $stop_time);
}


###############################################################################
#
# Ask the nexus to stop
#
sub stopNexus($) 
{

  my ($utc_stop) = @_;

  Dada::logMsg(2, $dl, "stopNexus(".$utc_stop.")");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;
  
  Dada::logMsg(2, $dl, "stopNexus: opening connection to ".$pwcc_host.":".$pwcc_port);
  $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);
  if (!$handle) {
    Dada::logMsg(0, $dl, "stopNexus: could not connect to dada_pwc_command ".$pwcc_host.":".$pwcc_port);
    return ("fail", "could not connect to nexus to issue STOP <UTC_STOP>");
  }

   # Ignore the "welcome" message
  $ignore = <$handle>;
  
  $cmd = "stop ".$utc_stop;
  
  Dada::logMsg(1, $dl, "nexus <- ".$cmd);
  ($result, $response) = Dada::sendTelnetCommand($handle, $cmd);
  Dada::logMsg(2, $dl, "stopNexus: nexus -> ".$result." ".$response);
  if ($result ne "ok") { 
    Dada::logMsg(0, $dl, "stopNexus: ".$cmd." failed: ".$result." ".$response);
    $response = $cmd." command failed on nexus";
  }
  Dada::logMsg(1, $dl, "nexus -> ".$result);

  $handle->close();

  return ($result, $response);

}

###############################################################################
#
# stop the observation in a background thread.
#
sub stopInBackground() {

  Dada::logMsg(2, $dl, "stopInBackground()");

  my $ignore = "";
  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;
  my $i = 0;

  Dada::logMsg(2, $dl, "stopInBackground: opening connection to ".$pwcc_host.":".$pwcc_port);
  $handle = Dada::connectToMachine($pwcc_host, $pwcc_port);
  if (!$handle) {
    Dada::logMsgWarn($error, "Could not connect to nexus to wait for IDLE state");
    $current_state = "Error";
    Dada::logMsg(2, $dl, "stopInBackground:  exiting");
    return 1;
  }

   # Ignore the "welcome" message
  $ignore = <$handle>;

  # Check we are in the IDLE state before continuing
  Dada::logMsg(1, $dl, "nexus waiting for IDLE state");
  if (Dada::waitForState("idle", $handle, 40) != 0) {
    Dada::logMsgWarn($error, "stopInBackground: nexus was not in the idle state after 40 seconds");
    $current_state = "Error";
  } else {
    Dada::logMsg(2, $dl, "stopInBackground: nexus now in idle state");
    $current_state = "Stopped";
  }

  # Close nexus connection
  $handle->close();

  Dada::logMsg(2, $dl, "stopInBackground:  exiting");
  return 0;

}

###############################################################################
#
# Send threaded commands to each of the demuxers
#
sub threadedDemuxerCommand($) {

  my ($cmd) = @_;

  my $i = 0;
  my @threads = ();
  my @results = ();
  my $host = "";
  my $port = $cfg{"DEMUX_CONTROL_PORT"};
  
  # start a thread for each demuxer 
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    $host = $cfg{"DEMUX_".$i};
    Dada::logMsg(2, $dl, "threadedDemuxerCommand: sending ".$cmd." to ".$host.":".$port);
    $threads[$i] = threads->new(\&demuxerThread, $host, $port, $cmd);
  }

  # join each thread
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    Dada::logMsg(2, $dl, "threadedDemuxerCommand: joining thread ".$i);
    $results[$i] = $threads[$i]->join();
  }

  Dada::logMsg(2, $dl, "threadedDemuxerCommand: all threads joined");

  # check the results
  my $overall_result = "ok";
  for ($i=0; $i<$cfg{"NUM_DEMUX"}; $i++) {
    if ($results[$i] ne "ok") {
      Dada::logMsgWarn($error, "threadedDemuxerCommand: ".$cmd." on ".
                       $cfg{"DEMUX_".$i}.":".$port." failed");
      $overall_result = "fail";
    }
  }
  return ($overall_result, ""); 

}

###############################################################################
#
# sends a command to the specified machine:port
#
sub demuxerThread($$$) {

  my ($host, $port, $cmd) = @_;
  
  my $handle = 0;
  my $result = "";
  my $response = "";

  $handle = Dada::connectToMachine($host, $port);
  if (!$handle) {
    Dada::logMsgWarn($error, "demuxerThread: could not connect to ".$host.":".$port);
    return "fail";
  }

  Dada::logMsg(2, $dl, $host.":".$port." <- ".$cmd);
  ($result,$response) = Dada::sendTelnetCommand($handle, $cmd);
  Dada::logMsg(2, $dl, $host.":".$port." -> ".$result." ".$response);

  if ($result ne "ok") {
    Dada::logMsgWarn($error, "demuxerThread: ".$cmd." failed: ".$response);
    return "fail";
  }

  $handle->close();

  return "ok";

}

###############################################################################
# 
# trim preceeding whitespace
# 
sub ltrim($)
{
  my $string = shift;
  $string =~ s/^\s+//;
  return $string;
}

sub getParameterFromArray($\@) {
                                                                                
  (my $parameter, my $arrayRef) = @_;
                                                                                
  my @array = @$arrayRef;

  # Generate the key/value combinations for the specification
  my @arr;
  my $line;
  my $value = "";
  my $i = 0;

  for ($i=0; $i<=$#array; $i++) {
    $line = $array[$i];

    # strip and comments
    $line =~ s/#.*//;

    if ($line =~ m/^$parameter /) {
      @arr = split(/ +/,$line);
      $value = $arr[1];
      chomp($value);
    }

  }
  return $value;
}


#
# Checks that TCS supplied us with the MINIMAL set of commands 
# necessary to run 
# 
sub parseTCSCommands() {

  my $result = "ok";
  my $response = "";

  my @cmds = qw(SOURCE RA DEC RECEIVER CFREQ PID NBIT NDIM NPOL BANDWIDTH PROC_FILE MODE RESOLUTION);
  my $cmd;

  if (exists $tcs_cmds{"MODE"}) {
    if ($tcs_cmds{"MODE"} eq "CAL") {
      push(@cmds, "CALFREQ");
    }
  }

  foreach $cmd (@cmds) {
    if (!(exists $tcs_cmds{$cmd})) {
      $result = "fail";
      $response .= " ".$cmd;
    } 
    elsif (!(defined $tcs_cmds{$cmd}))
    {
      $result = "fail";
      $response .= " ".$cmd;
    }
    elsif ($tcs_cmds{$cmd} eq "")
    {
      $result = "fail";
      $response .= " ".$cmd;
    }
    else
    {
      Dada::logMsg(2, $dl, "parseTCSCommands: found header parameter ".$cmd);
    }

  }
  if ($result eq "fail") {
    Dada::logMsg(0, $dl, "parseTCSCommands: missing header parameter(s) ".$response);
    return ("fail", "Missing Parameter(s)".$response);
  }

  # Check that the PROC_FILE exists in the CONFIG_DIR
  if (! -f $cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"} ) {
    Dada::logMsg(0, $dl, "parseTCSCommands: PROC_FILE [".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}."] did not exist");
    return ("fail", "PROC_FILE ".$tcs_cmds{"PROC_FILE"}." did not exist");
  }

  my $source   = $tcs_cmds{"SOURCE"};
  my $mode     = $tcs_cmds{"MODE"};
  my $proc_bin = "";

  $cmd = "grep PROC_CMD ".$cfg{"CONFIG_DIR"}."/".$tcs_cmds{"PROC_FILE"}." | awk '{print \$2}'";
  Dada::logMsg(2, $dl, "parseTCSCommands: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "parseTCSCommands: ".$result." ".$response);

  if ($result ne "ok") { 
    Dada::logMsg(0, $dl,  "parseTCSCommands: could not extract the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
    return ("fail", "could not determine the binary from the PROC_FILE [".$tcs_cmds{"PROC_FILE"}."]");
  }

  $proc_bin = $response;
  if (!-f $cfg{"SCRIPTS_DIR"}."/".$proc_bin) {
    Dada::logMsg(0, $dl,  "parseTCSCommands: binary [".$proc_bin."] specified in PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] did not exist in the bin dir [".$cfg{"SCRIPTS_DIR"}."]");
    return ("fail", "PROC_FILE [".$tcs_cmds{"PROC_FILE"}."] contain an invalid binary");
  }

  # Check the SOURCE, MODE make sense for a DSPSR based PROC_FILE
  if (($mode eq "PSR") && ($proc_bin =~ m/dspsr/)) {
    ($result, $response) = Dada::inCatalogue($source);
    if ($result eq "fail")
    {
      return ("fail", $response);
    }
  }

  # check for MULTI fold 
  if ($tcs_cmds{"PROC_FILE"} eq "dspsr.multi") {

    my $short_source = $source;
    $short_source =~ s/^[JB]//;
    $short_source =~ s/[a-zA-Z]*$//;

    # find the source in multi.txt
    $cmd = "grep ^".$short_source." ".$cfg{"CONFIG_DIR"}."/multi.txt";
    my $multi_string = `$cmd`;
    if ($? != 0) {
      Dada::logMsg(0, $dl,  "parseTCSCommands: SOURCE [".$short_source."] did not exist in ".$cfg{"CONFIG_DIR"}."/multi.txt");
      return ("fail", "SOURCE [".$source."] did not exist in CASPSRs multifold list");

    } else {

      chomp $multi_string;
      my @multis = split(/ +/,$multi_string);

      if (! -f $cfg{"CONFIG_DIR"}."/".$multis[2]) {
        Dada::logMsg(0, $dl,  "parseTCSCommands: Multi-source file [".$cfg{"CONFIG_DIR"}."/".$multis[2]."] did not exist");
        return ("fail", "The multifold source file [".$multis[2]."] did not exist");
      }
    }
  }

  return ("ok", "");

}


sub fixTCSCommands(\%) {

  my ($tcs_cmds_ref) = @_;

  my %cmds = %$tcs_cmds_ref;

  my %fix = ();
  $fix{"src"} = "SOURCE";
  $fix{"ra"} = "RA";
  $fix{"dec"} = "DEC";
  $fix{"band"} = "BANDWIDTH";
  $fix{"freq"} = "CFREQ";
  $fix{"procfil"} = "PROC_FILE";
  $fix{"PROCFIL"} = "PROC_FILE";
  $fix{"pid"} = "PID";
  $fix{"receiver"} = "RECEIVER";
  $fix{"observer"} = "OBSERVER";
  $fix{"refbeam"} = "REFBEAM";
  $fix{"nbeam"} = "NBEAM";
  $fix{"obsval"} = "OBSVAL";
  $fix{"obsunit"} = "OBSUNIT";
  
  my %add = (); 
  $add{"MODE"} = "PSR";
  $add{"CALFREQ"} = "11.123000";
  $add{"NBIT"} = "8";
  $add{"NPOL"} = "2";
  $add{"NDIM"} = "1";
  $add{"RESOLUTION"} = "1";

  my %new_cmds = ();

  my $key = "";

  foreach $key (keys (%cmds)) {

    if (exists $fix{$key}) {
      $new_cmds{$fix{$key}} = $cmds{$key};
    } else {
      $new_cmds{$key} = $cmds{$key};
    }
  }
 
  # Use JR's formula for 50cm observing
  if (abs($new_cmds{"BANDWIDTH"}) == 64) 
  {
    my $ibw = $new_cmds{"BANDWIDTH"} < 0 ? +1 : -1;
    $new_cmds{"CFREQ"} += (104.0 * $ibw);
    $new_cmds{"BANDWIDTH"} = 400.0 * $ibw;
  }
  else
  {
    # always set bandwidth to -400 irrespective of TCS
    # $new_cmds{"BANDWIDTH"} = "-400";
    #if (($new_cmds{"CFREQ"} > 1200) && ($new_cmds{"CFREQ"} < 1500)) 
    #{
    #  $new_cmds{"CFREQ"} = 1382;
    #}
  }

  if (exists $new_cmds{"SOURCE"})
  {
    if (($new_cmds{"SOURCE"} =~ m/_R$/) || ($new_cmds{"SOURCE"} =~ m/HYDRA/) || ($new_cmds{"SOURCE"} =~ m/CalDelay/)) {
      $add{"MODE"} = "CAL";

      if ($new_cmds{"SOURCE"} =~ m/CalDelay/) {
        $add{"CALFREQ"} = "200.0";
      }
    }
  }

  if ($new_cmds{"CFREQ"} == 628)
  {
    $new_cmds{"TDEC"} = 2;
  }

  foreach $key (keys (%add)) {
    if (!(exists $new_cmds{$key})) {
      $new_cmds{$key} = $add{$key};
    }
  }
  
  return %new_cmds;
}


#
# Addds the required keys,values to the TCS commands based
# on the hardwired constants of the DFB3. These are:
#
# 1. Always sending to 16 PWCs
# 2. Lowest freq band goes to apsr00, highest to apsr15
# 3. 16 Bands (for the moment)
#
sub addSiteConfig() {

  Dada::logMsg(2, $dl, "addSiteConfig()");

  my $key = "";
  my $bw = 0;
  my $i = 0;

  $tcs_cmds{"NUM_PWC"}     = $cfg{"NUM_PWC"};
  $tcs_cmds{"HDR_SIZE"}    = $site_cfg{"HDR_SIZE"};
  $tcs_cmds{"BW"}          = $tcs_cmds{"BANDWIDTH"};
  $tcs_cmds{"FREQ"}        = $tcs_cmds{"CFREQ"};

  # Add the site configuration to tcs_cmds
  foreach $key (keys (%site_cfg)) {
    $tcs_cmds{$key} = $site_cfg{$key};
  }

  $bw = $tcs_cmds{"BW"};
  # Determine the TSAMP based upon NDIM and BW 
  if ($bw == 0.0) {
    $tcs_cmds{"TSAMP"} = 0.0
  } else {
    $tcs_cmds{"TSAMP"} = (1.0 / abs($bw)) * ($tcs_cmds{"NDIM"} / 2);
  }

  # number of channels and bands hardcoded to 1
  $tcs_cmds{"NBAND"} = 1;
  $tcs_cmds{"NCHAN"} = 1;

  # Set the instrument
  $tcs_cmds{"INSTRUMENT"} = uc($cfg{"INSTRUMENT"});
    
}


#
# Generates the config file required for dada_pwc_command
#
sub generateConfigFile($) {

  my ($fname) = @_;

  my $string = "";
  
  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);

  print FH "# Header file created by ".$daemon_name."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  $string = Dada::headerFormat("NUM_PWC", $cfg{"NUM_PWC"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  # Port information for dada_pwc_command
  $string = Dada::headerFormat("PWC_PORT", $cfg{"PWC_PORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWC_LOGPORT", $cfg{"PWC_LOGPORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWCC_PORT", $cfg{"PWCC_PORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("PWCC_LOGPORT", $cfg{"PWCC_LOGPORT"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("LOGFILE_DIR", $cfg{"SERVER_LOG_DIR"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("HDR_SIZE", $site_cfg{"HDR_SIZE"});
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  $string = Dada::headerFormat("COM_POLL", "10");
  print FH $string."\n";
  Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);

  my $i=0;
  for($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $string = Dada::headerFormat("PWC_".$i, $cfg{"PWC_".$i});
    print FH $string."\n";
    Dada::logMsg(2, $dl, $tcs_cfg_file." ".$string);
  }
  close FH;

  return ("ok", "");

}

#
# Generate the specification file used in the dada_pwc_command's CONFIG command
#
sub generateSpecificationFile($) {

  my ($fname) = @_;

  open FH, ">".$fname or return ("fail", "Could not write to ".$fname);
  print FH "# Specification File created by ".$0."\n";
  print FH "# Created: ".Dada::getCurrentDadaTime()."\n\n";

  my %ignore = ();
  $ignore{"NUM_PWC"} = "yes";
  my $i=0;
  for ($i=0; $i<$tcs_cmds{"NUM_PWC"}; $i++) {
    $ignore{"PWC_".$i} = "yes";
  }

  # Print the keys
  my @sorted = sort (keys %tcs_cmds);

  my $line;
  foreach $line (@sorted) {
    if (!(exists $ignore{$line})) {
      print FH Dada::headerFormat($line, $tcs_cmds{$line})."\n";
      Dada::logMsg(2, $dl, $tcs_spec_file." ".Dada::headerFormat($line, $tcs_cmds{$line}));
    }
  }

  close FH;
  return ("ok","");
}
  

#
# delete all the files in the STATUS_DIR
#
sub clearStatusDir() {

  my $cmd = "";
  my $result = "";
  my $response = "";

  # Clear the /apsr/status files
  $cmd = "rm -f ".$cfg{"STATUS_DIR"}."/*";
  
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok") {
    Dada::logMsgWarn($warn, "Could clean STATUS_DIR: ".$response);
  }

}


###############################################################################
#
# Thread that issues the STOP command after the specified number of seconds have
# elapsed
#
sub utcStopCommand() {

  my $result = "";
  my $response = "";

  Dada::logMsg(2, $dl ,"utcStopCommand: stopping in ".$utc_stop_remaining." seconds");

  while ( (!$quit_daemon) && ($utc_stop_remaining > 0) && 
          (($current_state =~ m/Starting/) || ($current_state =~ m/Recording/)) ) {

    sleep 1;
    $utc_stop_remaining--;
    Dada::logMsg(2, $dl ,"utcStopCommand: ".$utc_stop_remaining." seconds remaining");

  }

  # if we have successfully timed out 
  if ( ($utc_stop_remaining == 0) && ($current_state =~ m/Recording/) ) {
    my $dud_handle = 0;
    Dada::logMsg(2, $dl ,"utcStopCommand: processTCSCommand('stop', ".$dud_handle.")");
    ($result, $response) = processTCSCommand("stop", $dud_handle);
    Dada::logMsg(2, $dl ,"utcStopCommand: processTCSCommand ".$result." ".$response);
  } else {
    Dada::logMsg(1, $dl ,"utcStopCommand: waiting loop ended prematurely");
  }

  $utc_stop_remaining = -1;

  return 0;

}

###############################################################################
#
# Try to reboot CASPSR when the backend is in an erroneous state
#
sub rebootCaspsr() 
{
  my $cmd = "";
  my $result = "";
  my $response = "";

  # first try a sane/soft stop
  $cmd = "caspsr_reconfigure.pl -s";
  Dada::logMsg(1, $dl , "rebootCaspsr: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(1, $dl , "rebootCaspsr: ".$result." ".$response);

  # then do a hard_stop just to be sure
  $cmd = "caspsr_hard_reset.pl";
  Dada::logMsg(1, $dl , "rebootCaspsr: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(1, $dl , "rebootCaspsr: ".$result." ".$response);

}


###############################################################################
#
#
#
sub controlThread($$) {

  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  Dada::logMsg(2, $dl ,"controlThread: quit detected");

  # Manually tell dada_pwc_command to quit
  quitPWCCommand();

  Dada::logMsg(2, $dl ,"controlThread: PWCC has exited");

  # ensure the global is set
  $quit_daemon = 1;

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(2, $dl ,"controlThread: exiting");

  return 0;
}
  


#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(5);
  print STDERR $daemon_name." : Exiting\n";
  exit 1;
  
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) {

  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

} 

#
# Test to ensure all module variables are set before main
#
sub good($) {

  my ($quit_file) = @_;
  my $cmd = "";
  my $result = "";
  my $response = "";

  # check the quit file does not exist on startup
  if (-f $quit_file) {
    return ("fail", "Error: quit file ".$quit_file." existed at startup");
  }

  # the calling script must have set this
  if (! defined($cfg{"INSTRUMENT"})) {
    return ("fail", "Error: package global hash cfg was uninitialized");
  }

  # this script can *only* be run on the configured server
  if (index($cfg{"SERVER_ALIASES"}, Dada::getHostMachineName()) < 0 ) {
    return ("fail", "Error: script must be run on ".$cfg{"SERVER_HOST"}.
                    ", not ".Dada::getHostMachineName());
  }

  # check IBOB connectivity
  $cmd = "ping -q -c 1 ".$cfg{"IBOB_CONTROL_IP"};
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    return ("fail", "Could not ping IBOB at ".$cfg{"IBOB_CONTROL_IP"}." - try manual reset of IBOB");
  }

  # check IBOB response to commands
  $cmd = "echo 'regread ip_ctr/reg_num_ips\nquit' | bibob_terminal ".
         $cfg{"IBOB_CONTROL_IP"}." ".$cfg{"IBOB_CONTROL_PORT"};
  ($result, $response) = Dada::mySystem($cmd);
  if ($result ne "ok")
  {
    return ("fail", "Could not interact with IBOB at ".$cfg{"IBOB_CONTROL_IP"}.":".
                    $cfg{"IBOB_CONTROL_PORT"}." - try manual reset of IBOB");
  }

  # check that the response is correct
  if (!($response =~ m/Received response in 63 bytes/))
  {
    return ("fail", "IBOB did not response to a control command, try ".
            "manual reset of IBOB");
  }

  # check the Demux nodes are receiving 800MB/s each on eth2
  my $handle = 0;
  my @bits = ();
  my @rates = ();
  my $lines_parsed = 0;
  my $lines_matched = 0;

  my $wait_for_ibob = 60;
  my $data_rates_ok = 0;

  while ((!$data_rates_ok) && ($wait_for_ibob > 0))
  {
    $handle = Dada::connectToMachine("demux0", "8650");
    @bits = ();
    @rates = ();

    while ($result = <$handle>) 
    {
      $lines_parsed++;
      if ($result =~ m/\"bytes_in\"/)
      { 
        $lines_matched++;
        @bits = split(/ /,$result);
        @bits = split(/"/,$bits[2]);
        push @rates, $bits[1];
      } 
    }
    $handle->close();

    # check the values returned from ganglia
    if ($#rates == 1)
    {
      if (($rates[0] >= 800000000) && ($rates[1] >= 800000000))
      {
        $data_rates_ok = 1;
      }
    }
    else
    {
      return ("fail", "could not extract input data rate metrics from ganglia on demux0");
    }

    # if the rates are bad
    if (!$data_rates_ok)
    {
      # and this is our first attempt
      if ($wait_for_ibob == 60)
      {
        $cmd = "cat /home/dada/ib_ibob_config.txt | bibob_terminal ".$cfg{"IBOB_CONTROL_IP"}." ".$cfg{"IBOB_CONTROL_PORT"};
        ($result, $response) = Dada::mySystem($cmd);
      }
      $wait_for_ibob--;
      sleep(1);
    }
  }

  if (!$data_rates_ok)
  {
    return ("fail", "Demuxer nodes not receiving 800MB/s each, reset IBOB");
  }

  $tcs_sock = new IO::Socket::INET (
    LocalHost => $tcs_host,
    LocalPort => $tcs_port,
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1
  );
  if (!$tcs_sock) {
    return ("fail", "Could not create listening socket: ".$tcs_host.":".$tcs_port);
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  # clear any warnings or errors associated with this daemon
  if (-f $warn)
  {
    unlink $warn;
  }
  if (-f $error)
  {
    unlink $error;
  }

  return ("ok", "");
}
