#!/usr/bin/env perl

# 
# Simple MOPSR processing script
#
#   Runs the antenna splitter on RECVING data block
# 
# Author:   Andrew Jameson
# 

use lib $ENV{"DADA_ROOT"}."/bin";

#
# Include Modules
#
use strict;
use warnings;

use Mopsr;          # DADA Module for configuration options

use File::Basename; 
use threads;         # standard perl threads
use threads::shared; # standard perl threads

#
# Function Prototypes
#
sub logMsg($$$);



sub usage() 
{
  print "Usage: ".basename($0)." PWC_ID\n";
  print "   PWC_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variable Declarations
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $pwc_id : shared;
our $in_db_key : shared;
our %cfg : shared;
our $log_host;
our $sys_log_port;
our $src_log_port;
our $sys_log_sock;
our $src_log_sock;
our $sys_log_file;
our $src_log_file;

#
# Initialize globals
#
$dl = 1;
$quit_daemon = 0;
$daemon_name = Dada::daemonBaseName($0);
$pwc_id = 0;
$in_db_key = "";
%cfg = Mopsr::getConfig();
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_SRC_LOG_PORT"};
$sys_log_sock = 0;
$src_log_sock = 0;
$sys_log_file = "";
$src_log_file = "";

#
# Local Variable Declarations
#
my $pid_file = "";
my $control_thread = 0;
my $prev_header = "";
my $out_db_key;

#
# Check command line arguments is 1
#
if ($#ARGV != 0) 
{
  usage();
  exit(1);
}
$pwc_id  = $ARGV[0];

# ensure that our pwc_id is valid 
if (($pwc_id >= 0) &&  ($pwc_id < $cfg{"NUM_PWC"}))
{
  # and matches configured hostname
  if (($cfg{"PWC_".$pwc_id} eq Dada::getHostMachineName()) || ($cfg{"PWC_".$pwc_id} eq "localhost"))
  {
    # determine the relevant PWC based configuration for this script 
    $in_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"RECEIVING_DATA_BLOCK"});
    $out_db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $pwc_id, $cfg{"NUM_PWC"}, $cfg{"ANTSELECTED_DATA_BLOCK"});
  }
  else
  {
    print STDERR "PWC_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "PWC_ID was not a valid integer between 0 and ".($cfg{"NUM_PWC"}-1)."\n";
  usage();
  exit(1);
}

# Also check that we are an ACTIVE or PASSIVE PWC
if (($cfg{"PWC_STATE_".$pwc_id} ne "active") && ($cfg{"PWC_STATE_".$pwc_id} ne "passive"))
{
  print STDOUT "Config file specified PWC_STATE_".$pwc_id."=".$cfg{"PWC_STATE_".$pwc_id}.", not starting\n";
  exit(0);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$pwc_id);


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd, $curr_raw_header, $prev_raw_header);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$pwc_id.".src.log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".pid";

  # register Signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($sys_log_file, $pid_file);

  # Auto flush output
  $| = 1;

  # Open a connection to the server_sys_monitor.pl script
  $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  if (!$sys_log_sock) {
    print STDERR "Could open sys log port: ".$log_host.":".$sys_log_port."\n";
  }

  $src_log_sock = Dada::nexusLogOpen($log_host, $src_log_port);
  if (!$src_log_sock) {
    print STDERR "Could open src log port: ".$log_host.":".$src_log_port."\n";
  }

  logMsg(1,"INFO", "STARTING SCRIPT");

  # this thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

	$curr_raw_header = "";
	$prev_raw_header = "";
	my %header;

  # Main Loop
  while (!$quit_daemon) 
  {
		%header = ();

		# next header to read from the receiving data_block
    $cmd =  "dada_header -k ".$in_db_key;
    logMsg(2, "INFO", "main: ".$cmd);
    ($result, $curr_raw_header) = Dada::mySystem($cmd);
    logMsg(2, "INFO", "main: ".$cmd." returned");

    if ($result ne "ok")
    {
      if ($quit_daemon)
      {
        logMsg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
        logMsg(0, "ERROR", "dada_header failed: ".$curr_raw_header);
        $quit_daemon = 1;
      }
    }

    if (!$quit_daemon)
    {
      $cmd = "ls -1d ".$cfg{"CONFIG_DIR"};
      logMsg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      logMsg(3, "INFO", "main: ".$cmd." config_dir=".$response);
      if ($response ne $cfg{"CONFIG_DIR"})
      {
        logMsg(0, "ERROR", "NFS automount for ".$cfg{"CONFIG_DIR"}." failed: ".$response);
        $quit_daemon = 1;
      }

      # default processing command
      $proc_cmd = "mopsr_dbselectantsdb ".$in_db_key." ".$out_db_key." 4 5 6 7 -s -z";

      if ($curr_raw_header eq $prev_raw_header)
      {
        logMsg(0, "ERROR", "main: header repeated, jettesioning observation");
        $proc_cmd = "dada_dbnull -k ".$in_db_key." -s -z";
      }
      else
      {
        %header = Dada::headerToHash($curr_raw_header);

        logMsg(1, "INFO", "START ".$proc_cmd);
        ($result, $response) = Dada::mySystemPiped ($proc_cmd, $src_log_file, 
                                                    $src_log_sock, "src", 
                                                    sprintf("%02d",$pwc_id), $daemon_name, 
                                                    "ants");
        logMsg(1, "INFO", "END   ".$proc_cmd);

        if ($result ne "ok")
        {
          logMsg(0, "ERROR", "processing failed, exiting");
          $quit_daemon = 1;
        }
      }

		  $prev_raw_header = $curr_raw_header;	
    }
  }

  logMsg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  logMsg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}


sub controlThread($) 
{
  my ($pid_file) = @_;

  logMsg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $pwc_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$pwc_id.".quit";

  my ($cmd, $result, $response);

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $pwc_quit_file))) 
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my $user = "mpsr";
  my $process = "^dada_header -k ".$in_db_key;

  logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
  ($result, $response) = Dada::killProcess($process, $user);
  logMsg(3, "INFO", "controlThread: killProcess ".$result." ".$response);
  if ($result ne "ok")
  {
    logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
  }

  my @processes_to_kill = ();
  push @processes_to_kill, "^dada_dbnull -k ".$in_db_key;
  push @processes_to_kill, "^mopsr_dbselectantsdb ".$in_db_key;

  foreach $process ( @processes_to_kill)
  {
    logMsg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    logMsg(2, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      logMsg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  if ( -f $pid_file) 
  {
    logMsg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    logMsg(1, "INFO", "controlThread: PID file did not exist on script exit");
  }

  logMsg(1, "INFO", "controlThread: exiting");
}


#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub logMsg($$$) 
{
  my ($level, $type, $msg) = @_;
  if ($level <= $dl) 
  {
    # remove backticks in error message
    $msg =~ s/`/'/;

    my $time = Dada::getCurrentDadaTime();
    if (!($sys_log_sock))
    {
      $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
    }
    if ($sys_log_sock)
    {
      Dada::nexusLogMessage($sys_log_sock, $pwc_id, $time, "sys", $type, "obs mngr", $msg);
    }
    print "[".$time."] ".$msg."\n";
  }
}


sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";

  # if we CTRL+C twice, just hard exit
  if ($quit_daemon) {
    print STDERR $daemon_name." : Recevied 2 signals, Exiting\n";
    exit 1;

  # Tell threads to try and quit
  } else {

    $quit_daemon = 1;
    if ($sys_log_sock) {
      close($sys_log_sock);
    }
    if ($src_log_sock) {
      close($src_log_sock);
    }
  }
}

sub sigPipeHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $sys_log_sock = 0;
  if ($log_host && $sys_log_port) {
    $sys_log_sock = Dada::nexusLogOpen($log_host, $sys_log_port);
  }
}
