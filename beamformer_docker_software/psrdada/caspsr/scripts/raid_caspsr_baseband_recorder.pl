#!/usr/bin/env perl

##############################################################################
#  
#     Copyright (C) 2011 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################
#
# Records baseband caspsr observations
#

#
# Constants
#
use constant META_DIR       => "/lfs/data0/caspsr";
use constant REQUIRED_HOST  => "raid0";
use constant LOG_PORT       => "39921";
use constant REQUIRED_USER  => "caspsr";

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use File::Basename;
use threads;
use threads::shared;
use Thread::Queue;
use Dada;
use Caspsr;

#
# function prototypes
#
sub good($);

#
# global variable definitions
#
our $dl;
our $daemon_name;
our %cfg;
our $quit_daemon : shared;
our $log_host : shared;
our $log_sock;
our $log_port : shared;
our $client_logger;
our $warn;
our $error;

#
# initialize globals
#
$dl = 2;
$daemon_name = Dada::daemonBaseName(basename($0));
%cfg = Caspsr::getConfig();
$warn = ""; 
$error = ""; 
$quit_daemon = 0;
$log_host = REQUIRED_HOST;
$log_port = LOG_PORT;
$log_sock = 0;
$client_logger = "client_caspsr_baseband_logger.pl";

{
  $warn  = META_DIR."/logs/".$daemon_name.".warn";
  $error = META_DIR."/logs/".$daemon_name.".error";

  my $log_file    = META_DIR."/logs/".$daemon_name.".log";
  my $pid_file    = META_DIR."/control/".$daemon_name.".pid";
  my $quit_file   = META_DIR."/control/".$daemon_name.".quit";

  my $cmd = "";
  my $result = "";
  my $response = "";
  my $control_thread = 0;
  my $i = 0;

  # sanity check on whether the module is good to go
  ($result, $response) = good($quit_file);
  if ($result ne "ok") {
    print STDERR $response."\n";
    exit 1;
  }

  # install signal handlers
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;
  $SIG{PIPE} = \&sigPipeHandle;

  # become a daemon
  Dada::daemonize($log_file, $pid_file);
  
  Dada::logMsg(0, $dl ,"STARTING SCRIPT");

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # create a thread queue to handle the incoming log messages for the
  # logging thread to handle
  my $in = new Thread::Queue;

  # start the control thread
  Dada::logMsg(2, $dl, "main: controlThread(".$quit_file.", ".$pid_file.")");
  $control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

  # start the logging thread
  Dada::logMsg(2, $dl, "starting loggingThread");
  my $logging_thread = threads->new(\&loggingThread, $log_file, $in);

  # start the comms thread
  Dada::logMsg(2, $dl, "starting CommsThread");
  my $comms_thread = threads->new(\&commsThread, $in);

  Dada::logMsg(1, $dl, "Starting CASPSR Baseband Recorder");

  # ensure the required datablocks are freshly
  my @hosts =  ("gpu0",  "gpu2",  "gpu1",  "gpu3");
  my @dbkeys = ("ca00",  "ca20",  "ca10",  "ca30");
  my @ports =  ("40000", "40002", "40001", "40003");
  my @disks =  ("/lfs/raid0/caspsr/baseband", "/lfs/raid0/caspsr/baseband", "/lfs/raid1/caspsr/baseband", "/lfs/raid1/caspsr/baseband");

  my $db_nbufs = 4;
  my $db_bufsz = 256000000;

  # kill any datablocks of the current user
  $cmd = "ipcrme";
  Dada::logMsg(2, $dl, "main: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl, "main: ".$result." ".$response);

  # create required datablocks of the current user
  for ($i=0; $i<=$#dbkeys; $i++)
  {
    $cmd = "dada_db -b ".$db_bufsz." -n ".$db_nbufs." -k ".$dbkeys[$i]." -l";
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "main: ".$result." ".$response);
  }

  # launch dbdisk threads for each host
  my @dbdisk_threads = ();
  my $core = 0;
  for ($i=0; $i<=$#hosts; $i++)
  {
    $core = 1 + (2 * $i); 
    Dada::logMsg(2, $dl, "main: dbdiskThread(".$hosts[$i].", ".$dbkeys[$i].", ".$disks[$i].", ".$core.")");
    $dbdisk_threads[$i] = threads->new(\&dbdiskThread, $hosts[$i], $dbkeys[$i], $disks[$i], $core);
  }

  # launch ibdb threads for each host
  my @ibdb_threads = ();
  for ($i=0; $i<=$#hosts; $i++)
  {
    $core = (2 * $i);
    Dada::logMsg(2, $dl, "main: ibdbThread(".$hosts[$i].", ".$dbkeys[$i].", ".$ports[$i].", ".$core.")");
    $ibdb_threads[$i] = threads->new(\&ibdbThread, $hosts[$i], $dbkeys[$i], $ports[$i], $core);
  }

  Dada::logMsg(2, $dl, "main: waiting for exit");
  while (!$quit_daemon)
  {
    sleep(1);
  }

  Dada::logMsg(2, $dl, "main: joining dbdisk and ibdb threads");
  for ($i=0; $i<=$#hosts; $i++)
  {
    $dbdisk_threads[$i]->join();
    $ibdb_threads[$i]->join();
  }

  # delete all datablock resources
  for ($i=0; $i<=$#dbkeys; $i++)
  {
    $cmd = "dada_db -d -k ".$dbkeys[$i];
    Dada::logMsg(2, $dl, "main: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl, "main: ".$result." ".$response);
  }

  Dada::logMsg(0, $dl, "STOPPING SCRIPT");

  # rejoin threads
  $control_thread->join();
  $comms_thread->join();
  $logging_thread->join();

  exit 0;
}


###############################################################################
#
# Functions
#

sub dbdiskThread($$$$) 
{
  my ($gpu, $dbkey, $path, $core) = @_;

  Dada::logMsg(1, $dl ,"dbdiskThread [".$gpu."] starting");

  my $cmd = "";
  my $result = "";
  my $response = "";

  if ( ! -d $path."/".$gpu )
  {
    $cmd = "mkdir -p ".$path."/".$gpu;
    Dada::logMsg(1, $dl ,"dbdiskThread [".$gpu."] ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, $dl ,"dbdiskThread [".$gpu."] ".$result." ".$response);
  }

  $cmd = "dada_dbdisk -b ".$core." -k ".$dbkey." -D ".$path."/".$gpu." -t 23068672 -z -o ".
         "2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." disk_".$gpu;

  # run this command directly, soas output is printed to log file
  Dada::logMsg(1, $dl ,"dbdiskThread [".$gpu."] ".$cmd);
  Dada::logMsg(1, $dl ,"dbdiskThread [".$gpu."] ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl ,"dbdiskThread [".$gpu."] ".$result." ".$response);

  if ($? == 0)
  {
    Dada::logMsg(2, $dl ,"dbdiskThread [".$gpu."] ok");
    return 0;
  }
  else
  {
    Dada::logMsg(2, $dl ,"dbdiskThread [".$gpu."] failed");
    return -1;
  }
}

sub ibdbThread($$$$) 
{ 

  my ($gpu, $dbkey, $port, $core) = @_;
    
  Dada::logMsg(1, $dl ,"ibdbThread [".$gpu."] starting");
  
  my $cmd = "";
  my $result = "";
  my $response = "";
  
  $cmd = "dada_ibdb -b ".$core." -c 16384 -k ".$dbkey." -p ".$port." ".
         "2>&1 | ".$cfg{"SCRIPTS_DIR"}."/".$client_logger." ibdb_".$gpu;
  Dada::logMsg(1, $dl ,"ibdbThread [".$gpu."] ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, $dl ,"ibdbThread [".$gpu."] ".$result." ".$response);

  Dada::logMsg(1, $dl ,"ibdbThread [".$gpu."] exiting");
  if ($result eq "ok")
  {
    return 0;
  }
  else
  {
    return -1;
  }
}


sub controlThread($$) 
{
  Dada::logMsg(1, $dl ,"controlThread: starting");

  my ($quit_file, $pid_file) = @_;

  Dada::logMsg(2, $dl ,"controlThread(".$quit_file.", ".$pid_file.")");

  # Poll for the existence of the control file
  while ((!(-f $quit_file)) && (!$quit_daemon)) {
    sleep(1);
  }

  # ensure the global is set
  $quit_daemon = 1;

  my $cmd = "";
  my $result = "";
  my $response = "";

  $cmd = "^dada_dbdisk";
  Dada::logMsg(2, $dl ,"controlThread: killProcess(".$cmd.", caspsr)");
  ($result, $response) = Dada::killProcess($cmd, "caspsr");
  Dada::logMsg(2, $dl ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^dada_dbnull";
  Dada::logMsg(2, $dl ,"controlThread: killProcess(".$cmd.", caspsr)");
  ($result, $response) = Dada::killProcess($cmd, "caspsr");
  Dada::logMsg(2, $dl ,"controlThread: killProcess() ".$result." ".$response);

  $cmd = "^dada_ibdb";
  Dada::logMsg(2, $dl ,"controlThread: killProcess(".$cmd.", caspsr)");
  ($result, $response) = Dada::killProcess($cmd, "caspsr");
  Dada::logMsg(2, $dl ,"controlThread: killProcess() ".$result." ".$response);

  if ( -f $pid_file) {
    Dada::logMsg(2, $dl ,"controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, $dl ,"controlThread: exiting");
  return 0;
}
  

###############################################################################
#
# listens on the required socket
#
sub commsThread($) 
{
  my ($in) = @_;
  Dada::logMsg(1, $dl, "commsThread: starting");

  my ($rh, $read_set, $handle, $domain);

  # create a read set for handle connections and data 
  $read_set = new IO::Select();
  $read_set->add($log_sock);

  Dada::logMsg(1, $dl, "commsThread: waiting for connections on ".$log_host.":".$log_port);
  
  my $read_something = 0;
  my $poll_handle = 0;
  my $line = "";
  
  while (!$quit_daemon) 
  {
    # Get all the readable handles from the log_sock 
    my ($readable_handles) = IO::Select->select($read_set, undef, undef, 1);
  
    foreach $rh (@$readable_handles)
    {
      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $log_sock) {

        $handle = $rh->accept();

        $handle->autoflush(1);

        # get the hostname for this connection
        Dada::logMsg(3, $dl, "commsThread: accepting connection");

        $read_set->add($handle);
        $handle = 0;

      }
      else
      {
  
        # get the hostname for this connection
        Dada::logMsg(3, $dl, "commsThread: processing message");

         # set the input record seperator to \r\n
        $/ = "\r\n";
  
        # make the socket non blocking
        $rh->blocking(0);

        $read_something = 0;
        $poll_handle = 1;

        # read as much data as we can from the socket
        while (!$quit_daemon && $poll_handle)
        {
          $line = $rh->getline;

          # if nothing at the socket 
          if ((!defined $line) || ($line eq ""))
          {
            # stop polling
            $poll_handle = 0;

            # if we haven't read anything, the socket is shutting down
            if (!$read_something)
            {
              $read_set->remove($rh);
              $rh->close();
            }
          }
          else
          {
            $read_something = 1;
            $line =~ s/\r\n$//;
            Dada::logMsg(3, $dl, "commsThread: <- ".$line);
            $in->enqueue($line);
          }
        }
      }
    }
  }
  Dada::logMsg(1, $dl, "commsThread: exiting");
}


###############################################################################
#
# Dequeues messages from the log queue and write them to the relevant logfiles
#
sub loggingThread($$) 
{
  my ($log_file, $in) = @_;

  Dada::logMsg(2, $dl, "loggingThread: starting");

  my $line = "";
  my $status_file = "";

  my @bits = ();
  my $src = "";
  my $time = "";
  my $type = "";
  my $class = "";
  my $program = "";
  my $message = "";

  while (!$quit_daemon)
  {
    if ($in->pending)
    {
      $message = $in->dequeue();

      Dada::logMsg(3, $dl, "loggingThread: dequeued message [".$message."]");
      # extract the message parameters
      @bits = split(/\|/, $message, 6);

      if ($#bits == 5) 
      {
#       src       source of message (hostname or PWC ID) 
#       time      timestamp of message
#       type      type of message (pwc, sys, src) 
#       class     class of message (INFO, WARN, ERROR) 
#       program   script or binary that generated message (e.g. obs mngr)
#       message   message itself

        $src     = $bits[0];
        $time    = $bits[1];
        $type    = $bits[2];
        $class   = $bits[3];
        $program = $bits[4];
        $message = $bits[5];

        if ($class eq "INFO") {
          $line = "[".$time."] ".$program.": ".$message;
        } else {
          $line = "[".$time."] ".$program.": ".$class.": ".$message;
        }

        print STDOUT $line."\n";
      }
    }
    else
    {
      Dada::logMsg(3, $dl, "loggingThread: no messages pending");
      sleep(1);
    }
  }
  
  Dada::logMsg(2, $dl, "loggingThread: exiting");
  return 0;
}



#
# Handle a SIGINT or SIGTERM
#
sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
}

# 
# Handle a SIGPIPE
#
sub sigPipeHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
} 


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

  # this script can *only* be run on the caspsr-raid0 server
  my $host = Dada::getHostMachineName();
  if ($host ne REQUIRED_HOST) {
    return ("fail", "Error: this script can only be run on ".REQUIRED_HOST);
  }

  my $curr_user = `whoami`;
  chomp $curr_user;
  if ($curr_user ne REQUIRED_USER) {
    return ("fail", "Error: this script can only be run as ".REQUIRED_USER);
  }

  # Ensure more than one copy of this daemon is not running
  ($result, $response) = Dada::checkScriptIsUnique(basename($0));
  if ($result ne "ok") {
    return ($result, $response);
  }

  # Ensure that no other pipeline programs are running the server
  my @backends = ("apsr", "bpsr", "caspsr");
  my $running_scripts = "";
  my $i = 0;
  my $j = 0;
  my $script = "";
  for ($i=0; $i<=$#backends; $i++)
  {
    $cmd = "find /lfs/data0/".$backends[$i]."/control -name '*.pid' -printf '\%f\n'";
    ($result, $response) = Dada::mySystem($cmd);
    if ($result ne "ok")
    {
      return ("fail", "Could not check /lfs/data0/".$backends[$i]."/control for PID files");
    }
    if ($response ne "") 
    {
      my @scripts = split(/\n/, $response);
      for ($j=0; $j<=$#scripts; $j++) 
      {
        $script = $scripts[$j];
        $script =~ s/\.pid$//;
        $script =~ s/\./ /g;
        $running_scripts .= $script." ";
      }
    }
  }
  if ($running_scripts ne "")
  {
    return ("fail", "found scripts running: ".$running_scripts);
  }

  my $n_listen = $cfg{"NUM_PWC"} * 3;
  $log_sock = new IO::Socket::INET (
    LocalHost => $log_host,
    LocalPort => $log_port,
    Proto => 'tcp',
    Listen => $n_listen,
    ReuseAddr => 1
  ); 
  if (!$log_sock) { 
    return ("fail", "Could not create listening socket: ".$log_host.":".$log_port);
  } 

  return ("ok", "");
}
