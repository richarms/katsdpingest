#!/usr/bin/env perl

###############################################################################
#
# server_bpsr_multibob_manager.pl
#

use lib $ENV{"DADA_ROOT"}."/bin";

use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Bpsr;           # BPSR Module 
use strict;         # strict mode (like -Wall)
use warnings;
use threads;
use threads::shared;

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0));

#
# Constants
#
use constant DL           => 1;
use constant PIDFILE      => "bpsr_multibob_manager.pid";
use constant LOGFILE      => "bpsr_multibob_manager.log";
use constant QUITFILE     => "bpsr_multibob_manager.quit";
use constant MULTIBOB_BIN => "multibob_server";


#
# Global Variables
#
our %cfg   = Bpsr::getConfig();
our %roach = Bpsr::getROACHConfig();
our $error = $cfg{"STATUS_DIR"}."/bpsr_multibob_manager.error";
our $warn  = $cfg{"STATUS_DIR"}."/bpsr_multibob_manager.warn";
our $quit_daemon : shared  = 0;

# Autoflush output
$| = 1;


# Signal Handler
$SIG{INT} = \&sigHandle;
$SIG{TERM} = \&sigHandle;


#
# Local Varaibles
#
my $log_file = $cfg{"SERVER_LOG_DIR"}."/".LOGFILE;
my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".PIDFILE;
my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".QUITFILE;

my $control_thread = 0;
my $multibob_thread = 0;
my $multibob_plot_thread = 0;


#
# Main
#

# clear the error and warning files if they exist
if ( -f $warn ) {
  unlink ($warn);
}
if ( -f $error) {
  unlink ($error);
}

Dada::daemonize($log_file, $pid_file);

Dada::logMsg(0, DL, "STARTING SCRIPT");

# Start the daemon control thread
$control_thread = threads->new(\&controlThread, $quit_file, $pid_file);

# Start the multibob_server thread
$multibob_thread = threads->new(\&multibobThread);

# Start the multibob plotting thread
$multibob_plot_thread = threads->new(\&multibobPlotThread);

# Wait for threads to return
while (!$quit_daemon) {
  sleep(2);
}

# rejoin threads
Dada::logMsg(2, DL, "main: joining threads");
$control_thread->join();
$multibob_thread->join();
$multibob_plot_thread->join();

Dada::logMsg(0, DL, "STOPPING SCRIPT");

exit 0;



###############################################################################
#
# Functions
#


#
# Runs the multibob_server on localhost. If the server fails, then 
# try to relaunch it
#
sub multibobThread() {

  Dada::logMsg(1, DL, "multibobThread: starting");

  my $runtime_dir   = $cfg{"SERVER_STATS_DIR"};
  my $port          = $cfg{"IBOB_MANAGER_PORT"};
  my $npwc          = $cfg{"NUM_PWC"};

  my $config_thread = 0;
  my $cmd = "";
  my $result = "";
  my $response = "";
  my $binary = "";

  if ($cfg{"USE_SIM"} eq "1")
  {
    $binary = $cfg{"SIM_BINARY"};
    my $dests = "";
    my $i = 0;
    for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
    {
      $dests .= " ".$cfg{"PWC_".$i}.":".$cfg{"PWC_UDP_PORT_".$i};  
    }
    $cmd =  "cd ".$runtime_dir."; echo '' | ".$binary." -j -a 25 ".$dests." -t 0 2>&1";
    $cmd .= " | server_bpsr_server_logger.pl";
  }
  else
  {
    $binary = MULTIBOB_BIN;
    $cmd =  "cd ".$runtime_dir."; echo '' | ".$binary." -n ".$npwc." -p ".$port." 2>&1";
    $cmd .= " | server_bpsr_server_logger.pl";
  }

  Dada::logMsg(2, DL, "multibobThread: stopMultibobServer(".$binary.")");
  ($result, $response) = stopMultibobServer($binary);
  Dada::logMsg(2, DL, "multibobThread: ".$result." ".$response);

  while (!$quit_daemon) {

    if ($cfg{"USE_SIM"} eq "1") 
    {
      Dada::logMsg(1, DL, "multibobThread: skipping configure");
    }
    else
    {
      Dada::logMsg(1, DL, "multibobThread: launching config thread");
      $config_thread = threads->new(\&configureMultibobServerWrapper);
    }

    Dada::logMsg(1, DL, "multibobThread: running ".$binary);

    # This command should "hang" until the multibob_server command has terminated
    Dada::logMsg(1, DL, "multibobThread: ".$cmd);
    system($cmd);

    if (!$quit_daemon) {
      Dada::logMsgWarn($warn, "multibobThread: ".$binary." exited unexpectedly, re-launching");
      sleep(1);
    }
   
    if ($config_thread)
    { 
      Dada::logMsg(2, DL, "multibobThread: joining config_thread");
      $config_thread->join();
      Dada::logMsg(2, DL, "multibobThread: config_thread joined");
      $config_thread = 0;
    }
  }

  Dada::logMsg(1, DL, "multibobThread: exiting");

}

#
# Configure the mulibob_server
#
sub configureMultibobServerWrapper() 
{

  Dada::logMsg(2, DL, "configureMultibobServer: configuring multibob_server");
  my ($result, $response) = Bpsr::configureMultibobServer();

  if ($result ne "ok") {
    Dada::logMsgWarn($error, "configureMultibobServer: failed ".$response);
  } else {
    Dada::logMsg(1, DL, MULTIBOB_BIN." configured");
    Dada::logMsg(2, DL, "configureMultibobServer: done");
  }
}

# 
# Monitors the /nfs/results/bpsr/stats directory creating the PD Bandpass plots
# as requried
#
sub multibobPlotThread()
{

  Dada::logMsg(1, DL, "multibobPlotThread: starting");

  my $bindir    = Dada::getCurrentBinaryVersion();
  my $stats_dir = $cfg{"SERVER_STATS_DIR"};
  my @bramfiles = ();
  my $bramfile  = "";
  my $plot_cmd  = "";
  my $cmd       = "";
  my $result    = "";
  my $response  = "";
  my $i = 0;

  while (!$quit_daemon) {
   
    Dada::logMsg(2, DL, "multibobPlotThread: looking for bramdump files in ".$stats_dir);
 
    # look for plot files
    opendir(DIR,$stats_dir);
    @bramfiles = sort grep { !/^\./ && /\.bramdump$/ } readdir(DIR);
    closedir DIR;

    if ($#bramfiles == -1) {
      Dada::logMsg(2, DL, "multibobPlotThread: no files, sleeping");
    }

    # plot any existing bramplot files
    for ($i=0; $i<=$#bramfiles; $i++) 
    {
      $bramfile = $stats_dir."/".$bramfiles[$i];

      $plot_cmd = $bindir."/bpsr_bramplot ".$bramfile;

      Dada::logMsg(2, DL, $plot_cmd);

      ($result, $response) = Dada::mySystem($plot_cmd);

      if ($result ne "ok") {
        Dada::logMsgWarn($warn, "plot of ".$bramfile." failed ".$response);
      } else {
        Dada::logMsg(3, DL, "bpsr_bramplot ".$bramfile.": ".$response);
      }
      unlink($bramfile);
    }

    sleep(1);

    my $beam = "";
    for ($i=0; $i < $cfg{"NUM_PWC"}; $i++ ) {
      $beam = $roach{"BEAM_".$i};
      removeOldPngs($stats_dir, $beam, "1024x768");
      removeOldPngs($stats_dir, $beam, "400x300");
      removeOldPngs($stats_dir, $beam, "112x84");
    }

    sleep(1);

  }

  Dada::logMsg(1, DL, "multibobPlotThread: exiting");
}


#
# Polls for the "quitdaemons" file in the control dir
#
sub controlThread($$) {

  Dada::logMsg(1, DL, "controlThread: starting");

  my ($quit_file, $pid_file) = @_;
  Dada::logMsg(2, DL ,"controlThread(".$quit_file.", ".$pid_file.")");

  my $result = "";
  my $response = "";

  # poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # signal threads to exit
  $quit_daemon = 1;

  # Get the multibob server to quit
  my $multibob_binary = MULTIBOB_BIN;
  if ($cfg{"USE_SIM"}) 
  {
    $multibob_binary = $cfg{"SIM_BINARY"};
  }
    
  Dada::logMsg(1, DL, "controlThread: stopMultibobServer(".$multibob_binary.")");
  ($result, $response) = stopMultibobServer($multibob_binary);
  Dada::logMsg(1, DL, "controlThread: stopMultibobServer: ".$result." ".$response);

  if (-f $pid_file) {
    Dada::logMsg(2, DL, "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    Dada::logMsgWarn($warn, "controlThread: PID file did not exist on script exit");
  }

  Dada::logMsg(1, DL, "controlThread: exiting");

}

#
# Handle INT AND TERM signals
#
sub sigHandle($) {

  my $sigName = shift;

  Dada::logMsgWarn($warn, basename($0).": Received SIG".$sigName);

  $quit_daemon = 1;
  sleep(5);
 
  Dada::logMsgWarn($warn, basename($0).": Exiting");

  exit(1);

}

sub sigIgnore($) {
                                                                                                        
  my $sigName = shift;
  Dada::logMsgWarn($warn, basename($0)." Received SIG".$sigName);
  Dada::logMsgWarn($warn, basename($0)." Ignoring");

}


sub removeOldPngs($$$) {

  my ($stats_dir, $beam, $res) = @_;

  Dada::logMsg(3, DL, "removeOldPngs(".$stats_dir.", ".$beam.", ".$res.")");

  # remove any existing plot files that are more than 10 seconds old
  my $cmd  = "find ".$stats_dir." -name '*".$beam."_".$res.".png' -printf \"%T@ %f\\n\" | sort -n -r";
  my $result = `$cmd`;
  my @array = split(/\n/,$result);

  my $time = 0;
  my $file = "";
  my $line = "";
  my $i = 0;

  # if there is more than one result in this category and its > 10 seconds old, delete it
  for ($i=1; $i<=$#array; $i++) {

    $line = $array[$i];
    ($time, $file) = split(/ /,$line,2);

    if (($time+20) < time)
    {
      $file = $stats_dir."/".$file;
      Dada::logMsg(3, DL, "removeOldPngs: unlink ".$file);
      unlink($file);
    }
  }

  Dada::logMsg(3, DL, "removeOldPngs: exiting");
}


#
# Ensures that the multibob server is not running
#
sub stopMultibobServer($) {

  (my $binary) = @_;

  Dada::logMsg(1, DL, "stopMultibobServer()");

  my $localhost = Dada::getHostMachineName();
  my $port      = $cfg{"IBOB_MANAGER_PORT"};

  my $result = "";
  my $response = "";
  my $cmd = "";
  my $handle = 0;

  # Check if the binary is running
  $cmd = "pgrep -lf ^".$binary;
  Dada::logMsg(2, DL, "stopMultibobServer: ".$cmd);
  ($result, $response) = Dada::mySystem($cmd);
  Dada::logMsg(2, DL, "stopMultibobServer: ".$result." ".$response);

  if ($result eq "ok") 
  {
    Dada::logMsg(1, DL, "stopMultibobServer: ".$binary." process existed");

    # set the global variable to quit the daemon
    $handle = Dada::connectToMachine($localhost, $port, 1);
    if (!$handle) {
      Dada::logMsgWarn($warn, "stopMultibobServer: could not connect to ".$binary." on ".$localhost.":".$port);
    } else {

      Dada::logMsg(0, DL, "stopMultibobServer: ignoring welome message..");
      # ignore welcome message
      $response = <$handle>;

      Dada::logMsg(0, DL, "stopMultibobServer: multibob <- close");
      ($result, $response) = Dada::sendTelnetCommand($handle, "close");
      Dada::logMsg(0, DL, "stopMultibobServer: multibob -> ".$result.":".$response);

      sleep(1);

      Dada::logMsg(0, DL, "stopMultibobServer: multibob_server <- quit");
      print $handle "quit\r\n";
      close($handle);
    }

    sleep(1);

    # try again to ensure it exited
    Dada::logMsg(2, DL, "stopMultibobServer: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(2, DL, "stopMultibobServer: ".$result." ".$response);

    # if is it still running
    if ($result eq "ok") {

      Dada::logMsgWarn($warn, "stopMultibobServer: multibob_server, refused to exit, killing...");
      Dada::logMsg(1, DL, "stopMultibobServer: killProcess(^".$binary.")");
      ($result, $response) = Dada::killProcess("^".$binary);
      Dada::logMsg(1, DL, "stopMultibobServer: killProcess ".$result." ".$response);
      return ("ok", $response);

    } else {
      return ("ok", "process exited correctly"); 
    }

  } else {
    return ("ok", "process not running");
  }

}
