#!/usr/bin/env perl

# 
# Simple MOPSR processing script
#
#   Runs digifil on the individiaul data blocks
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
use threads;
use threads::shared;
use IPC::Open3;
use IO::Select;
use LockFile::Simple qw(trylock lock unlock);
use Time::HiRes qw(usleep);
use Symbol 'gensym'; 

#
# Function Prototypes
#
sub msg($$$);
sub multiFork($\@);
sub multiForkLog($\@);
sub prepareObservation($$$);
sub createLocalDir($\%);


sub usage() 
{
  print "Usage: ".basename($0)." PROC_ID\n";
  print "   PROC_ID   The Primary Write Client ID this script will process\n";
}

#
# Global Variable Declarations
#
our $dl : shared;
our $quit_daemon : shared;
our $daemon_name : shared;
our $proc_id : shared;
our $db_key : shared;
our $beam : shared;
our %cfg : shared;
our %ct : shared;
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
$proc_id = 0;
$db_key = ();
$beam = ();
%cfg = Mopsr::getConfig("bp");
%ct = Mopsr::getCornerturnConfig("bp");
$log_host = $cfg{"SERVER_HOST"};
$sys_log_port = $cfg{"SERVER_BP_SYS_LOG_PORT"};
$src_log_port = $cfg{"SERVER_BP_SRC_LOG_PORT"};
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
my $quit = 0;

#
# Check command line arguments is 1
#
if ($#ARGV != 0) 
{
  usage();
  exit(1);
}
$proc_id  = $ARGV[0];

# ensure that our bp_id is valid 
if (($proc_id >= 0) &&  ($proc_id < $cfg{"NUM_BP"}))
{
  # and matches configured hostname
  if (($cfg{"BP_".$proc_id} eq Dada::getHostMachineName()) || ($cfg{"BP_".$proc_id} eq "localhost"))
  {
  }
  else
  {
    print STDERR "PROC_ID did not match configured hostname\n";
    usage();
    exit(1);
  }
}
else
{
  print STDERR "PROC_ID was not a valid integer between 0 and ".($cfg{"NUM_BP"}-1)."\n";
  usage();
  exit(1);
}

# Also check that we are an ACTIVE or PASSIVE PWC
if (($cfg{"BP_STATE_".$proc_id} ne "active") && ($cfg{"BP_STATE_".$proc_id} ne "passive"))
{
  print STDOUT "Config file specified BP_STATE_".$proc_id."=".$cfg{"BP_STATE_".$proc_id}.", not starting\n";
  exit(0);
}

#
# Sanity check to prevent multiple copies of this daemon running
#
Dada::preventDuplicateDaemon(basename($0)." ".$proc_id);


#
# Main
#
{
  my ($cmd, $result, $response, $proc_cmd, $curr_raw_header, $prev_raw_header);
  my ($i, $index, $line, $rval);

  $sys_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".log";
  $src_log_file = $cfg{"CLIENT_LOG_DIR"}."/".$daemon_name."_".$proc_id.".src.log";
  $pid_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".pid";

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

  msg(1,"INFO", "STARTING SCRIPT");

  sleep(5);

  # this thread will monitor for our daemon quit file
  $control_thread = threads->new(\&controlThread, $pid_file);

	$curr_raw_header = "";
	$prev_raw_header = "";
	my %header;

  my ($cmd, $raw_header);
  my ($obs_dir, $ibeam, $key);
  my $err = gensym;

  $db_key = Dada::getDBKey($cfg{"DATA_BLOCK_PREFIX"}, $proc_id, $cfg{"NUM_BP"}, $cfg{"BP_DATA_BLOCK"});
  $beam = 1;

  # Main Loop
  while (!$quit_daemon) 
  {
		%header = ();

    # run dada_header on all the input data blocks
    $cmd = "dada_header -k ".$db_key;

    msg(2, "INFO", "main: ".$cmd);
    $raw_header = `$cmd 2>&1`;
    msg(2, "INFO", "main: ".$cmd." returned");
    if ($? != 0)
    {
      if ($quit_daemon)
      {
        msg(2, "INFO", "dada_header failed, but quit_daemon true");
      }
      else
      {
        msg(0, "ERROR", "dada_header failed: ".$raw_header);
        $quit_daemon = 1;
      }
    }
    else
    {
      my %header = Dada::headerToHash($raw_header);
      msg (0, "INFO", "UTC_START=".$header{"UTC_START"}." NCHAN=".$header{"NCHAN"}." NANT=".$header{"NANT"});

      # ensure that our primary configuration directory (NFS) is mounted
      $cmd = "ls -1d ".$cfg{"CONFIG_DIR"};
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$cmd." config_dir=".$response);
      if ($response ne $cfg{"CONFIG_DIR"})
      {
        msg(0, "ERROR", "NFS automount for ".$cfg{"CONFIG_DIR"}." failed: ".$response);
        $quit_daemon = 1;
      }

      my %header = Dada::headerToHash($responses[0]);
      $cmd = "mv ".$cfg{"CLIENT_RECORDING_DIR"}."/".$header{"UTC_START"}."/obs.processing ".$cfg{"CLIENT_RECORDING_DIR"}."/".$header{"UTC_START"}."/obs.finished";
      msg(2, "INFO", "main: ".$cmd);
      ($result, $response) = Dada::mySystem($cmd);
      msg(3, "INFO", "main: ".$cmd." ".$response);
      if ($result ne "ok")
      {
        msg(0, "ERROR", "main: failed to move obs.processing to obs.finished for ".$header{"UTC_START"});
      }
    }
  }

  msg(2, "INFO", "main: joining controlThread");
  $control_thread->join();

  msg(0, "INFO", "STOPPING SCRIPT");
  Dada::nexusLogClose($sys_log_sock);
  Dada::nexusLogClose($src_log_sock);

  exit(0);
}


sub controlThread($) 
{
  my ($pid_file) = @_;

  msg(2, "INFO", "controlThread : starting");

  my $host_quit_file = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name.".quit";
  my $bp_quit_file  = $cfg{"CLIENT_CONTROL_DIR"}."/".$daemon_name."_".$proc_id.".quit";

  my ($cmd, $result, $response);

  while ((!$quit_daemon) && (!(-f $host_quit_file)) && (!(-f $bp_quit_file))) 
  {
    sleep(1);
  }

  $quit_daemon = 1;

  my $user = "mpsr";
  my @processes_to_kill = ();
  my ($i, $process);
  for ($i=0; $i<=$#db_keys; $i++)
  {
    push @processes_to_kill, "^dada_header -k ".$db_keys[$i];
  }

  foreach $process ( @processes_to_kill)
  {
    msg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    msg(3, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(0, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  @processes_to_kill = ();
  for ($i=0; $i<=$#db_keys; $i++)
  {
    push @processes_to_kill, "^digifil /tmp/mopsr_".$db_keys[$i].".info";
  }

  foreach $process ( @processes_to_kill)
  {
    msg(2, "INFO", "controlThread: killProcess(".$process.", ".$user.")");
    ($result, $response) = Dada::killProcess($process, $user);
    msg(3, "INFO", "controlThread: killProcess ".$result." ".$response);
    if ($result ne "ok")
    {
      msg(1, "WARN", "controlThread: killProcess for ".$process." failed: ".$response);
    }
  }

  msg(2, "INFO", "controlThread: checking for PID file");
  if ( -f $pid_file) 
  {
    msg(2, "INFO", "controlThread: unlinking PID file");
    unlink($pid_file);
  } else {
    msg(0, "WARN", "controlThread: PID file did not exist on script exit");
  }

  msg(2, "INFO", "controlThread: exiting");
}

sub prepareObservation($$$)
{
  my ($key, $beam, $header) = @_;

  my %h = Dada::headerToHash($header);

  # default command to return is jettision
  my $proc_cmd = "dada_dbnull -k ".$key." -s -z";

  # output data directory
  my $beam_dir = $cfg{"CLIENT_RECORDING_DIR"}."/".$h{"UTC_START"}."/".sprintf("BEAM_%03d",$beam);

  # create the local directories required
  if (createLocalDir($beam_dir, %h) < 0)
  {
    return ("fail", $beam_dir, $proc_cmd);
  }

  msg(2, "INFO", "prepareObservation: local dirs created");

  my $obs_header = $beam_dir."/obs.header";

  my $tmp_info_file =  "/tmp/mopsr_".$key.".info";
  # ensure a file exists with the write processing key
  if (! -f $tmp_info_file)
  {
    open FH, ">".$tmp_info_file;
    print FH "DADA INFO:\n";
    print FH "key ".$key."\n";
    close FH;
  }

  $proc_cmd = "digifil ".$tmp_info_file." -D 478.8 -c -b 8 -B 0.1 -o ".$h{"UTC_START"}.".fil";

  # processing must ocurring in the specified dir
  $proc_cmd = "cd ".$beam_dir."; ".$proc_cmd;

  msg(2, "INFO", "Final PROC_CMD: ".$proc_cmd);

  return ("ok", $proc_cmd);
}

#
# Create the local directories required for this observation
#
sub createLocalDir($\%)
{
  my ($beam_dir, $h_ref) = @_;

  msg(2, "INFO", "createLocalDir()");

  my %h = %$h_ref;

  my ($result, $response);
  msg(2, "INFO", "createLocalDir: mkdirRecursive(".$beam_dir.", 0755)");
  ($result, $response) = Dada::mkdirRecursive($beam_dir, 0755);
  msg(3, "INFO", "createLocalDir: ".$result." ".$response);
  if ($result ne "ok")
  {
    msg(0, "ERROR", "createLocalDir: failed to create dir [".$beam_dir."]: ".$response);
  }

  # create an obs.header file in the processing dir:
  msg(2, "INFO", "createLocalDir: creating obs.header");
  my $file = $beam_dir."/obs.header";
  open(FH,">".$file.".tmp");
  my $k = "";
  foreach $k ( keys %h)
  {
    print FH Dada::headerFormat($k, $h{$k})."\n";
  }
  close FH;
  rename($file.".tmp", $file);

  return 0;
}



#
# Logs a message to the nexus logger and print to STDOUT with timestamp
#
sub msg($$$) 
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
      Dada::nexusLogMessage($sys_log_sock, sprintf("%02d",$proc_id), $time, "sys", $type, "bp_digifil", $msg);
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

sub multiFork($\@)
{
  my ($ncmds, $cmds_ref) = @_;

  my @cmds = @$cmds_ref;

  my @pids = ();
  my @ins = ();
  my @outs = ();
  my @errs = ();

  my ($i, $cmd, $cmd_in, $cmd_out, $cmd_err, $index, $line, $rval, $pid);

  my @out_bufs = ();
  my @err_bufs = ();
  my @results = ();
  my @responses = ();

  my $err = gensym;
  my $sel = IO::Select->new();

  # first we need to run dada_header on the input data blocks 
  for ($i=0; $i<$ncmds; $i++)
  {
    $cmd = $cmds[$i];
    push (@results, "fail");
    push (@responses, "");
  
    my ($cmd_in, $cmd_out, $cmd_err);

    msg(2, "INFO", "START ".$cmd);
    #eval {
      $pid = open3 ($cmd_in, $cmd_out, $cmd_err, $cmd);
    #};
    #die "open3: $@\n" if $@;

    push (@pids, $pid);
    push (@ins, $cmd_in);

    $sel->add ($cmd_out);
    push (@outs, $cmd_out);
    push (@out_bufs, "");
   
    if ($cmd_err)
    { 
      $sel->add ($cmd_err);
      push (@outs, $cmd_err);
      push (@out_bufs, "");
    }

    # close STDIN
    close ($cmd_in);
  }

  # now that all are launched, we need to read back the data
  # from STDOUT and STDERR
  while (my @ready = $sel->can_read())
  {
    foreach my $handle (@ready)
    {
      $index = -1;
      for ($i=0; $i<$ncmds; $i++)
      {
        if ($handle == $outs[$i])
        {
          $index = $i;
        }
      }
      if ($index eq -1)
      {
        msg(0, "ERROR", "could not match handle to I/O FD");
        sleep (1);
        $index = 0;
      }

      # read from the handle in 1024 byte chunks
      my $bytes_read = sysread ($handle, my $buf='', 1024);
      if ($bytes_read == -1)
      {
        #warn("Error reading from child's STDOUT: $!\n");
        $sel->remove($handle);
        next;
      }
      if ($bytes_read == 0)
      {
        # print("Child's STDOUT closed\n");
        $sel->remove($handle);
        next;
      }
      else
      {
        my @lines = split(/\n/, $buf);
        my $nlines = $#lines + 1;
        if (!($buf =~ m/\n$/))
        {
          $nlines -= 1;
        }
        for ($i=0; $i<$nlines; $i++)
        {
          $line = $lines[$i];
          # print "index=$index lines[$i]=$line\n";

          if ($out_bufs[$index] ne "")
          {
            $line = $out_bufs[$index].$line;
            $out_bufs[$index] = "";
          }

          $responses[$index] .= $line."\n";
        }

        if (!($buf =~ m/\n$/))
        {
          $out_bufs[$index] = $lines[$nlines];
        }
      }
    }
  }

  for ($i=0; $i<$ncmds; $i++)
  {
    waitpid ($pids[$i], 0);
    $rval = $? >> 8;
    $results[$i] = ($rval == 0) ? "ok" : "fail";
    msg(2, "INFO", "END   ".$cmds[$i]);
    chomp ($responses[$i]);
  }

  return (\@results, \@responses);
}


sub multiForkLog($\@)
{
  my ($ncmds, $cmds_ref) = @_;

  my @cmds = @$cmds_ref;

  my @pids = ();
  my @ins = ();
  my @outs = ();
  my @errs = ();

  my ($i, $cmd, $cmd_in, $cmd_out, $cmd_err, $index, $line, $rval, $pid);

  my @out_bufs = ();
  my @err_bufs = ();
  my @results = ();

  my $err = gensym;
  my $sel = IO::Select->new();

  # first we need to run dada_header on the input data blocks 
  for ($i=0; $i<$ncmds; $i++)
  {
    $cmd = $cmds[$i];
    push (@results, "fail");
  
    my ($cmd_in, $cmd_out, $cmd_err);

    msg(2, "INFO", "START ".$cmd);
    #eval {
      $pid = open3 ($cmd_in, $cmd_out, $cmd_err, $cmd);
    #};
    #die "open3: $@\n" if $@;

    push (@pids, $pid);
    push (@ins, $cmd_in);

    $sel->add ($cmd_out);
    push (@outs, $cmd_out);
    push (@out_bufs, "");
   
    if ($cmd_err)
    { 
      $sel->add ($cmd_err);
      push (@outs, $cmd_err);
      push (@out_bufs, "");
    }

    # close STDIN
    close ($cmd_in);
  }

  # now that all are launched, we need to read back the data
  # from STDOUT and STDERR
  while (my @ready = $sel->can_read())
  {
    foreach my $handle (@ready)
    {
      $index = -1;
      for ($i=0; $i<$ncmds; $i++)
      {
        if ($handle == $outs[$i])
        {
          $index = $i;
        }
      }
      if ($index eq -1)
      {
        msg(0, "ERROR", "could not match handle to I/O FD");
        sleep (1);
        $index = 0;
      }

      # read from the handle in 1024 byte chunks
      my $bytes_read = sysread ($handle, my $buf='', 1024);
      if ($bytes_read == -1)
      {
        warn("Error reading from child's STDOUT: $!\n");
        $sel->remove($handle);
        next;
      }
      if ($bytes_read == 0)
      {
        #print("Child's STDOUT closed\n");
        $sel->remove($handle);
        next;
      }
      else
      {
        my @lines = split(/\n/, $buf);
        my $nlines = $#lines + 1;
        if (!($buf =~ m/\n$/))
        {
          $nlines -= 1;
        }
        for ($i=0; $i<$nlines; $i++)
        {
          $line = $lines[$i];
          #print "index=$index lines[$i]=$line\n";

          if ($out_bufs[$index] ne "")
          {
            $line = $out_bufs[$index].$line;
            $out_bufs[$index] = "";
          }

          Dada::nexusPipeLog ($line, $src_log_sock, $src_log_file, "src", $proc_id, $daemon_name, "bp_proc");
        }

        if (!($buf =~ m/\n$/))
        {
          $out_bufs[$index] = $lines[$nlines];
        }
      }
    }
  }

  for ($i=0; $i<$ncmds; $i++)
  {
    waitpid ($pids[$i], 0);
    $rval = $? >> 8;
    msg(2, "INFO", "END   ".$cmds[$i]);
    $results[$i] = ($rval == 0) ? "ok" : "fail";
  }

  return (\@results);
}
