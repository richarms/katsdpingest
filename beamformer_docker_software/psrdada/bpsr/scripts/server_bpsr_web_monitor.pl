#!/usr/bin/env perl

###############################################################################
#
# server_bpsr_web_monitor.pl
#
# This script maintains vital statistics for all the client/server machines
# and stores this information in memory. Web browsers will connect to a port
# on this script for this information
#

use lib $ENV{"DADA_ROOT"}."/bin";


use IO::Socket;     # Standard perl socket library
use IO::Select;     # Allows select polling on a socket
use Net::hostent;
use File::Basename;
use Math::Trig;
use Bpsr;           # Bpsr Module for configuration options
use strict;         # strict mode (like -Wall)
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

#
# Global Variables
#
our %cfg                        = Bpsr::getConfig();
our %roaches                    = Bpsr::getROACHConfig();
our $error                      = $cfg{"STATUS_DIR"}."/bpsr_web_monitor.error";
our $warn                       = $cfg{"STATUS_DIR"}."/bpsr_web_monitor.warn";
our $quit_daemon : shared       = 0;
our $daemon_name : shared       = Dada::daemonBaseName($0);
our $node_info : shared         = "";
our $curr_obs : shared          = "";
our $status_info : shared       = "";
our $tape_info_string : shared  = "";
our $beam_info_string : shared  = "";
our $image_string : shared      = "";

#
# Main
#
{

  # clear the error and warning files if they exist
  if ( -f $warn ) {
    unlink ($warn);
  }
  if ( -f $error) {
    unlink ($error);
  }

  # Autoflush output
  $| = 1;

  # Signal Handler
  $SIG{INT} = \&sigHandle;
  $SIG{TERM} = \&sigHandle;

  my $log_file = $cfg{"SERVER_LOG_DIR"}."/".$daemon_name.".log";
  my $pid_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".pid";

  my $control_thread = 0;
  my $sock = 0;
  my $rh = "";
  my $string = "";
  my $handle = 0;

  # Sanity check for this script
  if (index($cfg{"SERVER_ALIASES"}, $ENV{'HOSTNAME'}) < 0 ) {
    print STDERR "ERROR: Cannot run this script on ".$ENV{'HOSTNAME'}."\n";
    print STDERR "       Must be run on the configured server: ".$cfg{"SERVER_HOST"}."\n";
    exit(1);
  }

  $sock = new IO::Socket::INET (
    LocalHost => $cfg{"SERVER_HOST"}, 
    LocalPort => $cfg{"SERVER_WEB_MONITOR_PORT"},
    Proto => 'tcp',
    Listen => 1,
    Reuse => 1,
  );
  die "Could not create listening socket: $!\n" unless $sock;

  # Redirect standard output and error
  Dada::daemonize($log_file, $pid_file);

  Dada::logMsg(0, DL, "STARTING SCRIPT");

  # start the daemon control thread
  $control_thread = threads->new(\&controlThread, $pid_file);

  my $read_set = new IO::Select();  # create handle set for reading
  $read_set->add($sock);   # add the main socket to the set

  Dada::logMsg(2, DL, "Waiting for connection on ".$cfg{"SERVER_HOST"}.":".$cfg{"SERVER_WEB_MONITOR_PORT"});

  # Disk space on nodes [R]
  # Data block info on nodes [R]
  # Load info on nodes [R]
  my $node_info_thread = 0;

  # Current images for web display [L]
  my $image_info_thread = 0;

  # Current source information [L]
  my $curr_info_thread = 0;

  # Current status file information [L]
  my $status_info_thread = 0;

  # Current tape/xfer information [L]
  my $tape_info_thread = 0;

  # Current beam information [L] from TCS host
  my $beam_info_thread = 0;

  # Launch monitoring threads
  $curr_info_thread = threads->new(\&currentInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $tape_info_thread = threads->new(\&tapeInfoThread);
  $status_info_thread = threads->new(\&statusInfoThread);
  $node_info_thread = threads->new(\&nodeInfoThread);
  $image_info_thread = threads->new(\&imageInfoThread, $cfg{"SERVER_RESULTS_DIR"});
  $beam_info_thread = threads->new(\&beamInfoThread);

  # Sleep for a few seconds to allow the threads to start and collect
  # their first iteration of data
  sleep(3);

  while (!$quit_daemon)
  {
    # Get all the readable handles from the server
    my ($rh_set) = IO::Select->select($read_set, undef, undef, 2);

    foreach $rh (@$rh_set) 
    {
      # if it is the main socket then we have an incoming connection and
      # we should accept() it and then add the new socket to the $Read_Handles_Object
      if ($rh == $sock)
      {
        $handle = $rh->accept();
        $handle->autoflush();
        Dada::logMsg(2, DL, "Accepting connection");

        # Add this read handle to the set
        $read_set->add($handle); 
        $handle = 0;
      }
      else
      {
        $string = Dada::getLine($rh);

        if (! defined $string) 
        {
          Dada::logMsg(2, DL, "Closing a connection");
          $read_set->remove($rh);
          close($rh);
        } 
        else
        {
          Dada::logMsg(2, DL, "<- ".$string);
          my $r = "";

          if    ($string eq "node_info")     { $r = $node_info; }
          elsif ($string eq "img_info")      { $r = $image_string; }
          elsif ($string eq "curr_obs")      { $r = $curr_obs; }
          elsif ($string eq "status_info")   { $r = $status_info; }
          elsif ($string eq "tape_info")     { $r = $tape_info_string; }
          elsif ($string eq "beam_info")     { $r = $beam_info_string; }
          else    { Dada::logMsgWarn($warn, "unexpected command: ".$string); } 

          print $rh $r."\n";
          Dada::logMsg(2, DL, "-> ".$r);

          # experimental!
          $read_set->remove($rh);
          close($rh);
        }
      }
    }
  }

  # Rejoin our daemon control thread
  $control_thread->join();

  # Rejoin other threads
  $curr_info_thread->join();
  $tape_info_thread->join();
  $status_info_thread->join();
  $node_info_thread->join();
  $image_info_thread->join();
  $beam_info_thread->join();

  close($sock);
                                                                                  
  Dada::logMsg(0, DL, "STOPPING SCRIPT");
                                                                                  
  exit(0);
}

###############################################################################
#
# Functions
#

sub controlThread($) {

  (my $pid_file) = @_;

  Dada::logMsg(2, DL, "controlThread: starting");

  my $quit_file = $cfg{"SERVER_CONTROL_DIR"}."/".$daemon_name.".quit";

  # Poll for the existence of the control file
  while ((!-f $quit_file) && (!$quit_daemon)) {
    sleep(1);
  }

  # set the global variable to quit the daemon
  $quit_daemon = 1;

  Dada::logMsg(2, DL, "Unlinking PID file: ".$pid_file);
  unlink($pid_file);

  Dada::logMsg(1, DL, "controlThread: exiting");

}


#
# Handle INT AND TERM signals
#
sub sigHandle($) 
{
  my $sigName = shift;
  print STDERR $daemon_name." : Received SIG".$sigName."\n";
  $quit_daemon = 1;
  sleep(5);
  print STDERR $daemon_name." : Exiting\n";
  exit(1);
}


#
# maintains the curr_info string
#
sub currentInfoThread($) 
{
  (my $results_dir) = @_;

  Dada::logMsg(1, DL, "currentInfoThread: starting");

  my $sleep_time = 10;
  my $sleep_counter = 0;

  my $cmd = "find ".$results_dir." -ignore_readdir_race -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
  my $obs = "";
  my $tmp_str = "";
  my %cfg_file = ();

  while (!$quit_daemon) 
  {
    if ($sleep_counter > 0) 
    {
      $sleep_counter--;
      sleep(1);
    } 
    else
    {
      $sleep_counter = $sleep_time;

      Dada::logMsg(3, DL, "currentInfoThread: ".$cmd);
      $obs = `$cmd`;
      chomp $obs;
      Dada::logMsg(3, DL, "currentInfoThread: ".$obs);

      if (-f $results_dir."/".$obs."/obs.info")
      {
        $tmp_str = "";
        %cfg_file = Dada::readCFGFile($results_dir."/".$obs."/obs.info"); 

        $tmp_str .= "SOURCE:::".$cfg_file{"SOURCE"}.";;;";
        $tmp_str .= "RA:::".$cfg_file{"RA"}.";;;";
        $tmp_str .= "DEC:::".$cfg_file{"DEC"}.";;;";
        $tmp_str .= "CFREQ:::".$cfg_file{"CFREQ"}.";;;";
        $tmp_str .= "BANDWIDTH:::".$cfg_file{"BANDWIDTH"}.";;;";
        $tmp_str .= "ACC_LEN:::".$cfg_file{"ACC_LEN"}.";;;";
        $tmp_str .= "NUM_PWC:::".$cfg_file{"NUM_PWC"}.";;;";
        $tmp_str .= "PID:::".$cfg_file{"PID"}.";;;";
        $tmp_str .= "UTC_START:::".$cfg_file{"UTC_START"}.";;;";
        $tmp_str .= "PROC_FILE:::".$cfg_file{"PROC_FILE"}.";;;";
        $tmp_str .= "REF_BEAM:::".$cfg_file{"REF_BEAM"}.";;;";
        $tmp_str .= "NBEAM:::".$cfg_file{"NBEAM"}.";;;";
        $tmp_str .= "NPOL:::".$cfg_file{"NPOL"}.";;;";
        $tmp_str .= "INTERGRATED:::0;;;";
 
        Dada::logMsg(3, DL, $tmp_str); 

        # update the global variable 
        $curr_obs = $tmp_str;

        Dada::logMsg(2, DL, "currInfoThread: ".$curr_obs);
      }
    }
  }

  Dada::logMsg(1, DL, "currentInfoThread: exiting");
}


#
# Maintains a listing of the current images
#
sub imageInfoThread($)
{
  my ($results_dir) = @_;

  Dada::logMsg(1, DL, "imageInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;

  my $cmd = "";
  my @images = ();
  my @image_types = qw(bp ts fft dts pvf);
  my $obs = "";

  my $i = 0;
  #my @arr = ();
  my %bp_images = ();
  my %ts_images = ();
  my %fft_images = ();
  my %dts_images = ();
  my %pvf_images = ();
  my %pdbp_images = ();
  my %pdcp_images = ();
  my %pdhg_images = ();
  my @keys = ();
  my $k = "";
  my $dirs_string = ();
  my @dirs = ();
  my $no_image = "";
  my $xml = "";
  my $img_string = "";

  my $img_beam = "";
  my $img_file = "";
  my $img_time = "";
  my $img_info = "";
  my $img_type = "";
  my $img_plot = "";
  my $img_res = "";
  my $img_w = "";
  my $img_h = "";
  my %beams = ();
  my $cand_img = "";
  my $dm_vs_time_img = "";
  my $junk;

  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $beams{$i} = $roaches{"BEAM_".$i};
  }

  chdir $results_dir;

  while (!$quit_daemon)
  {
    if ($sleep_counter > 0)
    {
      $sleep_counter--;
      sleep(1);
    }
    else
    {
      $sleep_counter = $sleep_time;
      $xml = "";

      # find the most recent directory (ls gives arg list too long)
      $cmd = "find ".$results_dir." -ignore_readdir_race -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
      $obs = `$cmd`;
      chomp $obs;

      # get the listing of beam dirs
      @dirs = ();
      $cmd = "find ".$results_dir."/".$obs." -ignore_readdir_race -mindepth 1 -maxdepth 1 -type d -name '??' -printf '\%f '";
      $dirs_string = `$cmd`;
      chomp $dirs_string;

      # get the listing of small image files
      $cmd = "find ".$results_dir."/".$obs." -ignore_readdir_race -name '*_112x84.png' | sort | awk -F/ '{print \$(NF-1), \$(NF)}'";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      %bp_images = ();
      %ts_images = ();
      %fft_images = ();
      %dts_images = ();
      %pvf_images = ();
      %pdbp_images = ();
      %pdcp_images = ();
      %pdhg_images = ();

      for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
      {
        $b = $beams{$i};
        $no_image =  "../images/bpsr_beam_disabled_240x180.png";
        if ($dirs_string =~ m/$b/) 
        {
          $no_image = "../../images/blankimage.gif"; 
        }
        $bp_images{$b} = $no_image;
        $ts_images{$b} = $no_image;
        $fft_images{$b} = $no_image;
        $dts_images{$b} = $no_image;
        $pvf_images{$b} = $no_image;
        $pdbp_images{$b} = $no_image;
        $pdcp_images{$b} = $no_image;
        $pdhg_images{$b} = $no_image;
      }


      for ($i=0; $i<=$#images; $i++)
      {
        ($img_beam, $img_file) = split(/ /,$images[$i], 2);
        ($img_time, $img_info, $img_type) = split(/\./, $img_file);

        ($img_plot, $img_res) = split(/_/, $img_info);
        ($img_w, $img_h) = split(/x/, $img_res);

        if ($img_plot eq "bp")  { $bp_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "ts")  { $ts_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "fft") { $fft_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "dts") { $dts_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
        if ($img_plot eq "pvf") { $pvf_images{$img_beam} = $obs."/".$img_beam."/".$img_file; }
      }

      # now lookup the Pre Decimation Band Pass images
      $cmd = "find ".$results_dir."/stats -ignore_readdir_race -name '*_112x84.png' -printf '\%f\n' | sort";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      for ($i=0; $i<=$#images; $i++)
      {
        chomp $images[$i];
        ($img_time, $img_beam, $img_res) = split(/_/, $images[$i], 3);
        $pdbp_images{$img_beam} = $images[$i];
      }

      # lookup the Pre Decimation Cross Pol images
      $cmd = "find ".$results_dir."/stats -ignore_readdir_race -name '*_112x84_cross.png' -printf '\%f\n' | sort";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      for ($i=0; $i<=$#images; $i++)
      {
        chomp $images[$i];
        ($img_time, $img_beam, $img_res, $junk) = split(/_/, $images[$i], 4);
        $pdcp_images{$img_beam} = $images[$i];
      }

      # lookup the Histogram images
      $cmd = "find ".$results_dir."/stats -ignore_readdir_race -name '*_112x84_hist.png' -printf '\%f\n' | sort";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);

      for ($i=0; $i<=$#images; $i++)
      {
        chomp $images[$i];
        ($img_time, $img_beam, $img_res, $junk) = split(/_/, $images[$i], 4);
        $pdhg_images{$img_beam} = $images[$i];
      }

      for  ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
      {
        $b = $beams{$i};

        $xml .= "<beam name='".$b."'>";

        $xml .= "<img type='bp' width='112' height='84'>".$bp_images{$b}."</img>";
        $xml .= "<img type='ts' width='112' height='84'>".$ts_images{$b}."</img>";
        $xml .= "<img type='fft' width='112' height='84'>".$fft_images{$b}."</img>";
        $xml .= "<img type='dts' width='112' height='84'>".$dts_images{$b}."</img>";
        $xml .= "<img type='pvf' width='112' height='84'>".$pvf_images{$b}."</img>";
        #$xml .= "<img type='pdbp' width='112' height='84'>stats/".$pdbp_images{$b}."</img>";

        $xml .= "</beam>\n";
      }

      foreach $b (keys %pdbp_images) 
      {
        $xml .= "<beam name='".$b."'>";
        $xml .= "<img type='pdbp' width='112' height='84'>stats/".$pdbp_images{$b}."</img>";
        $xml .= "<img type='pdcp' width='112' height='84'>stats/".$pdcp_images{$b}."</img>";
        $xml .= "<img type='pdhg' width='112' height='84'>stats/".$pdhg_images{$b}."</img>";
        $xml .= "</beam>\n";
      }

      # now lookup the Candidate monitor files
      $cmd = "find ".$results_dir."/".$obs." -ignore_readdir_race -maxdepth 1 -name '2*cands_1024x768.png' -printf '%f\n' -o -name '2*dm_vs_time_*.png' -printf '%f\n'| sort";
      $img_string = `$cmd`;
      @images = split(/\n/, $img_string);
      $cand_img = "../../images/blankimage.gif";
      $dm_vs_time_img = "../../images/blankimage.gif";
      for ($i=0; $i<=$#images; $i++)
      {
        Dada::logMsg(2, DL, "imageInfoThread: images[$i]=".$images[$i]);  
        chomp $images[$i];
        if ($images[$i] =~ m/cands/)
        {
          $cand_img = $obs."/".$images[$i];
        }
        if ($images[$i] =~ m/dm_vs_time/)
        {
          $dm_vs_time_img = $obs."/".$images[$i];
        }
      }
      $xml .= "<transient_candidates>";
      if ($cand_img ne "")
      {
        $xml .= "<img type='cands' width='1024' height='768'>".$cand_img."</img>";
      }
      if ($dm_vs_time_img ne "")
      {
        $xml .= "<img type='dm_vs_time' width='700' height='240'>".$dm_vs_time_img."</img>";
      }
      $xml .= "</transient_candidates>";

      $image_string = $xml;

      Dada::logMsg(2, DL, "imageInfoThread: collected images");
    }
  }

  Dada::logMsg(1, DL, "imageInfoThread: exiting");
}


sub tapeInfoThread() {

  Dada::logMsg(1, DL, "tapeInfoThread: starting");

  my $sleep_time = 30;
  my $sleep_counter = 0;
  my $secondary_freq = 6;
  my $secondary_counter = 0;

  my $control_dir = $cfg{"SERVER_CONTROL_DIR"};
  my $swin_file   = $control_dir."/bpsr.swin.state";
  my $xfer_file   = $control_dir."/bpsr.xfer.state";

  my ($xml, $cmd, $result, $response, $tmp_str, $key, $val);
  my ($swin_tape, $swin_percent, $swin_time_left, $swin_pid, $num_swin, $swin_state);

  # Tape archiving
  my @swin_db = split(/:/, $cfg{"SWIN_DB_DIR"});
  my @swin_files_db = ();

  my $i = 0;
  my $j = 0;
  my @arr = ();
  my @arr2 = ();

  my @s_users = ();
  my @s_hosts = ();
  my @s_paths = ();
  my %swin = ();
  for ($i=0; $i<$cfg{"NUM_SWIN_DIRS"}; $i++) {
    @arr = split(/:/, $cfg{"SWIN_DIR_".$i});
    $s_users[$i] = $arr[0];
    $s_hosts[$i] = $arr[1];
    $s_paths[$i] = $arr[2];
  }
  
  my $xfer_state = "";
  my $xfer_pid = "";

  my $ready_to_send = 0;
  my @obs_fin = ();
  my %processing_count = ();
  my %processing_size = ();
  my %finished_count = ();
  my %finished_size = ();
  my %transferred_count = ();
  my %transferred_size = ();
  
  my $beams_finished = "";
  #my $beams_archived = "";
  my $beams_uploaded = "";
  my $beams_to_swin = "";
  my $beams_to_atnf = "";
  my $result = "";
  my @bits = ();
  my @keys = ();
  my @keys_finished = ();
  my ($key, $sum);
  my $obs = "";
  my $pid = "";
  my $cnt = 0;
  my $sz = 0;
  my $rval = 0;
  my $handle = 0;

  my $o = "";
  my $f = "";
  my $k = "";

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      $sleep_counter--;
      sleep(1);
    } else {

      $sleep_counter = $sleep_time;

      # determine the PID's of the current transfer manager and swin tapes
      $xfer_pid   = getDaemonPID($cfg{"SERVER_HOST"}, $cfg{"SERVER_XFER_PID_PORT"});
      $swin_pid   = getDaemonPID($cfg{"SERVER_HOST"}, $cfg{"SERVER_SWINTAPE_PID_PORT"});

      open FH, "<".$swin_file;
      read FH, $swin_state, 4096;
      close FH;
      chomp $swin_state;

      open FH, "<".$xfer_file;
      read FH, $xfer_state, 4096;
      close FH;
      chomp $xfer_state;

      # Update this information less frequently
      if ($secondary_counter > 0 ) {
        $secondary_counter--;
      } else {
        $secondary_counter = $secondary_freq;

        $swin_tape = "none";
        $swin_percent = 0;
        $swin_time_left = 0;
        for ($i=0; (($swin_tape eq "none") && ($i<=$#swin_files_db)); $i++) {
          @arr = split(/ /,$swin_files_db[$i]);
          if ($arr[3] eq "0") {
            $swin_tape = $arr[0];
            $swin_percent = sprintf("%5.2f",(int($arr[2]) / int($arr[1]))*100);
            $swin_time_left = sprintf("%5.1f",(((int($arr[1]) - int($arr[2])) * 1024.0) / (40.0 * 60.0)));
          }
        }

        Dada::logMsg(2, DL, "tapeInfoThread: percent swin=".$swin_percent."%");
        Dada::logMsg(2, DL, "tapeInfoThread: names swin=".$swin_tape);

        %processing_count = ();
        %processing_size = ();
        %finished_count = ();
        %finished_size = ();
        %transferred_count = ();
        %transferred_size = ();

        for ($i=0; (($i<8) && (!$quit_daemon)); $i++)
        {
          $cmd = "ssh -l bpsr hipsr".$i." \"web_results_helper.pl\"";
          Dada::logMsg(2, DL, "tapeInfoThread: ".$cmd);
          ($result, $tmp_str) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "tapeInfoThread: ".$result." ".$tmp_str);
          if ($result eq "ok")
          {
            @arr = split(/\n/, $tmp_str);
            for ($j=0; $j<=$#arr; $j++)
            {
              @bits = split(/ /,$arr[$j]);
              for ($k=1; $k<=$#bits; $k++)
              {
                ($pid, $cnt, $sz) = split(/:/, $bits[$k]);
                if ($bits[0] eq "PROCESSING")
                {
                  if (!exists($processing_count{$pid}))
                  {
                    $processing_count{$pid} = 0;
                    $processing_size{$pid} = 0;
                  }
                  $processing_count{$pid} += $cnt;
                  $processing_size{$pid} += $sz;
                  Dada::logMsg(2, DL, "tapeInfoThread: \$processing_size{".$pid."} += ".$sz);;

                }

                if ($bits[0] eq "FINISHED")
                { 
                  if (!exists($finished_count{$pid}))
                  {
                    $finished_count{$pid} = 0;
                    $finished_size{$pid} = 0;
                  }
                  $finished_count{$pid} += $cnt;
                  $finished_size{$pid} += $sz;
                  Dada::logMsg(2, DL, "tapeInfoThread: \$finished_size{".$pid."} += ".$sz);;
                }
                if ($bits[0] eq "TRANSFERRED")
                { 
                  if (!exists($transferred_count{$pid}))
                  {
                    $transferred_count{$pid} = 0;
                    $transferred_size{$pid} = 0;
                  }
                  $transferred_count{$pid} += $cnt;
                  $transferred_size{$pid} += $sz;
                }
              }
            }
          }
        }

        @keys = keys %processing_count;
        @keys_finished = keys %finished_count;
        foreach $key (@keys_finished)
        {
          Dada::logMsg(2, DL, "tapeInfoThread: testing if finished key [".$key."] is in processing keys");

          if (!(grep( /^$key$/, @keys)))
          {
            Dada::logMsg(2, DL, "tapeInfoThread: it wasn't, adding: ".$key);
            push @keys, $key;
          }
          else
          {
            Dada::logMsg(2, DL, "tapeInfoThread: it was, skipping: ".$key);
          }
        }

        $beams_finished = "";
        for $key ( @keys )
        {
          Dada::logMsg(2, DL, "tapeInfoThread: key=".$key);
          $sum = 0;
          if (exists($processing_size{$key}))
          {
            $sum += $processing_size{$key};
          }
          if (exists($finished_size{$key}))
          {
            $sum += $finished_size{$key};
          }
          $beams_finished .= $key."\t".sprintf("%0.1f", $sum)." GB\n";
        }
        if ($#keys == -1)
        {
          $beams_finished = "none";
        }

        Dada::logMsg(2, DL, "tapeInfoThread: beams_finished=".$beams_finished);

        # the number of beams waiting on raid to be sent to swin [for all projID's]
        $handle = Dada::connectToMachine($cfg{"RAID_HOST"}, $cfg{"RAID_WEB_MONITOR_PORT"}, 2);

        $beams_to_swin = "pipeline off";
        $beams_to_atnf = "pipeline off";
        #$beams_archived = "none";
        $beams_uploaded = "pipeline off";

        # ensure our file handle is valid
        if ($handle) 
        {
          ($result, $response) = Dada::sendTelnetCommand ($handle, "swin_send_info");
          if ($result eq "ok")
          {
            $beams_to_swin = formatRAIDReport($response);
          }
        
          ($result, $response) = Dada::sendTelnetCommand ($handle, "atnf_send_info");
          if ($result eq "ok")
          {
            $beams_to_atnf = formatRAIDReport($response);
          }

          ($result, $response) = Dada::sendTelnetCommand ($handle, "upload_info");
          if ($result eq "ok")
          {
            $beams_uploaded = formatRAIDReport($response);
          }

          #($result, $response) = Dada::sendTelnetCommand ($handle, "archived_info");
          #if ($result eq "ok")
          #{
          #  $beams_archived = formatRAIDReport($response);
          #}

          $handle->close();
        }
        
        # number of beams in staging area for SWIN_DIRS
        for ($i=0; $i<=$#s_users; $i++) {
          $cmd = "ssh -x -l ".$s_users[$i]." ".$s_hosts[$i]." 'cd ".$s_paths[$i]."/archive/".$swin_pid."; find . -mindepth 2 -maxdepth 2 -type d -printf \"\%f\\n\"' | wc -l";
          Dada::logMsg(2, DL, "tapeInfoThread: getting number of beams in staging_area for SWIN");
          $tmp_str = `$cmd`;
          chomp $tmp_str;
          $num_swin += int($tmp_str);
        }
        Dada::logMsg(2, DL, "tapeInfoThread: num_swin= ".$num_swin);
      }

      # $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>";
      $xml = "<tape_info>";

      $xml .= "<xfer_finished>".$beams_finished."</xfer_finished>";
      $xml .= "<xfer_uploaded>".$beams_uploaded."</xfer_uploaded>";
      $xml .= "<xfer_to_swin>".$beams_to_swin."</xfer_to_swin>";
      $xml .= "<xfer_to_atnf>".$beams_to_atnf."</xfer_to_atnf>";
      #$xml .= "<xfer_archived>".$beams_archived."</xfer_archived>";

      $xml .= "<swin_state>".$swin_state."</swin_state>";
      $xml .= "<swin_tape>".$swin_tape."</swin_tape>";
      $xml .= "<swin_percent>".$swin_percent."</swin_percent>";
      $xml .= "<swin_time_left>".$swin_time_left."</swin_time_left>";
      $xml .= "<swin_num>".$num_swin."</swin_num>";
      $xml .= "<swin_pid>".$swin_pid."</swin_pid>";

      $xml .= "</tape_info>";

      $tape_info_string = $xml;

      Dada::logMsg(2, DL, "tapeInfoThread: ".$tape_info_string);
    }
  }

  Dada::logMsg(1, DL, "tapeInfoThread: exiting");
}

#
# Monitors the STATUS_DIR for warnings and errors
#
sub statusInfoThread() 
{
  my $status_dir = $cfg{"STATUS_DIR"};

  Dada::logMsg(1, DL, "statusInfoThread: starting");

  my @files = ();
  my @arr = ();
  my $sleep_time = 10;
  my $sleep_counter = 0;

  my $i = 0;
  my $statuses = "";
  my $file = "";
  my $msg = "";
  my $pwc = ""; 
  my $tag = ""; 
  my $type = "";
  my $tmp_str = "";
  my $cmd = "";

  my %pwc_ids = ();
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++)
  {
    $pwc_ids{$cfg{"PWC_".$i}} = $i;
    $pwc_ids{sprintf("%02d",$i)} = $i;
  }

  while (!$quit_daemon) 
  {
    if ($sleep_counter > 0) 
    {
      sleep(1);
      $sleep_counter--;
    } 
    else
    {
      $sleep_counter = $sleep_time;

      $tmp_str = "";

      @files = ();

      $cmd = "ls -1 ".$status_dir;
      $statuses = `$cmd`;
      chomp($statuses);
      @files = split(/\n/, $statuses);

      # get the current warnings and errors
      for ($i=0; $i<=$#files; $i++) {
        $file = $files[$i];
        $msg = `tail -n 1 $status_dir/$file`;
        chomp $msg;

        @arr = ();
        @arr = split(/\./,$file);

        $pwc = "";
        $tag = "";
        $type = "";

        # for pwc, sys and src client errors
        if ($#arr == 2) 
        {
          $pwc = $pwc_ids{$arr[0]};
          $tag = $arr[1];
          $type = $arr[2];
        } 
        elsif ($#arr == 1)
        {
          $pwc = "server";
          $tag = $arr[0];
          $type = $arr[1];
        }
        else
        {
          Dada::logMsg(0, DL, "statusInfoThread: un-parseable status file: ".$file);
        }

        if ($type eq "warn")
        {
          $tmp_str .= "<daemon_status type='warning' pwc='".$pwc."' tag='".$tag."'>".$msg."</daemon_status>";
        }
        if ($type eq "error")
        {
          $tmp_str .= "<daemon_status type='error' pwc='".$pwc."' tag='".$tag."'>".$msg."</daemon_status>";
        }
      }

      $status_info = $tmp_str;
      Dada::logMsg(2, DL, "statusInfoThread: ".$status_info);

    }
  }

  Dada::logMsg(1, DL, "statusInfoThread: exiting");
}

#
# Maintains information with clients about DB, Disk and load
#
sub nodeInfoThread() 
{
  Dada::logMsg(1, DL, "nodeInfoThread: starting");

  my $sleep_time = 4;
  my $sleep_counter = 0;
  my $port = $cfg{"CLIENT_MASTER_PORT"};

  my @machines = ();
  my @results = ();
  my @responses = ();

  my $tmp_str = "";
  my $i = 0;
  my $result = "";
  my $response = "";

  # setup the list of machines that we will poll
  my %hosts = ();
  for ($i=0; $i<$cfg{"NUM_PWC"}; $i++) {
    $hosts{$cfg{"PWC_".$i}} = 1;
  }
  for ($i=0; $i<$cfg{"NUM_HELP"}; $i++) {
    $hosts{$cfg{"HELP_".$i}} = 1;
  }
  $hosts{"hipsr-srv0"}  = 1;

  @machines = keys %hosts;

  my $handle = 0;

  while (!$quit_daemon) {

    if ($sleep_counter > 0) {
      sleep(1);
      $sleep_counter--;
                                                                                                          
    # The time has come to check the status warnings
    } else {
      $sleep_counter = $sleep_time;

      @results = ();
      @responses = ();

      for ($i=0; ((!$quit_daemon) && $i<=$#machines); $i++) {

        $handle = Dada::connectToMachine($machines[$i], $port, 0);
        # ensure our file handle is valid
        if (!$handle) {
          $result = "fail";
          $response = "<status><host>".$machines[$i]."</host></status>";
        } else {
          ($result, $response) = Dada::sendTelnetCommand($handle, "get_status");
          $handle->close();
        }

        $results[$i] = $result;
        $responses[$i] = $response;
      }

      # now set the global string
      $tmp_str = "<node_statuses>";
      for ($i=0; $i<=$#machines; $i++) {
        $tmp_str .= $responses[$i];
      }
      $tmp_str .= "</node_statuses>";
      $node_info = $tmp_str;

      Dada::logMsg(2, DL, "nodeInfoThread: ".$node_info);
    }
  }  
  Dada::logMsg(1, DL, "nodeInfoThread: exiting");
}

#
# Maintains beam information about current observation, retrieving relevant data from
# TCS log file on TCS host
#
sub beamInfoThread() 
{
  Dada::logMsg(1, DL, "beamInfoThread: starting");
  
  my $results_dir = $cfg{"SERVER_RESULTS_DIR"};
  my $sleep_time = 10;
  my $sleep_counter = 0;

  my $cmd = "";
  my $result = "";
  my $rval = 0;
  my $response = "";

  my $tcs_host = "joffrey.atnf.csiro.au";
  my $tcs_user = "pksobs";
  my $tcs_path = "/home/pksobs/tcs/logs/";

  my $obs = "";
  my $beam_obs = "";

  my %rajs = ();
  my %decjs = ();
  my ($raj_deg, $decj_deg, $gl, $gb, $dm);
  my @lines = ();
  my $line = "";
  my @parts = ();
  my $i = 0;
  my @bits = ();
  my $beam = "";
  my $key = "";
  my $val = "";
  my $obs_beam_file = "";
  my $xml = "";

  while (!$quit_daemon)
  {
    # get the current / most recent observation
    $cmd = "find ".$results_dir." -ignore_readdir_race -maxdepth 1 -type d -name '2*' -printf '\%f\n' | sort | tail -n 1";
    Dada::logMsg(2, DL, "beamInfoThread: ".$cmd);
    ($result, $response) = Dada::mySystem($cmd);
    Dada::logMsg(3, DL, "beamInfoThread: ".$cmd);
    if ($result ne "ok")
    {
      Dada::logMsg(0, DL, "beamInfoThread: ".$cmd." failed: ".$response);
      sleep(1);
      next;
    }

    $obs = $response;
    Dada::logMsg(2, DL, "beamInfoThread: obs=".$obs);

    if ($beam_obs ne $obs)
    {
      # now get the RA and DEC from the tcs log file (currenlty being written) for this obs
      $cmd = "grep -E '  ra | dec ' ".$tcs_path."/".$obs."_bpsr.log";
      Dada::logMsg(2, DL, "beamInfoThread: ".$tcs_user."@".$tcs_host.":".$cmd);
      ($result, $rval, $response) = Dada::remoteSshCommand($tcs_user, $tcs_host, $cmd);
      Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$rval." ".$response);
      if (($result eq "ok") && ($rval == 0))
      {
        %rajs = ();
        %decjs = ();
        @lines = split(/\n/, $response);
        for ($i=0; $i<=$#lines; $i++)
        {
          $line = $lines[$i];
          @bits = split(/\s+/, $line);
          if ($#bits != 3)
          { 
            Dada::logMsg(0, DL, "beamInfoThread: badly formatted line: ".$line);
            next;
          }
          $beam = substr($bits[1], 5);
          $key  = $bits[2];
          $val  = $bits[3];
          Dada::logMsg(2, DL, "beamInfoThread: parsed beam=".$beam." key=".$key." val=".$val);

          if ($key eq "ra")
          {
            $rajs{$beam} = $val;
          }
          if ($key eq "dec")
          {
            $decjs{$beam} = $val;
          }
        }
        $obs_beam_file = $cfg{"SERVER_RESULTS_DIR"}."/".$obs."/beaminfo.xml";

        $xml = "";

        # convert RA, DEC from radians to degree string
        foreach $beam (sort keys %rajs)
        {
          # determine the GL and GB also
          $raj_deg = ($rajs{$beam} * 180) / pi;
          $decj_deg = ($decjs{$beam} * 180) / pi;

          # convert the RA
          Dada::logMsg(3, DL, "beamInfoThread: IN  rajs{".$beam."} = ".$rajs{$beam});
          ($result, $response) = Dada::convertRadiansToRA($rajs{$beam});
          Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, DL, "beamInfoThread: convertRadiansToRA failed: ".$response);
            next;
          }
          $rajs{$beam} = $response;

          #convert the DEC
          Dada::logMsg(3, DL, "beamInfoThread: IN  decjs{".$beam."} = ".$decjs{$beam});
          ($result, $response) = Dada::convertRadiansToDEC($decjs{$beam});
          Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, DL, "beamInfoThread: convertRadiansToDEC failed: ".$response);
            next;
          }
          $decjs{$beam} = $response;

          $cmd = "eq2gal.py ".$raj_deg." ".$decj_deg;
          Dada::logMsg(3, DL, "beamInfoThread: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$response);

          my ($gl, $gb) = split(/ /, $response, 2);

          # determine the DM for this GL and GB
          $cmd = "cd \$HOME/opt/NE2001/runtime; ./NE2001 ".$gl." ".$gb." 100 -1 | grep ModelDM | awk '{print \$1}'";
          Dada::logMsg(2, DL, "beamInfoThread: ".$cmd);
          ($result, $dm) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$dm);

          $xml .= "<beam_info beam='".$beam."' raj='".$rajs{$beam}."' decj='".$decjs{$beam}."' gl='".$gl."' gb='".$gb."' dm='".$dm."'>\n";

          # get a PSRCAT listing. FWHM = 14 arcmin = 0.233333 degrees, prefer 1/e rather than 1/2 which means 16.8 arcmin -> 0.28
          # $cmd = 'psrcat -c "JNAME DM RAJ DECJ S1400" -o short -nohead -nonumber -boundary "1 '.$rajs{$beam}.' '.$decjs{$beam}.' 0.28"';

          # so we get absolutely all sources, double radius (just diameter of beam)
          $cmd = 'psrcat -all -c "JNAME DM RAJ DECJ S1400" -o short -nohead -nonumber -boundary "1 '.$rajs{$beam}.' '.$decjs{$beam}.' 0.28"';
          Dada::logMsg(2, DL, "beamInfoThread: ".$cmd);
          ($result, $response) = Dada::mySystem($cmd);
          Dada::logMsg(3, DL, "beamInfoThread: ".$result." ".$response);
          if ($result ne "ok")
          {
            Dada::logMsg(0, DL, "beamInfoThread: psrcat command failed: ".$response);
            next;
          }
          @lines = split(/\n/, $response);
          foreach $line (@lines)
          {
            @bits = split(/\s+/, $line);
            if ($#bits != 4)
            {
              Dada::logMsg(0, DL, "beamInfoThread: malformed psrcat output: ".$line);
              next;
            }
            $xml .= "<psr name='".$bits[0]."' dm='".$bits[1]."' ra='".$bits[2]."' dec='".$bits[3]."' S1400='".$bits[4]."'/>\n";
          }

          $xml .= "</beam_info>\n";
        }

        # update the current beam_obs
        $beam_obs = $obs;

        Dada::logMsg(2, DL, "beamInfoThread: writing to ".$obs_beam_file);

        open FH, ">".$obs_beam_file;
        print FH $xml;
        close FH;

        $xml =~ s/\n//g;
        $beam_info_string = $xml;

      }    
    }

    Dada::logMsg(2, DL, "beamInfoThread: sleeping for ".$sleep_time." seconds");
    $sleep_counter = 0;
    while ((!$quit_daemon) && ($sleep_counter < $sleep_time))
    { 
      $sleep_counter++;
      sleep(1);
    }
  }

  Dada::logMsg(1, DL, "beamInfoThread: exiting");
}



#
# Opens a port to the machine, issues the command and collects the response
#
sub getDaemonPID($$) {

  my ($machine, $port) = @_;

  my $d_pid = "P858";

  my $response = "";

  Dada::logMsg(2, DL, "getDaemonPID: ".$machine.":".$port);
  my $handle = Dada::connectToMachine($machine, $port, 0);
  # ensure our file handle is valid
  if ($handle) {

    print $handle "get_pid\r\n";
    Dada::logMsg(2, DL, "getDaemonPID: -> get_pid");
    
    $response = Dada::getLine($handle);  
    Dada::logMsg(2, DL, "getDaemonPID: ".$response);

    $handle->close();

    if ($response ne "") {
      $d_pid = $response;
    }
  }

  Dada::logMsg(2, DL, "getDaemonPID: returning ".$d_pid);
  return $d_pid;

}

sub formatRAIDReport($)
{
  my ($str) = @_;

  my @lines = split(/\n/, $str);
  my $html = "<table>";
  $html = "";
  my ($key, $val, $line);

  foreach $line (@lines)
  {
    ($key, $val) = split(/ /,$line);
    $html .= $key."\t".sprintf("%0.1f", $val/1073741824)." GB\n";
  }

  if ($html eq "")
  {
    $html .= "none";
  }

  #$html .= "</table>";

  return $html;
}

