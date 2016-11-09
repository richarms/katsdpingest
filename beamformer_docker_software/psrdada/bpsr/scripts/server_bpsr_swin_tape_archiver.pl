#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Bpsr;
use Dada::server_tape_archiver qw(%cfg);

#
# Global Variable Declarations
#
%cfg = Bpsr::getConfig();

sub usage() {
  print STDERR "Usage: ".$0." PID\n";
  print STDERR "  PID   Project ID to process\n";
}

#
# Initialize module variables
#
$Dada::server_tape_archiver::dl = 1;
$Dada::server_tape_archiver::daemon_name = Dada::daemonBaseName($0);
$Dada::server_tape_archiver::robot = 1;
$Dada::server_tape_archiver::type = "swin";
$Dada::server_tape_archiver::ssh_prefix = "ssh -l pulsar shrek210 'ssh -l dada apsr-evlbi.atnf.csiro.au \"";
$Dada::server_tape_archiver::ssh_suffix = "\"'";
$Dada::server_tape_archiver::required_host = "shrek201.ssi.swin.edu.au";

# Autoflush STDOUT
$| = 1;

if ($#ARGV != 0) {
  usage();
  exit(1);
}

$Dada::server_tape_archiver::pid = $ARGV[0];

my $result = 0;

$result = Dada::server_tape_archiver->main();

exit($result);

