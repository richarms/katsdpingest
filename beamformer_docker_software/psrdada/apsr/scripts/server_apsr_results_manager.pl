#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Apsr;
use Dada::server_results_manager qw(%cfg);

#
# Global Variable Declarations
#
%cfg = Apsr::getConfig();

#
# Initialize module variables
#
$Dada::server_results_manager::dl = 1;
$Dada::server_results_manager::daemon_name = Dada::daemonBaseName($0);

# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = Dada::server_results_manager->main();

exit($result);

