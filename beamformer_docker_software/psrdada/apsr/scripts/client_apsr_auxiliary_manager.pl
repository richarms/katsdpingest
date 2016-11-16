#!/usr/bin/env perl

###############################################################################
#
# This script transfers data from a directory on the pwc, to a directory on the
# nexus machine

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;        
use warnings;        
use Apsr;
use Dada::client_auxiliary_manager qw(%cfg);


#
# Global Variables
# 
%cfg = Apsr::getConfig();

#
# Initialize module variables
#
$Dada::client_auxiliary_manager::dl = 1;
$Dada::client_auxiliary_manager::daemon_name = Dada::daemonBaseName($0);
$Dada::client_auxiliary_manager::user = "apsr";
$Dada::client_auxiliary_manager::dada_header_cmd = "dada_header -k ".lc($cfg{"AUXILIARY_DATA_BLOCK"});


# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = Dada::client_auxiliary_manager->main();

exit($result);

