#!/usr/bin/env perl

###############################################################################
# 
# Uses Dada::server_pwc_monitor to monitor CASPSR's warning and error messages
# from the PWCs
#
##############################################################################

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Caspsr;
use Dada::server_pwc_monitor qw(%cfg);

#
# Global Variable Declarations
#
%cfg = Caspsr::getConfig();

#
# Initialize module variables
#
$Dada::server_pwc_monitor::dl = 1;
$Dada::server_pwc_monitor::daemon_name = Dada::daemonBaseName($0);
$Dada::server_pwc_monitor::log_host = $cfg{"SERVER_HOST"};
$Dada::server_pwc_monitor::log_port = $cfg{"PWCC_LOGPORT"};

# Autoflush STDOUT and STDERR
my $ofh = select STDOUT;
$| = 1;
select STDERR;
$| = 1;
select $ofh;

my $result = 0;
$result = Dada::server_pwc_monitor->main();

exit($result);

