#!/usr/bin/env perl

use lib $ENV{"DADA_ROOT"}."/bin";

use strict;
use warnings;
use Bpsr;
use Dada::server_sys_monitor qw(%cfg);

#
# Global Variable Declarations
#
%cfg = Bpsr::getConfig();

#
# Initialize module variables
#
$Dada::server_sys_monitor::dl = 1;
$Dada::server_sys_monitor::daemon_name = Dada::daemonBaseName($0);
$Dada::server_sys_monitor::master_log_prefix = "nexus";
$Dada::server_sys_monitor::log_host = $cfg{"SERVER_HOST"};
$Dada::server_sys_monitor::log_port = $cfg{"SERVER_SYS_LOG_PORT"};

# Autoflush STDOUT
$| = 1;

my $result = 0;
$result = Dada::server_sys_monitor->main();

exit($result);


