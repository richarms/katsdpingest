<?PHP 
include("definitions_i.php");
include("functions_i.php");

$config = getConfigFile(SYS_CONFIG,TRUE);
$host = $config["SERVER_HOST"];
$port = $config["SERVER_WEB_MONITOR_PORT"];

list ($socket, $result) = openSocket($host, $port);

if ($result == "ok") {

  $bytes_written = socketWrite($socket, "curr_obs\r\n");
  list ($result, $string) = socketRead($socket);
  socket_close($socket);

} else {
  $string = "Could not connect to $host:$port<BR>\n";
}

echo $string;
?>
