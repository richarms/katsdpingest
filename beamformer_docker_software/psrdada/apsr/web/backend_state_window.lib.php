<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class backend_state extends apsr_webpage 
{

  function backend_state()
  {
    apsr_webpage::apsr_webpage();
    $this->title = "APSR Backend State Window";
  }

  function javaScriptCallback()
  {
    return "backend_state_request();";
  }

  function printJavaScriptHead()
  {

    $inst = new apsr();

?>
    <script type="text/javascript" src="/js/soundmanager2.js"></script>

    <script type='text/javascript'>  

      // setup Sound Manager and create the required files
      var sound_loaded = "false";
      var play_it = "false";

      // directory where SM2 .SWFs live
      soundManager.url = '/sounds/sm2-swf-movies/';
      soundManager.debugMode = false;
      soundManager.volume = 100;
      soundManager.waitForWindowLoad = true;

      // callback function which is called once the page has been loaded */
      soundManager.onload = function() {
        soundManager.createSound('fatal_error','/sounds/a_fatal_error_has_occurred.mp3');
        sound_loaded = "true";
      }

      // Information for the warn/error div scroller
      var scrollerwidth="450px";
      var scrollerheight="50px";
      var scrollerspeed=1;        // Scrollers speed here (larger is faster 1-10)
      var scrollercontent='OK'
      var pauseit=1
      var scroll_it = "false";
      var scroll_id;

      // Set all the imgs and links not in the excluded array to green
      function reset_others(excluded) 
      {

        var imgs = document.getElementsByTagName('img');
        var links = document.getElementsByTagName('a');
        var i=0;
        var green_count = 0;
        for (i=0; i<imgs.length; i++) {
          if (excluded.indexOf(imgs[i].id) == -1) {
            imgs[i].src = "/images/green_light.png";
            green_count++;
          }
        }
        for (i=0; i<links.length; i++) {
          if (excluded.indexOf(links[i].id) == -1) {
            links[i].title = "OK";
            links[i].href = "javascript:void(0)";
          }
        }

        // clear the scrolling text if everything is ok 
        if (green_count == imgs.length) {
          clearInterval(scroll_id);
          scroll_it = "false";
          play_it = "false";
        }
      }


      scrollerspeed=(document.all)? scrollerspeed : Math.max(1, scrollerspeed-1) //slow speed down by 1 for NS
      var copyspeed=scrollerspeed
      var iedom=document.all||document.getElementById
      var actualheight=''
      var cross_scroller, ns_scroller
      var pausespeed=(pauseit==0)? copyspeed: 0

      function populate() 
      {
        cross_scroller=document.getElementById? document.getElementById("iescroller") : document.all.iescroller
        cross_scroller=document.getElementById? document.getElementById("iescroller") : document.all.iescroller
        cross_scroller.style.top=parseInt(scrollerheight)+8+"px"
        cross_scroller.innerHTML=scrollercontent
        actualheight=cross_scroller.offsetHeight
        scroll_id=setInterval("scroll_scroller()",100)
      }

      function scroll_scroller() {

        if (parseInt(cross_scroller.style.top)>(actualheight*(-1)+4)) 
          cross_scroller.style.top=parseInt(cross_scroller.style.top)-copyspeed+"px"
        else 
          cross_scroller.style.top=parseInt(scrollerheight)+4+"px"

      }

      function handle_backend_state_request(bs_http_request) 
      {
        if (bs_http_request.readyState == 4) 
        {

          var response = String(bs_http_request.responseText)
          var set = new Array();
          var scroll_msg = "";

          if (response.length > 0) 
          {

            var lines = response.split("\n");
            var num_pwc = 0;
            var num_srv = 0;
            var values = "";

            var key;
            var state;
            var message;
            var host;
            var log_level;
            var img_id;
            var link_id;
            var instrument = "<?echo INSTRUMENT?>"
            var log_file;
            var loglengthIndex = "";
            var loglength = "";

            if ((typeof parent != "undefined") && (typeof parent.parent != "undefined") && 
                (typeof parent.parent.logheader != "undefined")) {
              loglengthIndex = parent.parent.logheader.document.getElementById("loglength").selectedIndex
              loglength = parent.parent.logheader.document.getElementById("loglength").options[loglengthIndex].value
            }
            var link_str = "";

            if (response.indexOf("Could not connect to") == -1) 
            {

              for (i=0; i < lines.length; i++) {

                values = lines[i].split(":::");
      
                key = values[0];
                state = values[1];
                message = values[2];

                keyparts = key.split("_");
                /* PWC, SRC or SYS message */
                if ((keyparts[0] == "pwc") || (keyparts[0] == "src") || (keyparts[0] == "sys")) {
                  host = keyparts[1];
                  log_file=instrument+"_"+keyparts[0]+"_monitor";
                } else {
                  host = "nexus";
                  log_file = key;
                }

                img_id = "img_"+key;
                link_id = "a_"+key;

                set.push("img_"+key);
                set.push("a_"+key);
      
                /* get light val */
                if (state == <?echo STATUS_OK?>) {
                  document.getElementById(img_id).src = "/images/green_light.png";
                  log_level="all";
                }
                if (state == <?echo STATUS_WARN?>) {
                  document.getElementById(img_id).src = "/images/yellow_light.png";
                  scroll_msg += "<font color=\"#c7d210\"><b>WARNING:</b></font><font size=-2> " + host + "  " + message + "</font><br>";
                  log_level="warn";
                  if (scroll_it == "false") {
                    scroll_it = "true";
                    populate();
                  }

                }
                if (state == <?echo STATUS_ERROR?>) {
                  document.getElementById(img_id).src = "/images/red_light.png";
                  log_level="error";
                  scroll_msg +=  "<font color=\"#c80000\"><b>ERROR:</b></font> " + host + " " + message + "<br>";
                  if (scroll_it == "false") {
                    scroll_it = "true";
                    populate();
                  }
                  if (play_it == "false") {
                    if (sound_loaded == "true") {
                      soundManager.play('fatal_error');
                      play_it = "true";
                    }
                  }
                }

                document.getElementById(link_id).title = message;
                if (loglength == "") { 
                  document.getElementById(link_id).href = "logwindow.php?machine="+host+"&loglevel="+log_level+"&daemon="+log_file+"&autoscroll=true"
                } else {
                  document.getElementById(link_id).href = "logwindow.php?machine="+host+"&loglevel="+log_level+"&loglength="+loglength+"&daemon="+log_file+"&autoscroll=true"
                }
              }
            }
          }
          if ((typeof cross_scroller != "undefined") && (scroll_msg != cross_scroller.innerHTML)) {
            cross_scroller.innerHTML = scroll_msg;
            actualheight=cross_scroller.offsetHeight
          }
          reset_others(set);
        }
      }

      function backend_state_request() 
      {
        var host = "<?echo $inst->config["SERVER_HOST"];?>";
        var port = "<?echo $inst->config["SERVER_WEB_MONITOR_PORT"];?>";
        var url = "backend_state.lib.php?update=true&host="+host+"&port="+port;

        if (window.XMLHttpRequest)
          bs_http_request = new XMLHttpRequest();
        else
          bs_http_request = new ActiveXObject("Microsoft.XMLHTTP");

        bs_http_request.onreadystatechange = function() {
          handle_backend_state_request(bs_http_request)
        };
        bs_http_request.open("GET", url, true);
        bs_http_request.send(null);
      }


      function set_active_machine(machine) 
      {

        var oldmachineIndex = parent.parent.logheader.document.getElementById("active_machine").selectedIndex
        var oldmachine = parent.parent.logheader.document.getElementById("active_machine").options[oldmachineIndex].value
        var loglengthIndex = parent.parent.logheader.document.getElementById("loglength").selectedIndex
        var loglength = parent.parent.logheader.document.getElementById("loglength").options[loglengthIndex].value

        /* Unselect the currently selected machine */
        document.getElementById("td_txt_"+oldmachine).className = "notselected"
        document.getElementById("td_pwc_"+oldmachine).className = "notselected"
        document.getElementById("td_src_"+oldmachine).className = "notselected"
        document.getElementById("td_sys_"+oldmachine).className = "notselected"
       
        select_value_set(parent.parent.logheader.document.getElementById("active_machine"), machine);                                                                               
        /* Select the new machine */
        document.getElementById("td_txt_"+machine).className = "selected"
        document.getElementById("td_pwc_"+machine).className = "selected"
        document.getElementById("td_src_"+machine).className = "selected"
        document.getElementById("td_sys_"+machine).className = "selected"
      }

      function select_value_set(select_object, value) 
      {
        for (index = 0; index < select_object.length; index++) 
        {
         if (select_object[index].value == value)
           select_object.selectedIndex = index;
         }
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {

    $inst = new apsr();

    $server_log_info = $inst->serverLogInfo();

    $keys = array_keys($server_log_info);
    $server_daemons = array();
    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      if ($server_log_info[$k]["tag"] == "server") {
        $server_daemons[$k] = $server_log_info[$k];
      }
    }

    # get the PWC, SRC and SYS status array
    $pwc_status = $inst->getClientStatuses();

    $nexus_status = STATUS_OK;
    $nexus_message = "no probs mate";

?>
  <table id="parent_table" border=0 cellspacing=0 cellpadding=0>
    <tr>
      <td>

      <table border=0 cellspacing=0 cellpadding=0 id="client_status_table">
        <tr>
          <td></td>
          <td width="30px" align="center" valign="top" class="selected" id="td_txt_nexus">
            <?echo $this->statusLink("nexus","Nexus");?>
          </td>
<?
    for ($i=0; $i<$inst->config["NUM_PWC"]; $i++) 
    {
      $host = $inst->config["PWC_".$i];
      echo "          <td width='30' align='center' class='notselected' id='td_txt_".$host."'>\n";
      echo "            ".$this->statusLink($host, $i)."\n";
      echo "          </td>\n";
    }
?>
        </tr>
  
        <tr height="25px">
          <td valign="center" class="notselected">PWC:</td>
          <td align="center" class="selected" id="td_nexus_pwc">
            <?echo $this->statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_pwc", "img_nexus_pwc")?>
          </td>
<?

    // PWC status
    for ($i=0; $i<$inst->config["NUM_PWC"]; $i++) 
    {

      $machine = $inst->config["PWC_".$i];
      $status = $pwc_status["PWC_".$i."_STATUS"];
      $message = $pwc_status["PWC_".$i."_MESSAGE"];
      $linkid = "a_pwc_".$machine;
      $imgid = "img_pwc_".$machine;
      $tdid = "td_pwc_".$machine;
?>
          <td class="notselected" id="<?echo $tdid?>" height="20px">
            <? echo $this->statusLight($status,$message,$machine,$linkid,$imgid);?>
          </td>
<? } ?>
        </tr>

        <!-- SRC Statuses -->
        <tr height="25px">
          <td valign="center" class="notselected">SRC:</td>
          <td align="center" class="selected" id="td_nexus_src">
            <?echo $this->statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_src", "img_nexus_src")?>
          </td>
<?
    for ($i=0; $i<$inst->config["NUM_PWC"]; $i++) 
    {
      $machine = $inst->config["PWC_".$i];
      $status = $pwc_status["SRC_".$i."_STATUS"];
      $message = $pwc_status["SRC_".$i."_MESSAGE"];
      $linkid = "a_src_".$machine;
      $imgid = "img_src_".$machine;
      $tdid = "td_src_".$machine;
?>
          <td class="notselected" align="center" id="<?echo $tdid?>">
            <?echo $this->statusLight($status,$message,$machine,$linkid,$imgid);?>
          </td>
<?  } ?>
        </tr>

        <!-- SYS Statuses -->
        <tr height="25px">
          <td valign="center" class="notselected">SYS:</td>
          <td align="center" class="selected" id="td_nexus_sys">
            <?echo $this->statusLight($nexus_status, $nexus_message, "nexus", "a_nexus_sys", "img_nexus_sys")?>
          </td>
<?
    for ($i=0; $i<$inst->config["NUM_PWC"]; $i++) 
    {
      $machine = $inst->config["PWC_".$i];
      $status = $pwc_status["SYS_".$i."_STATUS"];
      $message = $pwc_status["SYS_".$i."_MESSAGE"];
      $linkid = "a_sys_".$machine;
      $imgid = "img_sys_".$machine;
      $tdid = "td_sys_".$machine
                                                                                                                         
?>
          <td class="notselected" align="center" id="<?echo $tdid?>">
            <?echo $this->statusLight($status,$message,$machine,$linkid,$imgid);?>
          </td>
<?  } ?>
                                                                                                                           
        </tr>
      </table>
    </td>
    
    <td width="20px"</td>

    <td>

      <table border=0 cellspacing=0 cellpadding=0 id="server_status_table">
        <tr>
          <td class="notselected">&nbsp;</td>
        </tr>
<?
    $j = 0;
    $keys = array_keys($server_daemons);

    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      
      if ($j == 0) {
        echo "        <tr height='25px'>\n";
      }

      $link_id = "a_".$k;
      $img_id = "img_".$k;
      $td_id = "td_".$k;
      $name = $server_daemons[$k]["shortname"];

      echo "          <td id='".$td_id."' class='notselected' style='text-align: left'>\n";
      echo "            ".$this->statusLight(0, "OK", "nexus", $link_id, $img_id)."\n";
      echo "          </td>\n";
      echo "          <td class=\"notselected\"  style=\"text-align: left\">\n";
      echo "            &nbsp;&nbsp;".$name."&nbsp;&nbsp;&nbsp;&nbsp;\n";
      echo "          </td>\n";

      if ($j == 2) {
        echo "  </tr>\n";
      }

      if ($j == 2) {
        $j=0;
      } else { 
        $j++; 
      }
    } 
?>
      </table>
    </td>
  </tr>
  <tr>
    <td colspan=3 align=left>

      <div style="position:relative;width:100%;height:50px;overflow:hidden" onMouseover="copyspeed=pausespeed" onMouseout="copyspeed=scrollerspeed">
        <div id="iescroller" style="position:absolute;left:0px;top:0px;width:100%;"></div>
      </div>

    </td>
  </tr>
</table>
<?
  }

  function printUpdateHTML($get)
  {
    $host = $get["host"];
    $port = $get["port"];

    $timeout = 1;
    list ($socket, $result) = openSocket($host, $port, $timeout);
    if ($result == "ok") {

      $bytes_written = socketWrite($socket, "status_info\r\n");
      ($result, $string) = socketRead($socket);
      socket_close($socket);

    } else {
      $string = "Could not connect to $host:$port<BR>\n";
    }

    echo $string;

  }


  function statusLink($machine, $linktext) 
  {

    $url = "machine_status.php?machine=".$machine;

    $string = '<a target="logwindow" href="'.$url.'" '.
              'onClick="set_active_machine(\''.$machine.'\')">'.
              $linktext.'</a>';

    //return $string;
    return $linktext;

  }

  function statusLight($status, $message, $machine, $linkid, $imgid) 
  {

    $daemon = "";
    if (strstr($linkid, "nexus_pwc") !== FALSE) {
      $daemon = "&daemon=pwc";
    }
    if (strstr($linkid, "nexus_src") !== FALSE) {
      $daemon = "&daemon=src";
    }
    if (strstr($linkid, "nexus_sys") !== FALSE) {
      $daemon = "&daemon=sys";
    }

    if ($status == STATUS_OK) {
      //$url = "machine_status.php?machine=".$machine;
      $url = "logwindow.php?machine=".$machine."&loglevel=all".$daemon;
      $title = "OK: ".$machine;
    } else if ($status == STATUS_WARN) {
      $url = "logwindow.php?machine=".$machine."&loglevel=warn".$daemon;
      $title = "Warning: ".$machine;
    } else if ($status == STATUS_ERROR) {
      $url = "logwindow.php?machine=".$machine."&loglevel=error".$daemon;
      $title = "Error: ".$machine;
    } else {
      $url = "logwindow.php?machine=".$machine."&loglevel=all".$daemon;
      $title = "Message";
    }

    $string = '<a target="logwindow" id="'.$linkid.'" href="'.$url.'" '.
              'onClick="set_active_machine(\''.$machine.'\')"';

    $string .= ' TITLE="'.$title.': '.$message.'">';

    #$string .= ' TITLE="header=['.$title.'] body=['.$message.'] '.
    #            'cssbody=[ttbody] cssheader=[ttheader]">';

    $string .= '
       <img id="'.$imgid.'" border="none" width="15px" height="15px" src="/images/';

    if ($status == STATUS_OK) {
      return $string."green_light.png\" alt=\"OK\">\n      </a>\n";
    } else if ($status == STATUS_WARN) {
      return $string."yellow_light.png\" alt=\"WARN\">\n      </a>\n";
    } else {
      return $string."red_light.png\" alt=\"ERROR\">\n      </a>\n";
    }
  }
}

handledirect("backend_state");

