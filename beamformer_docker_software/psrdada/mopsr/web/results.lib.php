<?PHP

include_once("mopsr.lib.php");
include_once("mopsr_webpage.lib.php");

class results extends mopsr_webpage 
{

  var $filter_types = array("", "SOURCE", "FREQ", "BW", "UTC_START", "PROC_FILE");
  var $cfg = array();
  var $length;
  var $offset;
  var $inline_images;
  var $filter_type;
  var $filter_value;
  var $class = "new";
  var $results_dir;
  var $results_link;
  var $archive_dir;

  function results()
  {
    mopsr_webpage::mopsr_webpage();
    $inst = new mopsr();
    $this->cfg = $inst->config;

    $this->length = (isset($_GET["length"])) ? $_GET["length"] : 20;
    $this->offset = (isset($_GET["offset"])) ? $_GET["offset"] : 0;
    $this->inline_images = (isset($_GET["inline_images"])) ? $_GET["inline_images"] : "false";
    $this->filter_type = (isset($_GET["filter_type"])) ? $_GET["filter_type"] : "";
    $this->filter_value = (isset($_GET["filter_value"])) ? $_GET["filter_value"] : "";
    $this->results_dir = $this->cfg["SERVER_RESULTS_DIR"];
    $this->archive_dir = $this->cfg["SERVER_ARCHIVE_DIR"];
    $this->results_link = "/mopsr/results";
    $this->results_title = "Recent Results";
    $this->class = (isset($_GET["class"])) ? $_GET["class"] : "new";
    if ($this->class == "old")
    {
      $this->results_dir = $this->cfg["SERVER_OLD_RESULTS_DIR"];
      $this->archive_dir = $this->cfg["SERVER_OLD_ARCHIVE_DIR"];
      $this->results_link = "/mopsr/old_results";
      $this->results_title = "Archived Results";
    }
  }

  function printJavaScriptHead()
  {
?>
    <style type="text/css">
      .processing {
        background-color: #FFFFFF;
        padding-right: 10px;
      }

      .finished {
        background-color: #cae2ff;
      }

      .transferred {
        background-color: #caffe2;
      }
    </style>

    <script type='text/javascript'>

      var offset = <?echo $this->offset?>;
  
      function getLength()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        return parseInt(length);
      }

      function getOffset()
      {
        return offset;
      }

      function setOffset(value)
      {
        offset = value;
      }

      // If a page reload is required
      function changeLength() {

        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;

        var show_inline;
        if (document.getElementById("inline_images").checked)
          show_inline = "true";
        else
          show_inline = "false";

        var offset = getOffset()

        var url = "results.lib.php?single=true&offset="+offset+"&length="+length+"&inline_images="+show_inline;

        i = document.getElementById("filter_type").selectedIndex;
        var filter_type = document.getElementById("filter_type").options[i].value;

        var filter_value = document.getElementById("filter_value").value;

        if ((filter_value != "") && (filter_type != "")) {
          url = url + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }
        document.location = url;
      }

      function toggle_images()
      {
        var i = document.getElementById("displayLength").selectedIndex;
        var length = document.getElementById("displayLength").options[i].value;
        var img;
        var show_inline

        if (document.getElementById("inline_images").checked) {
          show_inline = true;
          document.getElementById("IMAGE_TR").innerHTML = "IMAGE";
        } else {
          show_inline = false;
          document.getElementById("IMAGE_TR").innerHTML = "";
        }

        for (i=0; i<length; i++) {
          img = document.getElementById("img_"+i);
          if (show_inline) 
          {
            img.style.display = "";
            img.className = "processing";
          }
          else
          {
            img.style.display = "none";
          }
        }
      }

      function results_update_request() 
      {
        var ru_http_requset;

        // get the offset 
        var offset = getOffset();
        var length = getLength();

        var url = "results.lib.php?update=true&offset="+offset+"&length="+length+"&class=<?echo $this->class?>";

        // check if inline images has been specified
        if (document.getElementById("inline_images").checked)
          show_inline = "true";
        else
          show_inline = "false";
        url = url + "&inline_images="+show_inline;

        // check if a filter has been specified
        var u, filter_type, filter_vaule;
        i = document.getElementById("filter_type").selectedIndex;
        filter_type = document.getElementById("filter_type").options[i].value;
        filter_value = document.getElementById("filter_value").value;
        if ((filter_value != "") && (filter_type != "")) {
          url = url + "&filter_type="+filter_type+"&filter_value="+filter_value;
        }

        if (window.XMLHttpRequest)
          ru_http_request = new XMLHttpRequest()
        else
          ru_http_request = new ActiveXObject("Microsoft.XMLHTTP");
    
        ru_http_request.onreadystatechange = function() 
        {
          handle_results_update_request(ru_http_request)
        }

        ru_http_request.open("GET", url, true)
        ru_http_request.send(null)

      }

      function handle_results_update_request(xml_request) 
      {
        if (xml_request.readyState == 4) {

          var xmlDoc=xml_request.responseXML;
          var xmlObj=xmlDoc.documentElement; 

          var i, j, k, result, key, value, span, this_result;

          var results = xmlObj.getElementsByTagName("result");

          // for each result returned in the XML DOC
          for (i=0; i<results.length; i++) {

            result = results[i];
            this_result = new Array();

            for (j=0; j<result.childNodes.length; j++) 
            {

              // if the child node is an element
              if (result.childNodes[j].nodeType == 1) {
                key = result.childNodes[j].nodeName;
                // if there is a text value in the element
                if (result.childNodes[j].childNodes.length == 1) {
                  value = result.childNodes[j].childNodes[0].nodeValue;
                } else {
                  value = "";
                }
                this_result[key] = value;
              }
            }

            utc_start = this_result["UTC_START"];

            for ( key in this_result) {
              value = this_result[key];

              if (key == "SOURCE") {

                // ignore

              } else if (key == "IMG") {

                var url = "result.lib.php?single=true&utc_start="+utc_start+"&class=<?echo $this->class?>";
                var link = document.getElementById("link_"+i);
                var img = value.replace("112x84","400x300");
                link.href = url;
                link.onmouseover = new Function("Tip(\"<img src='<?echo $this->results_link?>/"+utc_start+"/"+img+"' width=401 height=301>\")");
                link.onmouseout = new Function("UnTip()");
                link.innerHTML = this_result["SOURCE"];

                try {
                  document.getElementById("img_"+i).src = "<?echo $this->results_link?>/"+utc_start+"/"+value;
                } catch (e) {
                  // do nothing
                }

              } else if (key == "processing") {
                if (value == 1)
                  document.getElementById("row_"+i).className = "processing";
                else
                  document.getElementById("row_"+i).className = "finished";
  
              } else {
                try {
                  span = document.getElementById(key+"_"+i);
                  span.innerHTML = value;
                } catch (e) {
                  // do nothing 
                }
              }
            } // end for
          }
          
          // update then showing_from and showing_to spans
          var offset = getOffset();
          var length = getLength();
          document.getElementById("showing_from").innerHTML = offset;
          document.getElementById("showing_to").innerHTML = (offset + length);
        }
      }

    </script>
<?
  }

  function printJavaScriptBody()
  {
?>
    <!--  Load tooltip module for images as tooltips, hawt -->
    <script type="text/javascript" src="/js/wz_tooltip.js"></script>
<?
  }

  function printHTML() 
  {
?>

    <table cellpadding="10px" width="100%">

      <tr>
        <td width='210px' height='60px'><img src='/mopsr/images/mopsr_logo.png' width='200px' height='60px'></td>
        <td align=left><font size='+2'><?echo $this->results_title?></font></td>
      </tr>

      <tr>
        <td valign="top" width="200px">
<?
    $this->openBlockHeader("Search Parameters");
?>
    <table>
      <tr>
        <td>Filter</td>
        <td>
          <select name="filter_type" id="filter_type">
<?
          for ($i=0; $i<count($this->filter_types); $i++) {
            $t = $this->filter_types[$i];
            echoOption($t, $t, FALSE, $this->filter_type);
          }
?>
          </select>
        </td>
      </tr>
      <tr>
        <td>For</td>
        <td><input name="filter_value" id="filter_value" value="<?echo $this->filter_value?>" onChange="changeLength()"></td>
      </tr>

      <tr>
        <td>Images</td>
        <td>
          <input type=checkbox id="inline_images" name="inline_images" onChange="toggle_images()"<? if($this->inline_images == "true") echo " checked";?>>
        </td>
      </tr>

      <tr>
        <td>Show</td>
        <td>
          <select name="displayLength" id="displayLength" onChange='changeLength()'>
<?
        echoOption("20", "20", FALSE, $this->length);
        echoOption("50", "50", FALSE, $this->length);
        echoOption("100", "100", FALSE, $this->length);
?>
          </select>
        </td>
      </tr>
      <tr>
        <td colspan=2><a href="/mopsr/results.lib.php?single=true">Recent MOPSR Results</a></td>
      </tr>
      <tr>
        <td colspan=2><a href="/mopsr/results.lib.php?single=true&class=old">Archived MOPSR Results</a></td>
      </tr>

    </table>
  <?
    $this->closeBlockHeader();

    echo "</td><td>\n";

    $this->openBlockHeader("Matching Observations");

    $cmd = "";

    if (($this->filter_type == "") && ($this->filter_value == "")) {
      $cmd = "find ".$this->results_dir." -maxdepth 2 -name 'obs.info' | wc -l";
    } else {
      if ($this->filter_type == "UTC_START") {
        $cmd = "find ".$this->results_dir."/*".$this->filter_value."* -maxdepth 1 -name 'obs.info' | wc -l";
      } else {
        $cmd = "find ".$this->results_dir." -maxdepth 2 -type f -name obs.info | xargs grep ".$this->filter_type." | grep ".$this->filter_value." | wc -l";
      }
    }
    $total_num_results = exec($cmd);

    $results = $this->getResultsArray($this->results_dir,
                                      $this->offset, $this->length, 
                                      $this->filter_type, $this->filter_value);

    ?>
    <div style="text-align: right; padding-bottom:10px;">
      <span style="padding-right: 10px">
        <a href="javascript:newest()">&#171; Newest</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:newer()">&#8249; Newer</a>
      </span>
      <span style="padding-right: 10px">
        Showing <span id="showing_from"><?echo $this->offset?></span> - <span id="showing_to"><?echo (min($this->length, $total_num_results))?></span> of <?echo $total_num_results?>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:older()">Older &#8250;</a>
      </span>
      <span style="padding-right: 10px">
        <a href="javascript:oldest()">Oldest &#187;</a>
      </span>
    </div>

    <script type="text/javascript">

      var total_num_results = <?echo $total_num_results?>;

      function newest()
      {
        setOffset(0);
        results_update_request();    
      }

      function newer()
      {
        var length = getLength();
        var offset = getOffset();
        if ((offset - length) < 0)
          offset = 0;
        else
          offset -= length;
        setOffset(offset);
        results_update_request();

      }

      function older() 
      {
        var length = getLength();
        var offset = getOffset();
        if ((offset + length > total_num_results) < 0)
          offset = total_num_results - length;
        else
          offset += length;
        setOffset(offset);
        results_update_request();
      }

      function oldest()
      {
        var length = getLength();
        var offset = getOffset();
        if ((total_num_results - length) < 0)
          offset = 0;
        else
          offset = total_num_results - length;
        setOffset(offset);
        results_update_request();
      }

    </script>

    <table width="100%">
      <tr>
<?
      if ($this->inline_images == "true")
        echo "        <th id='IMAGE_TR' align=left>IMAGE</th>\n";
      else 
        echo "        <th id='IMAGE_TR' align=left></th>\n";
?>
        <th align=left>SOURCE</th>
        <th align=left>UTC START</th>
        <th align=left>CONFIG</th>
<!--        <th align=left>FREQ</th>
        <th align=left>BW</th>-->
        <th align=left>LENGTH</th>
        <th align=left>PROC_FILEs</th>
        <th class="trunc">Annotation</th>
      </tr>
<?

        $keys = array_keys($results);
        rsort($keys);
        $result_url = "result.lib.php";

        for ($i=0; $i < count($keys); $i++) {

          $k = $keys[$i];
          $r = $results[$k];
          
          $url = $result_url."?single=true&utc_start=".$k."&class=".$this->class;

          // guess the larger image size
          $image = $r["IMG"];

          $mousein = "onmouseover=\"Tip('<img src=\'".$this->results_link."/".$k."/".$image."\' width=201 height=151>')\"";
          $mouseout = "onmouseout=\"UnTip()\"";
  
          // If archives have been finalised and its not a brand new obs
          echo "  <tr id='row_".$i."' class='".(($results[$keys[$i]]["processing"] === 1) ? "processing" : "finished")."'>\n";

          // IMAGE
          if ($this->inline_images == "true") 
            $style = "";
          else
            $style = "display: none;";

          $bg_style = "class='processing'";

          echo "    <td ".$bg_style."><img style='".$style."' id='img_".$i."' src=".$this->results_link."/".$k."/".$r["IMG"]." width=64 height=48>\n";
          
          // SOURCE 
          echo "    <td ".$bg_style."><a id='link_".$i."' href='".$url."' ".$mousein." ".$mouseout.">".$r["SOURCE"]."</a></td>\n";

          // UTC_START 
          echo "    <td ".$bg_style."><span id='UTC_START_".$i."'>".$k."</span></td>\n";

          // CONFIG
          echo "    <td ".$bg_style."><span id='CONFIG_".$i."'>".$r["CONFIG"]."</span></td>\n";

          // FREQ
          //echo "    <td ".$bg_style."><span id='FREQ_".$i."'>".$r["FREQ"]."</span></td>\n";

          // BW
          //echo "    <td ".$bg_style."><span id='BW_".$i."'>".$r["BW"]."</span></td>\n";

          // INTERGRATION LENGTH
          echo "    <td ".$bg_style."><span id='INT_".$i."'>".$r["INT"]."</span></td>\n";

          // PROC_FILEs
          if (array_key_exists("AQ_PROC_FILE", $r))
          {
            echo "    <td ".$bg_style.">";
              echo "<span id='AQ_PROC_FILE_".$i."'>".$r["AQ_PROC_FILE"]."</span> ";
              echo "<span id='BF_PROC_FILE_".$i."'>".$r["BF_PROC_FILE"]."</span> ";
              echo "<span id='BP_PROC_FILE_".$i."'>".$r["BP_PROC_FILE"]."</span>";
            echo "</td>\n";
          }
          else
          {
            echo "    <td ".$bg_style."><span id='PROC_FILE_".$i."'>".$r["PROC_FILE"]."</span></td>\n";
          }

          // ANNOTATION
          echo "    <td ".$bg_style." class=\"trunc\"><div><span id='annotation_".$i."'>".$r["annotation"]."</span></div></td>\n";

          echo "  </tr>\n";

        }
?>
    </table>
<?
    $this->closeBlockHeader();

    echo "</td></tr></table>\n";
  }

  /*************************************************************************************************** 
   *
   * Prints raw text to be parsed by the javascript XMLHttpRequest
   *
   ***************************************************************************************************/
  function printUpdateHTML($get)
  {
    $results = $this->getResultsArray($this->results_dir,
                                      $this->offset, $this->length, 
                                      $this->filter_type, $this->filter_value);

    $keys = array_keys($results);
    rsort($keys);

    $xml = "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
    $xml .= "<results>\n";
    for ($i=0; $i<count($keys); $i++) {
      $k = $keys[$i];
      $r = $results[$k];
      $rkeys = array_keys($r);
      $xml .= "<result>\n";
      $xml .= "<UTC_START>".$k."</UTC_START>\n";
      for ($j=0; $j<count($rkeys); $j++) {
        $rk = $rkeys[$j]; 
        if (($rk != "FRES_AR") && ($rk != "TRES_AR")) {
          $xml .= "<".$rk.">".htmlspecialchars($r[$rk])."</".$rk.">\n";
        }
      }
      $xml .= "</result>\n";
    }
    $xml .= "</results>\n";

    header('Content-type: text/xml');
    echo $xml;

  }

  function handleRequest()
  {
    if ($_GET["update"] == "true") {
      $this->printUpdateHTML($_GET);
    } else {
      $this->printHTML($_GET);
    }

  }

  function getResultsArray($results_dir, $offset=0, $length=0, $filter_type, $filter_value) 
  {
    $all_results = array();

    $observations = array();
    $dir = $results_dir;

    if (($filter_type == "") || ($filter_value == "")) 
    {
      $observations = getSubDirs($results_dir, $offset, $length, 1);
    } 
    else 
    {

      # get a complete list
      if ($filter_type == "UTC_START") {
        $cmd = "find ".$results_dir."/*".$filter_value."* -maxdepth 1 -name 'obs.info' -printf '%h\n' | awk -F/ '{print \$NF}' | sort -r";
      } else {
        $cmd = "find ".$results_dir." -maxdepth 2 -type f -name obs.info | xargs grep ".$filter_type." | grep ".$filter_value." | awk -F/ '{print $(NF-1)}' | sort -r";
      }
      $last = exec($cmd, $all_obs, $rval);
      $observations = array_slice($all_obs, $offset, $length);
    }


    for ($i=0; $i<count($observations); $i++) {

      $o = $observations[$i];
      $dir = $results_dir."/".$o;
      // read the obs.info file into an array 
      if (file_exists($dir."/obs.info")) {
        $arr = getConfigFile($dir."/obs.info");
        $all_results[$o]["SOURCE"] = $arr["SOURCE"];
        $all_results[$o]["CONFIG"] = $arr["CONFIG"];
        //$all_results[$o]["FREQ"] = sprintf("%5.2f",$arr["FREQ"]);
        //$all_results[$o]["BW"] = $arr["BW"];
        $all_results[$o]["PID"] = $arr["PID"];
        if (array_key_exists("PROC_FILE", $arr))
          $all_results[$o]["PROC_FILE"] = $arr["PROC_FILE"];
        else
        {
          $all_results[$o]["AQ_PROC_FILE"] = $arr["AQ_PROC_FILE"];
          $all_results[$o]["BF_PROC_FILE"] = $arr["BF_PROC_FILE"];
          $all_results[$o]["BP_PROC_FILE"] = $arr["BP_PROC_FILE"];
        }
        // these will only exist after this page has loaded and the values have been calculated once
        $all_results[$o]["INT"] = (array_key_exists("INT", $arr)) ? $arr["INT"] : "NA";
        $all_results[$o]["IMG"] = (array_key_exists("IMG", $arr)) ? $arr["IMG"] : "NA";
      }

      # if this observation is finished check if any of the observations require updating of the obs.info
      if (file_exists($dir."/obs.finished")) {
        $finished = 1;
      } else {
        $finished = 0;
      }

      # get the image bp or pvf 
      if (($all_results[$o]["IMG"] == "NA") || ($all_results[$o]["IMG"] == "")) 
      {
        $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
               "-name '*.fl.120x90.png' -printf '%f\n' ".
               "-o -name '*CH00.ad.160x120.png' -printf '%f\n' ".
               "| sort -n | head -n 1";
        $img = exec($cmd, $output, $rval);
        if (($rval == 0) && ($img != "")) {
          $all_results[$o]["IMG"] = $img;
        } else {
          $cmd = "find ".$dir." -mindepth 1 -maxdepth 1 -type f ".
                 "-name '*.FB.00.*png' -printf '%f\n' ".
                "| sort -n | head -n 1";
          $img = exec($cmd, $output, $rval);
          if (($rval == 0) && ($img != "")) {
            $all_results[$o]["IMG"] = $img;
          } else {
            $all_results[$o]["IMG"] = "../../../images/blankimage.gif";
          }
        }
        if ($finished) {
          system("perl -ni -e 'print unless /^IMG/' ".$results_dir."/".$o."/obs.info");
          system("echo 'IMG                 ".$img."' >> ".$results_dir."/".$o."/obs.info");
        }
      }
      # get the integration length 
      if (($all_results[$o]["INT"] == "NA") || ($all_results[$o]["INT"] <= 0))
      {
        if (($all_results[$o]["IMG"] != "NA") && ($all_results[$o]["IMG"] != "") &&
            (strpos($all_results[$o]["IMG"], "blank") === FALSE)) 
        {
          $int = $this->calcIntLength($o);
          $all_results[$o]["INT"] = $int;
        } else {
          $all_results[$o]["INT"] = "NA";
        } 
        if ($finished) {
          system("perl -ni -e 'print unless /^INT/' ".$results_dir."/".$o."/obs.info");
          system("echo 'INT              ".$int."' >> ".$results_dir."/".$o."/obs.info");
        }
      }
      if (file_exists($dir."/obs.txt")) {
        $all_results[$o]["annotation"] = file_get_contents($dir."/obs.txt");
      } else {
        $all_results[$o]["annotation"] = "";
      }

      if (file_exists($dir."/obs.processing")) {
        $all_results[$o]["processing"] = 1;
      } else {
        $all_results[$o]["processing"] = 0;
      }
    }

    return $all_results;
  }

  function calcIntLength($utc_start) 
  {
    $dir = $this->results_dir."/".$utc_start." ".$this->archive_dir."/".$utc_start;

    if (file_exists($this->results_dir."/".$utc_start."/all_candidates.dat"))
    {
      $cmd = "tail -n 1 ".$this->results_dir."/".$utc_start."/all_candidates.dat | awk '{print $3}'";
      $length = exec($cmd, $output, $rval);
      return sprintf("%5.0f", $length);
    }
    else if (file_exists ($this->results_dir."/".$utc_start."/cc.sum"))
    {
      $cmd = "find ".$this->archive_dir."/".$utc_start." -name '*.ac' | sort -n | tail -n 1";
      $output = array();
      $ac = exec($cmd, $output, $rval);

      $parts = explode("_", $ac);
      $bytes_to = $parts[count($parts)-1];

      $cmd = "grep BYTES_PER_SECOND ".$this->archive_dir."/".$utc_start."/obs.header | awk '{print $2}'";
      $output = array();
      $Bps = exec($cmd, $output, $rval);

      $length = $bytes_to / $Bps;
      return sprintf ("%5.0f", $length);
    }
    else
    {
      # try to find a TB/*_f.tot file
      $cmd = "find ".$this->results_dir."/".$utc_start."/TB -mindepth 1 -maxdepth 1 -type f -name '*_f.tot' | sort -n | tail -n 1";
      $tot = exec($cmd, $output, $rval);
      if ($tot != "")
      {
        $cmd = $this->cfg["SCRIPTS_DIR"]."/psredit -Q -c length ".$tot;
        $output = array();
        $result = exec($cmd, $output, $rval);
        list ($file, $length) = split(" ", $result);
        if ($length != "" && $rval == 0)
          return sprintf ("%5.0f",$length);
      }
    
      # try to find a 2*.ar file
      $cmd = "find ".$dir." -mindepth 2 -maxdepth 2 -type f -name '2*.ar' -printf '%f\n' | sort -n | tail -n 1";
      $ar = exec($cmd, $output, $rval);
      if ($ar != "")
      {
        $array = split("\.",$ar);
        $ar_time_str = $array[0];

        # if image is pvf, then it is a local time, convert to unix time
        $ar_time_unix = unixTimeFromGMTime($ar_time_str);
       
        # add ten as the 10 second image file has a UTC referring to the first byte of the file 
        $length = $ar_time_unix - unixTimeFromGMTime($utc_start);
      }

      return $length;
    }
  }
}
handledirect("results");
