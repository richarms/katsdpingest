<?PHP

include("apsr.lib.php");
include("apsr_webpage.lib.php");

class process_obs extends apsr_webpage 
{
  var $action = "none";
  var $utc_start = "none";
  var $annotation = "";
  var $inst = 0;
  var $errors = "";

  function process_obs()
  {
    apsr_webpage::apsr_webpage();

    $this->action = $_GET["action"];
    $this->utc_start = urldecode($_GET["utc_start"]);
    $this->annotation = $_GET["annotation"];
    $this->inst = new apsr();

    if ($this->action == "") 
    {
      $this->errors .= "<p>ERROR: action not set in GET parameters</p>\n";
    }
    if ($this->utc_start == "") 
    {
      $this->errors .= "<p>ERROR: utc_start not set in GET parameters</p>\n";
    }

    # Be sure that the Observation is legitimate
    if (!(preg_match("/^\d\d\d\d-\d\d-\d\d-\d\d:\d\d:\d\d$/", $this->utc_start))) {
      $this->errors .= "<p>ERROR: UTC_START did not match the expected YYYY-MM-DD-HH:MM:SS pattern</p>\n";
    }

    if (! (file_exists($this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start) &&
           file_exists($this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start))) {
      $this->errors .= "<p>ERROR: UTC_START dirs did not exist in results and/or archive dir(s)</p>\n";
    }
  }

  function printJavaScriptHead()
  {
?>
    <script type='text/javascript'>  

      function finish() 
      {
        window.opener.location.href=window.opener.location.href;
        window.opener.focus();
        window.close();
      }

      function finishParent() 
      {
        window.opener.history.back();
        window.close();
      }

    </script>
<?
  }

  /* HTML for this page */
  function printHTML() 
  {
?>
    <table border=0 cellspacing=10px>
      <tr>
        <td width='380px' height='60px'><img src='/apsr/images/apsr_logo.png' width='380px' height='60px'></td>
      </tr>
    </table>
<?
    if ($this->errors != "") 
    {
      echo $this->errors;
      return;
    }

    // show the annotation editing form
    if ($this->action == "annotate") 
    {
      $this->openBlockHeader("Annotate Observation: ".$this->utc_start);

      $file = $this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start."/obs.txt";

      $text = array();

      # If there is an existing annotation, read it
      if (file_exists($file)) {
        $handle = fopen($file, "r");
        while (!feof($handle)) {
          $buffer = fgets($handle, 4096);
          array_push($text, $buffer);
        }
        fclose($handle);
      }
?>
      <p>This text will be archived with the observation in a file named
         obs.txt</p>
      <form name="annotation" action="process_obs.lib.php" method="get">
      <p>
<?
      echo "      <textarea name='annotation' cols='80' rows='20' wrap='soft'>";
      for ($i=0; $i<count($text); $i++) {
        echo $text[$i];
      }
      echo "</textarea>\n";
?>
      </p>
      <input type="hidden" name="single" value="true">
      <input type="hidden" name="action" value="write_annotation">
      <input type="hidden" name="utc_start" value="<?echo $this->utc_start?>">
      <input type="submit" value="Save"></input>
      <input type="button" value="Close" onclick="finish()"></input>
      </form>
<?
      $this->closeBlockHeader();
    }

    // write the supplied annotation to the obs.txt file
    else if ($this->action == "write_annotation")
    {
      if ($this->annotation == "") 
      {
        echo "<p>ERROR: no annotation provided</p>";
        return;
      }

      $file = $this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start."/obs.txt";
      $fptr = fopen($file, "w");
      if (!$fptr)
      {
        echo "<p>ERROR: failed to write to file: ".$file."</p>";      
        return;
      }
      fwrite($fptr, $this->annotation);
      fclose($fptr);
      echo "<script type='text/javascript'>finish()</script>";
    }

    // delete the observation
    else if ($this->action == "delete")
    {
      $this->openBlockHeader("Deleting ".$this->utc_start);
      flush();

      $results_dir = $this->inst->config["SERVER_RESULTS_DIR"]."/".$this->utc_start;
      $archive_dir = $this->inst->config["SERVER_ARCHIVE_DIR"]."/".$this->utc_start;

      # get a list of the hosts on which this observation resides
      $cmd = "ls -1d /nfs/apsr??/apsr/archives/".$this->utc_start." | awk -F/ '{print $3}'";
      $line = exec($cmd, $output, $rval);
      $hosts = array();
      if ($rval == 0)
      {
        for ($i=0; $i<count($output); $i++)
        {
          $hosts[$i] = rtrim($output[$i]);
        }
      }

      $delete_ok = 1;
      $rval = 0;

      $cmd = "rm -rf ".$results_dir;
      echo "Deleting ".$results_dir."<br>\n";
      $line = exec($cmd, $output, $rval);
      if ($rval != 0) 
      {
        echo "<p><font color='red'>Problem ocurred whilst deleting ".
              $this->utc_start." in results dir:</font><br>\n";
        for ($j=0; $j<count($output); $j++)
        {
          echo $output[$i]."<BR>\n";
        }
        echo "</p>\n";
        flush();
        $delete_ok = 0;
      }

      $cmd = "rm -rf ".$archive_dir;
      echo "Deleting ".$archive_dir."<br>\n";
      $line = exec($cmd, $output, $rval);

      if ($rval != 0) 
      {
        echo "<p><font color='red'>Problem ocurred whilst deleting ".
              $this->utc_start." in archive dir:</font><br>\n";
        for ($j=0; $j<count($output); $j++) {
          echo $output[$i]."<BR>\n";
        }
        echo "</p>\n";
        flush();
        $delete_ok = 0;
      }

      for ($i=0; $i<count($hosts); $i++)
      {
        $host = $hosts[$i];
        $cmd = "ssh -o BatchMode=yes -l apsr ".$host." 'rm -rf ".
                $this->inst->config["CLIENT_ARCHIVE_DIR"]."/".$this->utc_start.  " ".
                $this->inst->config["CLIENT_RESULTS_DIR"]."/".$this->utc_start."'";
        echo "Deleting archives on ".$host."<br>\n";
        $line = exec($cmd, $output, $rval);

        if ($rval != 0) 
        {
          echo "<p><font color='red'>Problem ocurred whilst deleting ".
               $this->utc_start." on ".$host.":</font><br>\n";
          for ($j=0; $j<count($output); $j++) {
            echo $output[$i]."<BR>\n";
          }
          echo "</p>\n";
          flush();
          $delete_ok = 0;
        }
      }

      if ($delete_ok) {
        echo "    <script type='text/javascript'>\n";
        echo "      setTimeout('finishParent()',2000);\n";
        echo "    </script>\n";
      }
      $this->closeBlockHeader();
    }
    else
    {
      $this->openBlockHeader("Process Observation");
      echo "<p>ERROR: unrecognized action [".$this->action."]</p>\n";
      $this->closeBlockHeader();
    }
  }

}

handledirect("process_obs");
