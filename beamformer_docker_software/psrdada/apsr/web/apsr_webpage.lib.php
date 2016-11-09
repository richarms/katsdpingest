<?PHP

if (!$_APSR_WEBPAGE_LIB_PHP) { $_APSR_WEBPAGE_LIB_PHP = 1;

class apsr_webpage 
{

  /* array of css pages requried */
  var $css = array("apsr.css");
  var $ejs = array();
  var $title = "APSR";
  var $callback_freq = 4000;
  var $doc_type = "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\" \"http://www.w3.org/TR/html4/loose.dtd\">";
  var $show_logo = 0;
  var $logo_text = "";

  function apsr_webpage() 
  {

  }

  function javaScriptCallBack()
  {

  }

  function printJavaScriptHead() 
  {

  }

  function printJavaScriptBody()
  {

  }

  function printHTML()
  {

  }

  function printUpdateHTML()
  {

  }

  function openBlockHeader($block_title)
  {
    echo "<table class='wrapper'>\n";
    if ($block_title != "") 
    {
      echo "  <tr><th class='wrapper'>".$block_title."</th></tr>\n";
      echo "  <tr><td class='wrapper_bot'>\n";
    }
    else
      echo "  <tr><td class='wrapper_all'>\n";

  }

  function closeBlockHeader()
  {
    echo "  </td></tr>\n";
    echo "</table>\n";
  }



} // END CLASS DEFINITION


#
# handle a direct inclusion of the specified class
#
function handleDirect($child_class) 
{

  // if this parameter is defined, output the HTML for the
  // specified pages
  if ($_GET["single"] == "true") 
  {
    $obj = new $child_class();

    echo $obj->doc_type."\n";
    echo "<html>\n";
    echo "  <head>\n";
    echo "    <title>".$obj->title."</title>\n";
    echo "    <link rel='shortcut icon' href='/apsr/images/apsr_favico.ico'/>\n";

    // css and javascript includes
    for ($i=0; $i<count($obj->css); $i++)
      echo "    <link rel='stylesheet' type='text/css' href='".$obj->css[$i]."'>\n";
    for ($i=0; $i<count($obj->ejs); $i++)
      echo "    <script type='text/javascript' src='".$obj->ejs[$i]."'></script>\n";

    // callbacks for javascript pollings
    if ($obj->javaScriptCallback() != "")
    {
      echo "    <script type='text/javascript'>\n";
      echo "      function poll_server()\n";
      echo "      {\n";
      echo "        ".$obj->javaScriptCallback()."\n";
      echo "        setTimeout('poll_server()', ".$obj->callback_freq.");\n";
      echo "      }\n";
      echo "    </script>\n";
    }

    // javascript head scripts
    if ($obj->printJavaScriptHead() != "")
      $obj->printJavaScriptHead();

    echo "  </head>\n";
  
    // if we need to run the callback
    if ($obj->javaScriptCallback() != "")
      echo "<body onload='poll_server()'>\n";
    else
      echo "<body>\n";
    
    echo "<div class='background'></div>\n";
    echo "<div class='Main'>\n";

    $obj->printJavaScriptBody();

    if ($obj->show_logo)
    {
      echo "<table cellspacing=0 cellpadding=10px border=0 width='100%'>\n";
      echo "  <tr>\n";
      echo "  <td width='210px' height='60px'><img src='/apsr/images/apsr_logo.png' width='200px' height='60px'></td>\n";
      echo "  <td align=left><font size='+2'>".$obj->logo_text."</font></td>\n";
      echo "  </tr>\n";
      echo "</table>\n";
    }

    if (method_exists($obj, printSideBarHTML))
    {
      echo "<table width='100%' cellpadding='10px' border=0>\n";
      echo "  <tr>\n";
      echo "    <td style='vertical-align: top; width: 250px'>\n";
      $obj->printSideBarHTML();
      echo "    </td>\n";
      echo "    <td style='vertical-align: top;'>\n";
      $obj->printHTML();
      echo "    </td>\n";
      echo "  </tr>\n";
      echo "</table>\n";
    }
    else
    {
      $obj->printHTML();
    }

    echo "</div>\n";
    echo "</body>\n";
    echo "</html>\n";

  } else if ($_GET["update"] == "true") {

    $obj = new $child_class();
    $obj->printUpdateHTML($_GET);

  } else {
    # do nothing :)
  }
}

} // _APSR_WEBPAGE_LIB_PHP

