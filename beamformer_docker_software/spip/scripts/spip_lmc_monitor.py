#!/usr/bin/env python

##############################################################################
#  
#     Copyright (C) 2015 by Andrew Jameson
#     Licensed under the Academic Free License version 2.1
# 
###############################################################################

#
# spip_lmc_monitor - 
#

import os, threading, sys, time, socket, select, signal, traceback, json, re
from spip.utils import core, sockets
from spip_smrb import SMRBDaemon

#################################################################
# host based monitoring thread
#   disk capacity
#   smrb capacities
#   load values
#   temperature values
#   fan speeds
#   gpu properties

# read disk capacity information for specified directories
def getDiskCapacity (dirs, dl):

  result = 0  
  disks = {}

  for dir in dirs:

    cmd = "df " + dir + " -B 1048576 -P | tail -n 1 | awk '{print $2,$3,$4}'"
    rval, lines = core.system (cmd, 3 <= dl)
    result += rval

    if rval == 0 and len(lines) == 1:
      parts = lines[0].split()
      disks[dir] = {"size": parts[0], "used": parts[1], "available": parts[2]}

  return result, disks

def getLoads (dl):

  result = 0
  loads = {}

  cmd = "uptime"
  rval, lines = core.system (cmd, 3 <= dl)

  if rval == 0 and len(lines) == 1:
    parts = lines[0].split("load average: ") 
    if len(parts) == 2:
      load_parts = parts[1].split(', ')
      if len(load_parts) == 3:
        loads = {"1min" : load_parts[0], "5min": load_parts[1], "15min": load_parts[2]}

  cmd = "nproc"
  rval, lines = core.system (cmd, 3 <= dl)
  if rval == 0 and len(lines) == 1:
    loads["ncore"] = lines[0]
  else: 
    loads["ncore"] = 0

  return result, loads


# request SMRB state from SMRB monitoring points
def getSMRBCapacity (stream_ids, quit_event, dl):

  smrbs = {}
  rval = 0

  for stream_id in stream_ids:

    if quit_event.isSet(): 
      continue
    port = SMRBDaemon.getDBMonPort (stream_id)
    sock = sockets.openSocket (dl, "localhost", port, 1)
    if sock:
      sock.send ("smrb_status\n")
      data = sock.recv(65536)
      smrbs[stream_id] = json.loads(data) 
      sock.close()

  return rval, smrbs 

def getIPMISensors (dl):

  sensors = {}

  cmd = "ipmitool sensor"
  (rval, lines) = core.system (cmd, 3 <= dl)

  if rval == 0 and len(lines) > 0:
    for line in lines:
      data = line.split("|")
      metric_name = data[0].strip().lower().replace("+", "").replace(" ", "_")
      value = data[1].strip()
      lower = data[6].strip()
      upper = data[7].strip()

      # Skip missing sensors
      if value == 'na':
        continue

      state = "na"
      metric_units = "na"
      metric_value = "na"

      # if sensor is a boolean value
      if re.search("(0x)", value):
        if value == "0x1":
          metric_value = "ok"
          metric_units = "na"
          state = "ok"
        else:
          metric_value = "fail"
          metric_units = "na"
          state = "fail"

      else:
        # Extract out a float value
        vmatch = re.search("([0-9.]+)", value)
        if not vmatch:
          continue

        metric_value = vmatch.group(1)
        metric_units = data[2].strip().replace("degrees C", "C")

        state = data[3]

      #lnr = float(data[4].strip())
      #lcr = float(data[5].strip())
      #lnc = float(data[6].strip())
      #unc = float(data[7].strip())
      #ucr = float(data[8].strip())
      #unr = float(data[9].strip())
      #state = "ok"
      #if value <= lnr or value >= unr:
      #  state = "non recoverable"
      #if value <= lcr or value >= ucr:
      #  state = "critical"
      #if value <= lnc or value >= unc:
      #  state = "non critical"

      sensors[metric_name] = { "value": metric_value, "units": metric_units, "state": state, "lower":lower, "upper":upper }

  return rval, sensors

class monThread (threading.Thread):

  def __init__(self, quit_event, streams, dl):
    threading.Thread.__init__(self)
    self.quit_event = quit_event
    self.streams = streams

  def run (self):

    print "run"

if __name__ == "__main__":
  test_dl = 2

  quit_event = threading.Event()

  core.logMsg(2, test_dl, "getting disk capcity for /")
  rval, disks = getDiskCapacity ("/", 3)
  core.logMsg(2, test_dl, "rval="+str(rval))
  core.logMsg(2, test_dl, "disks="+str(disks))

  core.logMsg(2, test_dl, "reading SMRB info")
  stream_ids = [0]
  rval, smrbs = getSMRBCapacity (stream_ids, quit_event, test_dl)
  core.logMsg(2, test_dl, "rval="+str(rval))
  core.logMsg(2, test_dl, "smrbs="+str(smrbs))

  core.logMsg(2, test_dl, "reading load info")
  rval, loads = getLoads (test_dl)
  core.logMsg(2, test_dl, "rval="+str(rval))
  core.logMsg(2, test_dl, "loads="+str(loads))

  core.logMsg(2, test_dl, "reading IPMI sensor info")
  rval, sensors = getIPMISensors (test_dl)
  core.logMsg(2, test_dl, "rval="+str(rval))
  core.logMsg(2, test_dl, "sensors="+str(sensors))

  sys.exit(0)
    

