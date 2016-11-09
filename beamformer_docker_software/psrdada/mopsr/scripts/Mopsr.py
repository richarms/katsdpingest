#!/usr/bin/env python

#
# MOPSR python module
#

import Dada, struct, math, time, os
import sys, cStringIO, traceback, StringIO
import numpy, fnmatch

import matplotlib
matplotlib.use('agg')
import pylab 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

def getConfig():
    
  config_file = Dada.DADA_ROOT + "/share/mopsr.cfg"
  config = Dada.readCFGFileIntoDict(config_file)

  ct_config_file = Dada.DADA_ROOT + "/share/mopsr_cornerturn.cfg"
  ct_config = Dada.readCFGFileIntoDict(ct_config_file)
  config.update(ct_config)

  return config

def unpack (nchan, nsamp, ndim, raw):
  nelement = nchan * nsamp * ndim
  format = str(nelement) + 'b'
  complex = numpy.array (struct.unpack_from(format, raw), dtype=numpy.float32).view(numpy.complex64)
  return complex.reshape((nchan, nsamp), order='F')

# extract data for histogramming
def extractChannel (channel, nchan, nsamp, dimtype, raw):

  real = []
  imag = []

  if dimtype == 'real' or dimtype == 'both':
    if channel == -1:
      real = raw.real.flatten()
    else:
      real = raw[channel,:].real

  if dimtype == 'imag' or dimtype == 'both':
    if channel == -1:
      imag = raw.imag.flatten()
    else:
      imag = raw[channel,:].imag
  
  return (real, imag)

def detectAndIntegrateInPlace (spectra, nchan, nsamp, ndim, raw):
  absolute = numpy.square(numpy.absolute(raw))
  spectra = numpy.sum(absolute, axis=0)

# square law detect, produce bandpass
def detectAndIntegrate (nchan, nsamp, ndim, raw):
  absolute = numpy.square(numpy.absolute(raw))
  spectrum = numpy.sum(absolute, axis=1)
  return spectrum

# square law detect, produce waterfall
def detect (nchan, nsamp, ndim, raw):
  absolute = numpy.square(numpy.absolute(raw))
  spectra = numpy.sum(absolute, axis=0)

# square law detect, produce waterfall
def detectTranspose (nchan, nsamp, ndim, raw):
  absolute = numpy.square(numpy.absolute(raw))
  return absolute

class InlinePlot (object):

  def __init__(self):
    self.xres = 240
    self.yres = 180
    self.log = False
    self.plain_plot = False
    self.title = ''
    self.xlabel = ''
    self.ylabel = ''
    self.imgdata = StringIO.StringIO()
    self.fig = matplotlib.figure.Figure(facecolor='black')
    self.dpi = self.fig.get_dpi()
    self.ax = []

  def setTranspose (self, val):
    self.log = val;

  def setLog (self, val):
    self.log = val;

  def setPlain (self, val):
    self.plain_plot = val;

  def setZap(self, val):
    self.zap = val;

  # change the resolution for this plot
  def setResolution (self, xres, yres):
    # if the resolution is different
    if ((xres != self.xres) and (yres != self.yres)):
      xinches = float(xres) / float(self.dpi)
      yinches = float(yres) / float(self.dpi)
      self.fig.set_size_inches((xinches, yinches))

      # save the new/current resolution
      self.xres = xres
      self.yres = yres

  def setLabels (self, title='', xlabel='', ylabel=''):
    self.title = title
    self.xlabel = xlabel
    self.ylabel = ylabel

  # start plotting a new imagea
  def openPlot (self, xres, yres, plain):

    # ensure plot is of the right size
    self.setResolution (xres, yres)

    # always add axes on a new plot [TODO check!]
    if plain:
      self.ax = self.fig.add_axes((0,0,1,1))
    else:
      self.ax = self.fig.add_subplot(1,1,1)

    set_foregroundcolor(self.ax, 'white')
    set_backgroundcolor(self.ax, 'black')

    if not plain:
      self.ax.set_title(self.title)
      self.ax.set_xlabel(self.xlabel)
      self.ax.set_ylabel(self.ylabel)

    self.ax.grid(False)

  def closePlot (self):
    FigureCanvas(self.fig).print_png(self.imgdata)
    self.fig.delaxes(self.ax)
    self.imgdata.seek(0)
    self.fig.clf()

  def getRawImage (self):
    return self.imgdata.buf

# plot a histogram:
class HistogramPlot (InlinePlot):

  def __init__(self):
    super(HistogramPlot, self).__init__()
    self.configure (-1)
    self.setLabels ('Histogram', '', '')

  def configure (self, channel):
    if channel == -1:
      self.setLabels ('Histogram all channels', '', '') 
    else:
      self.setLabels ('Histogram channel '+str(channel), '', '') 

  def plot (self, xres, yres, plain, real, imag, nbins):
    self.openPlot(xres, yres, plain)
    if len(real) > 0:
      self.ax.hist(real, nbins, range=[-128, 127], color='red', label='real', histtype='step')
    if len(imag) > 0:
      self.ax.hist(imag, nbins, range=[-128, 127], color='green', label='imag', histtype='step' )
    self.closePlot()

# plot a complex timeseries
class TimeseriesPlot (InlinePlot):

  def __init__(self):
    super(TimeseriesPlot, self).__init__()
    self.nsamps = 0
    self.configure (64, self.nsamps)
    self.samps = numpy.arange (self.nsamps)

  def configure (self, channel, nsamps):
    self.setLabels ('Timeseries Channel ' + str(channel), 'Time (samples)', 'States (Voltages)')
    if self.nsamps != nsamps:
      self.nsamps = nsamps
      self.samps = numpy.arange (self.nsamps)
 
  def plot (self, xres, yres, plain, real, imag):
    self.openPlot(xres, yres, plain)

    if len(real) > 0:
      self.ax.plot(self.samps, real, c='r', marker=',', linestyle='None', label='real')
    if len(imag) > 0:
      self.ax.plot(self.samps, imag, c='g', marker=',', linestyle='None', label='imag')

    self.ax.set_xlim((0, self.nsamps))
    self.ax.set_ylim((-128.0, 127.0))

    self.closePlot()

class BandpassPlot (InlinePlot):

  def __init__(self):
    super(BandpassPlot, self).__init__()
    self.setLabels ('Bandpass', 'Channel', 'Power')
    self.nchan = 0
    self.xvals = numpy.arange (self.nchan)
    self.configure (False, False, False, 0)

  def configure (self, log, zap, transpose, nchan):
    self.log = log
    self.zap = zap
    self.transpose = transpose
    if (self.nchan != nchan):
      self.nchan = nchan
      self.xvals = numpy.arange(self.nchan)

  def plot (self, xres, yres, plain, nchan, spectrum):
    self.openPlot (xres, yres, plain)
    if self.log: 
      self.ax.set_yscale ('log', nonposy='clip')
    else:
      self.ax.set_yscale ('linear')
    if self.zap:
      spectrum[0] = 0
    if self.transpose:
      self.ax.plot(spectrum, self.xvals, c='w')
      self.ax.set_ylim((0, nchan))
    else:
      ymin = numpy.amin(spectrum)
      ymax = numpy.amax(spectrum)
      if ymax == ymin:
        ymax = ymin + 1
        spectrum[0] = 1
      self.ax.plot(self.xvals, spectrum, c='w')
      self.ax.set_xlim((0, nchan))
      self.ax.set_ylim((ymin, ymax))

    self.closePlot()

class FreqTimePlot (InlinePlot):

  def __init__(self):
    super(FreqTimePlot, self).__init__()
    self.setLabels ('Waterfall', 'Time (sample)', 'Channel')
    self.configure (False, False, False)

  def configure (self,log, zap, transpose):
    self.log = log
    self.zap = zap
    self.transpose = transpose

  def plot (self, xres, yres, plain, spectra, nchan, nsamps):
    self.openPlot(xres, yres, plain)

    if numpy.amax(spectra) == numpy.amin(spectra):
      spectra[0][0] = 1

    if self.zap:
      spectra[0,:] = 0

    if self.log:
      vmax = numpy.log(numpy.amax(spectra))
      self.ax.imshow(spectra, extent=(0, nsamps, 0, nchan), aspect='auto', 
                     origin='lower', interpolation='nearest', norm=LogNorm(vmin=0.0001,vmax=vmax), 
                     cmap=cm.get_cmap('hot'))
    else:
      self.ax.imshow(spectra, extent=(0, nsamps, 0, nchan), aspect='auto', 
                     origin='lower', interpolation='nearest', 
                     cmap=cm.get_cmap('hot'))

    self.closePlot()


def printBandpass (nchan, spectra):
  for ichan in range(nchan):
    Dada.logMsg(0, 2, '[' + str(ichan) + '] = ' + str(spectra[ichan]))


# creates a figure of the specified size
def createFigure(xdim, ydim):

  fig = matplotlib.figure.Figure(facecolor='black')
  dpi = fig.get_dpi()
  curr_size = fig.get_size_inches()
  xinches = float(xdim) / float(dpi)
  yinches = float(ydim) / float(dpi)
  fig.set_size_inches((xinches, yinches))
  return fig


def set_foregroundcolor(ax, color):
  for tl in ax.get_xticklines() + ax.get_yticklines():
    tl.set_color(color)
  for spine in ax.spines:
    ax.spines[spine].set_edgecolor(color)
  for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_color(color)
  for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_color(color)
  ax.axes.xaxis.label.set_color(color)
  ax.axes.yaxis.label.set_color(color)
  ax.axes.xaxis.get_offset_text().set_color(color)
  ax.axes.yaxis.get_offset_text().set_color(color)
  ax.axes.title.set_color(color)
  lh = ax.get_legend()
  if lh != None:
    lh.get_title().set_color(color)
    lh.legendPatch.set_edgecolor('none')
    labels = lh.get_texts()
    for lab in labels:
      lab.set_color(color)
  for tl in ax.get_xticklabels():
    tl.set_color(color)
  for tl in ax.get_yticklabels():
    tl.set_color(color)


def set_backgroundcolor(ax, color):
     ax.patch.set_facecolor(color)
     ax.set_axis_bgcolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)

