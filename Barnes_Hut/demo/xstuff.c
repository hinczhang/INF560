#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#define DEFAULT_BGCOLOR "white"
#define DEFAULT_FGCOLOR "black"
#define DEFAULT_BDWIDTH 1
#define DEFAULT_FONT    "fixed"

typedef struct APP_PARAMS
{
  char *name;
  char **p_value_string;
} APP_PARAMS;

typedef struct XWIN 
{
  Window xid;
  Window parent;
  void *data;
  int (*event_handler)();
} XWIN;

/* Default parameter values */
char *theBGcolor = DEFAULT_BGCOLOR,
  *theFGcolor = DEFAULT_FGCOLOR,
  *theFont = DEFAULT_FONT,
  *theGeom_rsrc = NULL,
  *display_name = NULL,
  *theGeom_cline = NULL,
  *theGeom = NULL;

APP_PARAMS app_resources[] = {
  {"background", &theBGcolor},
  {"foreground", &theFGcolor},
  {"font", &theFont},
  {"geometry", &theGeom_rsrc}
};

int num_resources = sizeof(app_resources) / sizeof(APP_PARAMS);

/* List of command-line options */
APP_PARAMS app_options[] =
{
  {"-display", &display_name},
  {"-d", &display_name},
  {"-geometry", &theGeom_cline},
  {"-g", &theGeom_cline},
};

int num_options = sizeof(app_options) / sizeof(APP_PARAMS);

void usage(void);

Display *theDisplay;
GC theGC;
int AppDone = 0;
XEvent theEvent;
XFontStruct *theFontStruct;
unsigned long theBGpix, theFGpix;
char *theAppName = " ";
Window theMain;
XWindowAttributes MainXWA;
XContext xwin_context;   
XColor red;
XColor black;

void initapp(int argc, char **argv, int x, int y, 
	     unsigned width, unsigned height)
{
  int i, j;
  char *tmpstr;
  Colormap default_cmap;
  XColor color;
  int bitmask;
  XGCValues gcv;
  XWMHints xwmh;
  XSizeHints xsh;
  char default_geometry[80];

  AppDone = 0;

  if (argv != NULL){
    theAppName = argv[0];
    
    for(i=1; i<argc; i+=2) {
      for(j=1; j<num_options; j++){
	if (strcmp(app_options[j].name, argv[i]) == 0) {
	  *app_options[j].p_value_string = argv[i+1];
	  break;
	}
      }
      if (j>= num_options)
	usage();
    }
  }
    
  if((theDisplay = XOpenDisplay(display_name)) == NULL){
    fprintf(stderr, "%s: can't open display named %s\n", 
	    argv[0], XDisplayName(display_name));
    exit(1);
  }
  
  for (i=0; i<num_resources; i++){
    if((tmpstr = XGetDefault(theDisplay, theAppName,
			     app_resources[i].name)) != NULL)
      *app_resources[i].p_value_string = tmpstr;
  }
  
  if((theFontStruct = XLoadQueryFont(theDisplay, theFont)) == NULL) {
    fprintf(stderr, "%s: display %s cannot load font %s\n",
	    theAppName, DisplayString(theDisplay), theFont);
    exit(1);
  }

  default_cmap = DefaultColormap(theDisplay, DefaultScreen(theDisplay));
  if (XParseColor(theDisplay, default_cmap, theBGcolor,
		  &color) == 0 ||
      XAllocColor(theDisplay, default_cmap, &color) == 0 ){
    /* Use white background in case of failure */
    theBGpix = WhitePixel(theDisplay, DefaultScreen(theDisplay));
  }
  else {
    theBGpix = color.pixel;
  }
  if (XParseColor(theDisplay, default_cmap, theFGcolor,
		  &color) == 0 ||
      XAllocColor(theDisplay, default_cmap, &color) == 0 ){
    /* Use white background in case of failure */
    theFGpix = WhitePixel(theDisplay, DefaultScreen(theDisplay));
  }
  else {
    theFGpix = color.pixel;
  }

  /* allocate the set of colors we will want to use for the drawing. */
  int rc = XAllocNamedColor(theDisplay, default_cmap, "red", &red, &red);
  if (rc == 0) {
    fprintf(stderr, "XAllocNamedColor - failed to allocated 'red' color.\n");
    exit(1);
  }

  /* allocate the set of colors we will want to use for the drawing. */
  rc = XAllocNamedColor(theDisplay, default_cmap, "black", &black, &black);
  if (rc == 0) {
    fprintf(stderr, "XAllocNamedColor - failed to allocated 'black' color.\n");
    exit(1);
  }

  /* Fill out a XsizeHints structure to inform the window manager about
     the desired size and location of main window */
  xsh.flags = (PPosition | PSize | PMinSize);

  xsh.height = height;
  xsh.min_height = height;
  xsh.width = width;
  xsh.min_width = xsh.width;
  xsh.x = x;
  xsh.y = y;

  /* make default geometry string */
  sprintf(default_geometry, "%dx%d+%d+%d", xsh.width,
	  xsh.height, xsh.x, xsh.y);
  theGeom = default_geometry;

  /* Override the geometry if necessary */
  if(theGeom_cline != NULL) theGeom = theGeom_cline;
  else
    if(theGeom_rsrc != NULL) theGeom = theGeom_rsrc;

  /* process geometry specifications */
  bitmask = XGeometry(theDisplay, DefaultScreen(theDisplay),
		      theGeom, default_geometry, DEFAULT_BDWIDTH,
		      theFontStruct->max_bounds.width,
		      theFontStruct->max_bounds.ascent +
		      theFontStruct->max_bounds.descent,
		      1, 1, &(xsh.x), &(xsh.y),
		      &(xsh.width), &(xsh.height));

  /* Check the bitmask and set flags in XSizeHints structure */
  if (bitmask & (XValue | YValue)) xsh.flags |= USPosition;
  if (bitmask & (WidthValue | HeightValue))
    xsh.flags |= USSize;
  
  /* Create the main window */
  theMain = XCreateSimpleWindow(theDisplay,
			       DefaultRootWindow(theDisplay),
			       xsh.x, xsh.y, xsh.width, xsh.height,
			       DEFAULT_BDWIDTH, theFGpix, theBGpix);

  /*Set window manager properties */
  XSetStandardProperties(theDisplay, theMain, theAppName, 
			 theAppName, None, argv, argc, &xsh);

  /* Give other hints */
  xwmh.flags = (InputHint | StateHint);
  xwmh.input = True;
  xwmh.initial_state = NormalState;

  XSetWMHints(theDisplay, theMain, &xwmh);

  /* Create a graphics context ofr the main window */
  gcv.font = theFontStruct->fid;
  gcv.foreground = theFGpix;
  gcv.background = theBGpix;
  theGC = XCreateGC(theDisplay, theMain, 
		    (GCFont | GCForeground | GCBackground), &gcv);

  /* Select exposure events for the main window */
  XSelectInput(theDisplay, theMain, ExposureMask|StructureNotifyMask);

  /* Map the main window */
  XMapWindow(theDisplay, theMain);
  if(XGetWindowAttributes(theDisplay, theMain, &MainXWA) == 0){
    fprintf(stderr, "Error getting attributes of main\n");
    exit(2);
  }


  /* wait until the window is displayed */
  while(1) {
    XEvent e;
    XNextEvent(theDisplay, &e);
    if (e.type == MapNotify) {
      break;
    }
  }
}

void usage(void){
  fprintf(stderr, "%s [-display name] [-geometry geom]\n",
	  theAppName);
}

void xwin_init(void)
{
  xwin_context = XUniqueContext();
}

