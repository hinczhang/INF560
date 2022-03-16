#ifndef UI_H
#define UI_H

#include <X11/Xlib.h>
#include <X11/Xutil.h>

void simple_init(int x, int y, unsigned width, unsigned height);
void init_display(int argc, char ** argv, int x, int y, unsigned width, unsigned height);
void draw_rect(int x1, int y1, int x2, int y2);
void draw_point(int x, int y);
void draw_red_point(int x, int y);
void clear_display() ;
void close_display() ;
void flush_display() ;


extern Display *theDisplay;
extern GC theGC;
extern Window theMain;
extern XColor black;
extern XColor red;

#endif /* UI_H */
