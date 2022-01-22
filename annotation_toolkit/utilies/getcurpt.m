function [x,y] = getcurpt(axHandle)
%GETCURPT Get current point.
%   [X,Y] = GETCURPT(AXHANDLE) gets the x- and y-coordinates of
%   the current point of AXHANDLE.  GETCURPT compensates these
%   coordinates for the fact that get(gca,'CurrentPoint') returns
%   the data-space coordinates of the idealized left edge of the
%   screen pixel that the user clicked on.  For IPT functions, we
%   want the coordinates of the idealized center of the screen
%   pixel that the user clicked on.

%   Steven L. Eddins, March 1997
%   Copyright 1993-1998 The MathWorks, Inc.  All Rights Reserved.
%   $Revision: 1.2 $  $Date: 1997/11/24 15:55:45 $

pt = get(axHandle, 'CurrentPoint');
x = pt(1,1);
y = pt(1,2);

% What is the extent of the idealized screen pixel in axes
% data space?

axUnits = get(axHandle, 'Units');
set(axHandle, 'Units', 'pixels');
axPos = get(axHandle, 'Position');
set(axHandle, 'Units', axUnits);

axPixelWidth = axPos(3);
axPixelHeight = axPos(4);

axXLim = get(axHandle, 'XLim');
axYLim = get(axHandle, 'YLim');

xExtentPerPixel = abs(diff(axXLim)) / axPixelWidth;
yExtentPerPixel = abs(diff(axYLim)) / axPixelHeight;

x = x + xExtentPerPixel/2;
y = y + yExtentPerPixel/2;
